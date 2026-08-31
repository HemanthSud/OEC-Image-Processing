"""Two-level MPC, peer coupling: a routing MPC and a depth MPC as equals.

Dr. Liu: "Hierarchical MPC -- one MPC to select the best path, another one
[for the rest]." hier.py splits admission/budget from depth but routes
statically at BOTH levels, so the routing/depth split was never actually
built. This module is one of the two ways of building it (the other is
hier.HierRouteMPCScheduler); run_all --twolevel --hier-route reports both.

Here the two MPCs are peers on the same timescale:

  routing MPC   a min-cost multicommodity-flow LP over per-(task, path, tau)
                flow variables. Given the bit demand the depth MPC currently
                wants to move, it picks how to spread that demand across the
                candidate paths from routing.build_multi.

  depth MPC     the existing depth/schedule MILP, but with its capacity rows
                written against the routing MPC's fractional edge shares
                instead of one hard-coded path.

They exchange (demand -> mix -> demand) and iterate to a damped fixed point,
reusing the method-of-successive-averages loop shape the predictive router
already uses. Iteration 0 IS the flat MPC, and the loop keeps the
best-objective iterate, so mpc-2level can never plan worse than mpc.

Why the flow LP needs capacity rows: a linear delay objective with no
capacity would simply re-pick the shortest path, i.e. Dijkstra in a more
expensive wrapper. Congestion has to enter endogenously, through the duals
of the capacity rows -- that is what makes this a routing *optimizer* rather
than a repackaged shortest path.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

from . import config as C
from . import routing as R
from .schedulers import MPCScheduler, _edge_keys_from_path, _cap_bits


@dataclass
class RouteMix:
    """What the routing MPC hands the depth MPC. Peer of hier.Directive."""
    epoch_slot: int
    edge_share: dict = field(default_factory=dict)   # (kid,tau) -> {edge_key: share}
    delay_s: dict = field(default_factory=dict)      # (kid,tau) -> effective delay
    split: dict = field(default_factory=dict)        # (kid,tau) -> {path: theta}
    n_paths: dict = field(default_factory=dict)      # (kid,tau) -> paths used
    shortfall_bits: float = 0.0


class TwoLevelMPCScheduler(MPCScheduler):
    name = 'mpc-2level'

    def __init__(self, topo, tasks, horizon=None):
        super().__init__(topo, tasks, horizon=horizon, route_mode='static')
        self.name = 'mpc-2level'
        self.predictor = R.PredictiveRouter(topo, self.isl_index)
        self.route_rows = []          # for routes_<sched>.csv

    # -- candidate paths ------------------------------------------------------
    def _candidates(self, t, H_r, active, load):
        """{(kid, tau): [(path_tuple, edge_keys, delay_s), ...]} deduped."""
        sets = self.predictor.build_multi(t, H_r, load, K=C.MPC_ROUTE_NPATHS)
        out = {}
        for k in active:
            for tau in range(H_r):
                seen, cands = set(), []
                for rs in sets:
                    p = rs.path(tau, k.dst_gs, k.src_sat)
                    if p is None:
                        continue
                    key = tuple(p)
                    if key in seen:
                        continue
                    keys = _edge_keys_from_path(p, self.isl_index)
                    if keys is None:
                        continue
                    seen.add(key)
                    cands.append((key, keys,
                                  rs.delay_s(tau, k.dst_gs, k.src_sat)))
                if cands:
                    out[(k.kid, tau)] = cands
        return out

    # -- the routing MPC ------------------------------------------------------
    def _route_lp(self, t, H_r, active, demand_bits, cands):
        """min-cost multicommodity flow over the candidate paths."""
        t0 = time.perf_counter()
        fvars, shvars = [], []
        for (kid, tau), cl in cands.items():
            if demand_bits.get((kid, tau), 0.0) <= 1e-6:
                continue
            for pi in range(len(cl)):
                fvars.append((kid, tau, pi))
            shvars.append((kid, tau))
        if not fvars:
            return None
        fi = {v: i for i, v in enumerate(fvars)}
        si = {v: len(fvars) + i for i, v in enumerate(shvars)}
        n = len(fvars) + len(shvars)

        c = np.zeros(n)
        for (kid, tau, pi), i in fi.items():
            c[i] = cands[(kid, tau)][pi][2] / C.MPC2L_DELAY_REF_S
        for v, i in si.items():
            c[i] = C.MPC2L_SHORTFALL_PENALTY

        rows, cols, vals, lbs, ubs = [], [], [], [], []
        r = 0

        def add(entries, lb, ub):
            nonlocal r
            for i, v in entries:
                rows.append(r); cols.append(i); vals.append(v)
            lbs.append(lb); ubs.append(ub)
            r += 1

        # demand: everything the depth MPC wants moved is either routed or
        # explicitly short. The shortfall column is what keeps the LP feasible
        # no matter how congested the fabric gets.
        for (kid, tau) in shvars:
            ent = [(fi[(kid, tau, pi)], 1.0)
                   for pi in range(len(cands[(kid, tau)]))]
            ent.append((si[(kid, tau)], 1.0))
            add(ent, demand_bits[(kid, tau)], demand_bits[(kid, tau)])

        # capacity per (tau, edge): this is where congestion enters
        edge_users = {}
        for (kid, tau, pi), i in fi.items():
            for key in cands[(kid, tau)][pi][1]:
                edge_users.setdefault((tau, key), []).append(i)
        for (tau, key), idxs in edge_users.items():
            add([(i, 1.0) for i in idxs], -np.inf, _cap_bits(key))

        A = csr_matrix((vals, (rows, cols)), shape=(r, n))
        res = milp(c=c, constraints=LinearConstraint(A, np.array(lbs),
                                                     np.array(ubs)),
                   integrality=np.zeros(n),
                   bounds=Bounds(np.zeros(n), np.full(n, np.inf)))
        wall = time.perf_counter() - t0
        self.solve_log.append(dict(
            t_s=t * C.SLOT_S, level='route', n_active=len(active),
            n_vars=n, n_rows=r,
            status='optimal' if res.x is not None else 'infeasible',
            objective=float(res.fun) if res.x is not None else float('nan'),
            wall_s=wall))
        if res.x is None:
            return None

        mix = RouteMix(epoch_slot=t)
        by_kt = {}
        for (kid, tau, pi), i in fi.items():
            if res.x[i] > 1e-6:
                by_kt.setdefault((kid, tau), []).append((pi, float(res.x[i])))
        for (kid, tau), lst in by_kt.items():
            tot = sum(v for _, v in lst)
            if tot <= 1e-9:
                continue
            share, split, delay = {}, {}, 0.0
            for pi, v in lst:
                th = v / tot
                path, keys, d = cands[(kid, tau)][pi]
                split[path] = th
                delay += th * d
                for key in keys:
                    share[key] = share.get(key, 0.0) + th
            mix.edge_share[(kid, tau)] = share
            mix.split[(kid, tau)] = split
            mix.delay_s[(kid, tau)] = delay
            mix.n_paths[(kid, tau)] = len(lst)
        mix.shortfall_bits = sum(float(res.x[i]) for i in si.values())
        return mix

    # -- demand handed to the routing MPC ------------------------------------
    @staticmethod
    def _plan_depths(plan, active):
        """Depth each task is currently headed for, from the last plan."""
        out = {}
        for k in active:
            if k.depth is not None:
                out[k.kid] = k.depth
                continue
            per_q = {}
            for (kid, q, _tau), y in plan.items():
                if kid == k.kid:
                    per_q[q] = per_q.get(q, 0.0) + y
            out[k.kid] = (max(per_q, key=per_q.get) if per_q
                          else max(C.DEPTHS))
        return out

    def _desired_bits(self, t, H_r, active, depths):
        """What each task WANTS to move per step, not what it was already
        conceded.

        This distinction is the whole correctness of the coupling. Deriving
        the routing demand from the depth MILP's plan is circular: that plan
        was already made feasible against the static single paths, so the
        routing LP would face no contention, find the shortest path optimal
        for everything, and reproduce Dijkstra exactly -- which is precisely
        what it did before this was fixed (mpc-2level came out bit-for-bit
        equal to mpc even on a fabric with 30% of links oversubscribed).

        The honest demand is what the encoder can supply and the task still
        owes, capped per slot by the encoder pipeline.
        """
        per_slot_imgs = C.ENC_IMGS_PER_S * C.SLOT_S
        d = {}
        for k in active:
            q = depths.get(k.kid, max(C.DEPTHS))
            bits_per_img = C.PAYLOAD_B[q] * 8
            rem = max(k.n_images - k.delivered, 0.0)
            for tau in range(H_r):
                if rem <= 1e-9:
                    break
                enc = max(k.encoded_by((t + tau + 1) * C.SLOT_S)
                          - k.delivered, 0.0)
                want = min(per_slot_imgs, rem, enc)
                if want > 1e-9:
                    d[(k.kid, tau)] = want * bits_per_img
                rem -= want
        return d

    @staticmethod
    def _demand_delta(old, new):
        keys = set(old) | set(new)
        tot = sum(max(old.get(k, 0.0), new.get(k, 0.0)) for k in keys)
        if tot <= 0.0:
            return 0.0
        return sum(abs(old.get(k, 0.0) - new.get(k, 0.0)) for k in keys) / tot

    # -- the peer fixed point -------------------------------------------------
    def _solve(self, t, active):
        H = min(self.H, C.N_SLOTS - t)
        if H <= 0 or not active:
            self.plan, self.plan_paths, self.plan_delay = {}, {}, {}
            return
        H_r = min(H, C.MPC2L_ROUTE_HORIZON)
        t_start = time.perf_counter()

        # iteration 0 is exactly the flat MPC, which is both the warm start
        # and the guarantee that this scheduler can never plan worse than mpc
        paths, rs = self._paths_static(t, H, active)
        sol = self._solve_milp(t, H, active, paths, rs)
        self._log(t, 0, active, sol)
        best, best_paths, best_delay = sol, paths, None
        if sol['status'] != 'optimal':
            self.plan, self.plan_paths, self.plan_delay = {}, {}, {}
            return

        depths = self._plan_depths(sol['plan'], active)
        D = self._desired_bits(t, H_r, active, depths)
        load = {}
        for it in range(1, max(1, C.MPC2L_ITERS) + 1):
            cands = self._candidates(t, H_r, active, load)
            mix = self._route_lp(t, H_r, active, D, cands)
            if mix is None:
                break
            # steps past the routing horizon keep their static single path
            mixed = dict(paths)
            mixed.update(mix.edge_share)
            sol2 = self._solve_milp(t, H, active, mixed, rs,
                                    delay_of=mix.delay_s)
            self._log(t, it, active, sol2)
            if sol2['status'] == 'optimal':
                if sol2['objective'] > best['objective']:
                    best, best_paths, best_delay = sol2, mixed, dict(mix.delay_s)
                # re-derive desire at the depths this iterate settled on
                depths_new = self._plan_depths(sol2['plan'], active)
                D_new = self._desired_bits(t, H_r, active, depths_new)
                delta = self._demand_delta(D, D_new)
                eta = 1.0 / (it + 2)          # MSA damping, as in schedulers
                keys = set(D) | set(D_new)
                D = {k: (1 - eta) * D.get(k, 0.0) + eta * D_new.get(k, 0.0)
                     for k in keys}
                load = self._accumulate_load(t, sol2['plan'], mixed)
                self._record_routes(t, mix)
                if delta < C.MPC2L_TOL:
                    break
            if time.perf_counter() - t_start > C.MPC_REPLAN_TIME_BUDGET_S:
                break

        self.plan = best['plan']
        self.plan_paths = {(kid, t + tau): keys
                           for (kid, tau), keys in best_paths.items()}
        if best_delay is not None:
            self.plan_delay = {(kid, t + tau): d
                               for (kid, tau), d in best_delay.items()}
        else:
            self.plan_delay = {(k.kid, t + tau): self._delay_for(t, tau, k, rs)
                               for k in active for tau in range(H)
                               if (k.kid, tau) in best_paths}
        self._commit_depths(active)

    def _log(self, t, it, active, sol):
        self.solve_log.append(dict(
            t_s=t * C.SLOT_S, level='depth', iter=it, n_active=len(active),
            n_vars=sol['n_vars'], n_rows=sol['n_rows'], status=sol['status'],
            objective=sol['objective'], wall_s=sol['wall_s']))

    def _record_routes(self, t, mix):
        for (kid, tau), split in mix.split.items():
            if tau != 0:
                continue                      # only the executed slot
            for pi, (path, th) in enumerate(sorted(split.items(),
                                                   key=lambda kv: -kv[1])):
                self.route_rows.append(dict(
                    t_s=t * C.SLOT_S, kid=kid, path_id=pi, share=round(th, 6),
                    hops=len(path) - 1,
                    delay_s=round(mix.delay_s.get((kid, tau), 0.0), 6),
                    n_paths=mix.n_paths.get((kid, tau), 1)))

    def _commit_depths(self, active):
        for k in active:
            if k.depth is None:
                planned = {q: sum(v for (kid, q2, _), v in self.plan.items()
                                  if kid == k.kid and q2 == q)
                           for q in C.DEPTHS}
                bestq = max(planned, key=planned.get)
                if planned[bestq] > 1e-6:
                    k.depth = bestq
                    for key in list(self.plan):
                        if key[0] == k.kid and key[1] != bestq:
                            del self.plan[key]
