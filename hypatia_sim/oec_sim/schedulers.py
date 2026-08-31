"""
Schedulers for the OEC downlink problem (OEC_RQ_NAC.pdf eq. 8).

Shared engine model (per scheduling slot, duration SLOT_S):
  * task data sits at its source satellite; within a slot it can flow over
    a shortest-delay path to its destination GBS (per-packet
    propagation+transmission time << slot length, so flow-through is valid);
  * images available for transmission are limited by the on-board encoder
    pipeline (ENC_IMGS_PER_S, starts at task arrival);
  * capacity: each ISL carries ISL_RATE_BPS; each satellite-to-GBS downlink
    carries GSL_RATE_BPS; each GBS additionally has an aggregate receive
    budget GS_RATE_BPS shared by every flow terminating there.

Schedulers:
  * GreedyScheduler(depth=q)      — fixed depth, earliest-deadline-first,
                                     routes over the static geometric
                                     shortest-delay path (StaticRouter)
  * GreedyScheduler(depth=None)   — myopic adaptive depth at arrival
  * MPCScheduler                  — rolling-horizon MILP, jointly picks the
                                     compression depth x_{k,q} and the
                                     transmission schedule y over H slots.
                                     Routing defaults to the static path
                                     (route_mode='static'); route_mode='predictive'
                                     runs the Dr.-Liu-directed fixed point:
                                     MPC predicts per-edge congestion, Dijkstra
                                     recomputes the path at each predicted
                                     step, repeat (routing.py).
"""

import time

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

from . import config as C
from . import routing as R
from . import utility


# ── shared helpers ────────────────────────────────────────────────────────────

def _edge_keys_from_path(path, isl_index):
    """Capacity-edge keys for a node path [src_sat, ..., gs_node], or None."""
    if path is None:
        return None
    keys = []
    for u, v in zip(path[:-1], path[1:]):
        if v >= C.N_SATS:
            g = v - C.N_SATS
            keys.append(('gsl', u, g))
            keys.append(('gs', g))
        else:
            keys.append(('isl', isl_index[(min(u, v), max(u, v))]))
    return keys


def _edge_keys(topo, isl_index, t, task):
    """Static geometric-route capacity-edge keys at slot t, or None."""
    return _edge_keys_from_path(topo.path(t, task.dst_gs, task.src_sat),
                                isl_index)


def _iter_shares(keys):
    """Yield (edge_key, share) for a route entry.

    A single path is a list of edge keys, each carrying the task's whole
    flow (share 1.0). A multipath mix (twolevel.RouteMix) is a dict
    {edge_key: share in [0,1]}, which is what lets several candidate paths
    enter the depth MILP without adding any path variables to it: the
    capacity coefficient becomes b_q * 8 * share instead of b_q * 8.
    """
    if isinstance(keys, dict):
        return list(keys.items())
    return [(key, 1.0) for key in keys]


def _cap_bits(key):
    if key[0] == 'gs':
        rate = C.GS_RATE_BPS
    elif key[0] == 'gsl':
        rate = C.GSL_RATE_BPS
    else:
        rate = C.ISL_RATE_BPS
    return rate * C.SLOT_S


def _record_delivery(task, t, images, route_delay_s=0.0):
    if images <= 1e-9:
        return
    t_end = (t + 1) * C.SLOT_S + route_delay_s
    task.delivered += images
    task.delivery_slots[t] = task.delivery_slots.get(t, 0.0) + images
    task.delivered_utility += utility.accumulate(task, images, t_end)
    if task.first_delivery_slot is None:
        task.first_delivery_slot = t
    if task.delivered >= task.n_images - 1e-6 and task.completion_slot is None:
        task.completion_slot = t
    if t_end > task.deadline_s:
        task.late_images += images


class _Base:
    name = 'base'

    def __init__(self, topo, tasks):
        self.topo = topo
        self.tasks = tasks
        # Clear any per-task utility state. run_all hands every scheduler a
        # freshly generated task list, but a carried-over coverage accumulator
        # would silently corrupt the score rather than fail loudly, so start
        # from a clean slate unconditionally.
        for k in tasks:
            utility.reset(k)
        self.isl_index = {(min(a, b), max(a, b)): l
                          for l, (a, b) in enumerate(topo.isl_pairs)}
        self.static = R.StaticRouter(topo)
        self.solve_log = []          # list of dict rows, see instrument.py

    def run(self):
        """Simulate all slots; returns per-slot aggregate history."""
        hist = {'t_s': [], 'delivered_images': [], 'utility': [],
                'backlog_bits': [], 'n_active': [], 'n_dropped': []}
        for t in range(C.N_SLOTS):
            active = [k for k in self.tasks
                      if k.arrival_slot <= t and k.delivered < k.n_images - 1e-6
                      and not k.dropped]
            if active:
                self.decide(t, active)
            backlog = sum((k.n_images - k.delivered)
                          * C.PAYLOAD_B[k.depth or max(C.DEPTHS)] * 8
                          for k in self.tasks
                          if k.arrival_slot <= t and not k.dropped)
            hist['t_s'].append(t * C.SLOT_S)
            hist['delivered_images'].append(sum(k.delivered for k in self.tasks))
            arrived = [k for k in self.tasks if k.arrival_slot <= t]
            hist['utility'].append(utility.run_utility(arrived)[0])
            hist['backlog_bits'].append(backlog)
            hist['n_active'].append(len(active))
            hist['n_dropped'].append(sum(1 for k in self.tasks if k.dropped))
        return hist


# ── greedy baselines ──────────────────────────────────────────────────────────

class GreedyScheduler(_Base):
    """EDF greedy.  depth=q fixes the depth; depth=None picks it myopically
    at arrival (largest q whose payload fits the GBS budget until deadline,
    split evenly with the other active tasks at that GBS). Always routes
    over the static geometric shortest-delay path."""

    def __init__(self, topo, tasks, depth=None):
        super().__init__(topo, tasks)
        self.fixed_depth = depth
        self.name = f'greedy-fixed-{depth}' if depth else 'greedy-adaptive'

    def _choose_depth(self, t, task, active):
        if self.fixed_depth:
            return self.fixed_depth
        sharing = 1 + sum(1 for k in active
                          if k is not task and k.dst_gs == task.dst_gs
                          and k.delivered < k.n_images)
        budget = C.GS_RATE_BPS * max(task.deadline_s - t * C.SLOT_S, 0) / sharing
        for q in sorted(C.DEPTHS, reverse=True):
            if task.n_images * C.PAYLOAD_B[q] * 8 <= budget:
                return q
        return C.DEPTHS[0]

    def decide(self, t, active):
        residual = {}
        for task in sorted(active, key=lambda k: k.deadline_s):
            if task.depth is None:
                task.depth = self._choose_depth(t, task, active)
            keys = _edge_keys(self.topo, self.isl_index, t, task)
            if keys is None:
                continue
            for key in keys:
                residual.setdefault(key, _cap_bits(key))
            bits_ok = min(residual[key] for key in keys)
            img_bits = C.PAYLOAD_B[task.depth] * 8
            y = min(task.encoded_by((t + 1) * C.SLOT_S) - task.delivered,
                    bits_ok / img_bits)
            if y <= 1e-9:
                continue
            for key in keys:
                residual[key] -= y * img_bits
            delay = self.static.delay_s(t, task.dst_gs, task.src_sat)
            _record_delivery(task, t, y, route_delay_s=delay)


# ── rolling-horizon MPC (eq. 8) ───────────────────────────────────────────────

class MPCScheduler(_Base):
    """Deterministic MPC with predicted contact/route traces. At each
    decision epoch it solves the MILP (8) over H slots, executes the first
    slot, and re-plans on new arrivals or at latest every MPC_RESOLVE_EVERY
    slots.

    route_mode='static'     — today's behaviour: routes fixed at the
                               geometric shortest-delay path for the whole
                               solve (fast, one MILP per re-plan).
    route_mode='predictive' — Dr. Liu's directive: MPC predicts per-edge
                               congestion from its own plan, Dijkstra
                               recomputes the route at each predicted step,
                               repeated to a damped fixed point
                               (routing.py PredictiveRouter,
                               MPC_ROUTE_ITERS rounds).
    """

    name = 'mpc'

    def __init__(self, topo, tasks, horizon=None, route_mode='static'):
        super().__init__(topo, tasks)
        self.H = horizon or C.MPC_HORIZON_SLOTS
        self.route_mode = route_mode
        if route_mode == 'predictive':
            self.name = 'mpc-congestion'
            self.predictor = R.PredictiveRouter(topo, self.isl_index)
        self.plan = {}           # (kid, q, abs_slot) -> images
        self.plan_paths = {}     # (kid, abs_slot) -> keys, for _execute
        self.plan_delay = {}     # (kid, abs_slot) -> route delay s, for _execute
        self.last_solve = -10**9
        self.known = set()

    def decide(self, t, active):
        new = {k.kid for k in active} - self.known
        if new or t - self.last_solve >= C.MPC_RESOLVE_EVERY:
            self.known |= new
            self._solve(t, active)
            self.last_solve = t
        self._execute(t, active)

    # -- routing for one solve: geometric (iter 0) or predicted (iter > 0) ----
    def _paths_static(self, t, H, active):
        paths = {}
        for k in active:
            for tau in range(H):
                keys = _edge_keys(self.topo, self.isl_index, t + tau, k)
                if keys is not None:
                    paths[(k.kid, tau)] = keys
        return paths, None

    def _paths_predictive(self, t, H, active, load):
        rs = self.predictor.build(t, H, load)
        paths = {}
        for k in active:
            for tau in range(H):
                p = rs.path(tau, k.dst_gs, k.src_sat)
                keys = _edge_keys_from_path(p, self.isl_index)
                if keys is not None:
                    paths[(k.kid, tau)] = keys
        return paths, rs

    def _delay_for(self, t, tau, k, rs):
        if rs is not None:
            return rs.delay_s(tau, k.dst_gs, k.src_sat)
        return self.static.delay_s(t + tau, k.dst_gs, k.src_sat)

    # -- MILP over the horizon, with a fixed-point outer loop over routes ----
    def _solve(self, t, active):
        H = min(self.H, C.N_SLOTS - t)
        if H <= 0 or not active:
            self.plan, self.plan_paths, self.plan_delay = {}, {}, {}
            return

        n_rounds = C.MPC_ROUTE_ITERS if self.route_mode == 'predictive' else 1
        load, best, best_rs, best_paths = {}, None, None, None
        t_start = time.perf_counter()
        for it in range(max(1, n_rounds)):
            if self.route_mode == 'predictive' and it > 0:
                paths, rs = self._paths_predictive(t, H, active, load)
            else:
                paths, rs = self._paths_static(t, H, active)
            sol = self._solve_milp(t, H, active, paths, rs)
            self.solve_log.append(dict(
                t_s=t * C.SLOT_S, iter=it, n_active=len(active),
                n_vars=sol['n_vars'], n_rows=sol['n_rows'],
                status=sol['status'], objective=sol['objective'],
                wall_s=sol['wall_s']))
            if sol['status'] == 'optimal' and (
                    best is None or sol['objective'] > best['objective']):
                best, best_rs, best_paths = sol, rs, paths
            if self.route_mode != 'predictive' or n_rounds <= 1:
                break
            new_load = self._accumulate_load(t, sol['plan'], paths)
            eta = 1.0 / (it + 2)
            delta = self._load_delta(load, new_load)
            keys = set(load) | set(new_load)
            load = {e: (1 - eta) * load.get(e, 0.0) + eta * new_load.get(e, 0.0)
                    for e in keys}
            if delta < C.MPC_ROUTE_TOL:
                break
            if time.perf_counter() - t_start > C.MPC_REPLAN_TIME_BUDGET_S:
                break

        if best is None:
            self.plan, self.plan_paths, self.plan_delay = {}, {}, {}
            return
        self.plan = best['plan']
        # Execute against the *winning* iterate's routes. Re-deriving them here
        # from the post-loop `load` (as an earlier version did) could hand
        # _execute a different path set than the MILP that produced `best` was
        # solved against, so the executed capacity edges need not be the ones
        # the plan respected. Carrying best_paths/best_rs also skips a
        # redundant Dijkstra rebuild.
        self.plan_paths = {(kid, t + tau): keys
                           for (kid, tau), keys in best_paths.items()}
        # ...and score the delivery on the delay of the route actually used,
        # not the geometric one (they differ whenever route_mode='predictive'
        # and the fabric is congested enough for the router to detour).
        self.plan_delay = {(k.kid, t + tau): self._delay_for(t, tau, k, best_rs)
                           for k in active for tau in range(H)
                           if (k.kid, tau) in best_paths}
        # commit depth for tasks the plan actually starts transmitting
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

    def _accumulate_load(self, t, plan, paths):
        """Predicted bits crossing each (tau, edge) under this iterate's plan."""
        load = {}
        for (kid, q, tau_abs), y in plan.items():
            tau = tau_abs - t
            keys = paths.get((kid, tau))
            if not keys:
                continue
            bits = y * C.PAYLOAD_B[q] * 8
            for key, share in _iter_shares(keys):
                load[(tau, key)] = load.get((tau, key), 0.0) + bits * share
        return load

    @staticmethod
    def _load_delta(old, new):
        keys = set(old) | set(new)
        if not keys:
            return 0.0
        return max(abs(old.get(k, 0.0) - new.get(k, 0.0))
                   / max(_cap_bits(k[1]), 1.0) for k in keys)

    def _solve_milp(self, t, H, active, paths, rs, delay_of=None):
        """delay_of: optional {(kid, tau) -> seconds}, the flow-weighted
        effective delay of a multipath mix. Falls back to the router's
        single-path delay when absent."""
        t0 = time.perf_counter()
        def _delay(tau, k):
            if delay_of is not None and (k.kid, tau) in delay_of:
                return delay_of[(k.kid, tau)]
            return self._delay_for(t, tau, k, rs)
        yvars = []
        for k in active:
            qs = [k.depth] if k.depth else C.DEPTHS
            for tau in range(H):
                if (k.kid, tau) not in paths:
                    continue
                for q in qs:
                    yvars.append((k.kid, q, tau))
        if not yvars:
            return dict(plan={}, objective=0.0, status='infeasible',
                       n_vars=0, n_rows=0, wall_s=time.perf_counter() - t0)

        xvars = [(k.kid, q) for k in active if k.depth is None
                 for q in C.DEPTHS]
        yi = {v: i for i, v in enumerate(yvars)}
        xi = {v: len(yvars) + i for i, v in enumerate(xvars)}
        Yi = {(k.kid, tau): len(yvars) + len(xvars) + i
              for i, (k, tau) in enumerate((k, tau)
                                           for k in active for tau in range(H))}
        # Concave-coverage segments (utility.SegmentBlock). Indexing lambda by
        # q as well as j is what lets the quality term s_q multiply the
        # coverage gain while both stay linear. Inert when J == 1.
        seg = utility.SegmentBlock(yi, len(yvars) + len(xvars) + len(Yi),
                                   group_of=utility.stride_grouper())
        n = len(yvars) + len(xvars) + len(Yi) + len(seg)
        # Fairness: one continuous column carrying the maximin floor
        # min_k Uhat_k. Jain is a ratio of quadratics and is not
        # MILP-representable -- optimizing it would leave oracle.py with no
        # valid bound to compare against -- so the floor is what we optimize
        # and Jain is reported as a diagnostic only.
        umin_i = n if C.UTIL_W_FAIR else None
        if umin_i is not None:
            n += 1
        task_by_id = {k.kid: k for k in active}

        # objective (milp minimizes -> negate). The value terms come from
        # utility.y_coeff, the single definition shared with the realized
        # score, hier.py and oracle.py -- these four used to disagree. Value
        # uses freshness at the *arrival* time, i.e. slot-end plus this
        # route's propagation delay (Xuanhao's ask: a real delay-related
        # term, not just the freshness discount alone).
        # `early` and the x[k,q] cost below stay objective-only regularizers:
        # they were never part of the score, and legacy must preserve that.
        c = np.zeros(n)
        val = np.zeros(n)      # pure reported-score coefficients, no regularizers
        for (kid, q, tau), i in yi.items():
            k = task_by_id[kid]
            delay_s = _delay(tau, k)
            t_arrive = (t + tau + 1) * C.SLOT_S + delay_s
            util = utility.y_coeff(k, q, t_arrive)
            val[i] = util
            early = (C.MPC_LAMBDA_QUEUE * (H - tau)
                     * C.PAYLOAD_B[q] * 8 / 1e9)          # backlog term, Gbit
            tardy = utility.objective_tardiness_coeff(k, t_arrive)
            c[i] = -(util + early - tardy)
        seg.set_objective(
            c, task_by_id,
            lambda g: ((t + g[2] * C.UTIL_SEG_STRIDE + 1) * C.SLOT_S
                       + _delay(g[2] * C.UTIL_SEG_STRIDE, task_by_id[g[0]])),
            val=val)
        if umin_i is not None:
            c[umin_i] = -(C.UTIL_W_FAIR * max(len(self.tasks), 1))
        for (kid, q), i in xi.items():
            k = task_by_id[kid]
            rem = k.n_images - k.delivered
            c[i] = C.MPC_LAMBDA_QUEUE * H * rem * C.PAYLOAD_B[q] * 8 / 1e9

        rows, cols, vals, lbs, ubs = [], [], [], [], []
        r = 0

        def add(entries, lb, ub):
            nonlocal r
            for i, v in entries:
                rows.append(r); cols.append(i); vals.append(v)
            lbs.append(lb); ubs.append(ub)
            r += 1

        ub_var = np.full(n, np.inf)

        for k in active:
            rem = k.n_images - k.delivered
            qs = [k.depth] if k.depth else C.DEPTHS
            for q in qs:
                ent = [(yi[(k.kid, q, tau)], 1.0) for tau in range(H)
                       if (k.kid, q, tau) in yi]
                if not ent:
                    continue
                if k.depth is None:
                    ent.append((xi[(k.kid, q)], -rem))
                    add(ent, -np.inf, 0.0)
                else:
                    add(ent, -np.inf, rem)
            if k.depth is None:
                add([(xi[(k.kid, q)], 1.0) for q in C.DEPTHS], 1.0, 1.0)

            # O(H) cumulative-delivery / encoder-pipeline constraint (was
            # O(H^2): rebuilding the running sum at every tau blew the MILP
            # up to ~468k nonzeros for 64 active tasks at H=60). Y[k,tau] is
            # the cumulative images sent through tau; its own upper bound
            # (a variable bound, not a row) enforces the encoder pipeline.
            prev = None
            for tau in range(H):
                yi_tau = Yi[(k.kid, tau)]
                enc = max(k.encoded_by((t + tau + 1) * C.SLOT_S) - k.delivered, 0.0)
                ub_var[yi_tau] = enc
                ent = [(yi_tau, 1.0)]
                if prev is not None:
                    ent.append((prev, -1.0))
                for q in qs:
                    if (k.kid, q, tau) in yi:
                        ent.append((yi[(k.kid, q, tau)], -1.0))
                add(ent, 0.0, 0.0)
                prev = yi_tau

        # Coverage-segment linking + residual caps. The residual matters:
        # coverage already delivered before this horizon has consumed the
        # steep early segments, and without it the MPC re-earns segment-1
        # credit at every single re-plan.
        seg.add_rows(add, ub_var, yi, task_by_id)

        if umin_i is not None:
            # u_min <= Uhat_k for every active task. Uhat_k is CUMULATIVE
            # (already-realized utility as a constant plus the in-horizon
            # linear expression) -- using only the within-horizon gain would
            # make every late-admitted task look starved and turn the term
            # into noise.
            for k in active:
                const_k, scale_k = utility.fair_row_coeffs(k)
                ent = [(umin_i, 1.0)]
                ent += [(i, -scale_k * val[i])
                        for (kid, q, tau), i in yi.items() if kid == k.kid]
                ent += [(i, -scale_k * val[i]) for i in seg.columns_of(k.kid)]
                add(ent, -np.inf, const_k)

        # link capacities per (slot, edge) — only rows with >=1 user exist
        edge_users = {}
        for (kid, tau), keys in paths.items():
            for key, share in _iter_shares(keys):
                edge_users.setdefault((tau, key), []).append((kid, share))
        for (tau, key), users in edge_users.items():
            ent = []
            for kid, share in users:
                k = task_by_id[kid]
                for q in ([k.depth] if k.depth else C.DEPTHS):
                    if (kid, q, tau) in yi:
                        ent.append((yi[(kid, q, tau)],
                                    C.PAYLOAD_B[q] * 8.0 * share))
            if ent:
                add(ent, -np.inf, _cap_bits(key))

        A = csr_matrix((vals, (rows, cols)), shape=(r, n))
        integrality = np.zeros(n)
        for v, i in xi.items():
            integrality[i] = 1
            ub_var[i] = 1.0
        lb_var = np.zeros(n)
        if umin_i is not None:
            # Uhat_k can go negative once the tardiness/cost penalties exceed
            # the quality gain, and a floor pinned at 0 would then make the
            # row u_min <= Uhat_k infeasible.
            lb_var[umin_i] = -np.inf
        res = milp(c=c,
                   constraints=LinearConstraint(A, np.array(lbs), np.array(ubs)),
                   integrality=integrality,
                   bounds=Bounds(lb_var, ub_var))
        wall_s = time.perf_counter() - t0
        plan = {}
        status = 'optimal' if res.x is not None else 'infeasible'
        obj = -float(res.fun) if res.x is not None else -np.inf
        if res.x is not None:
            for (kid, q, tau), i in yi.items():
                if res.x[i] > 1e-6:
                    plan[(kid, q, t + tau)] = float(res.x[i])
        return dict(plan=plan, objective=obj, status=status,
                   n_vars=n, n_rows=r, wall_s=wall_s)

    # -- execute first-slot decisions, re-checking real capacity ---------------
    def _execute(self, t, active):
        residual = {}
        for k in sorted(active, key=lambda k: k.deadline_s):
            y = self.plan.pop((k.kid, k.depth, t), 0.0) if k.depth else 0.0
            if y <= 1e-9:
                continue
            keys = self.plan_paths.get((k.kid, t))
            if keys is None:
                keys = _edge_keys(self.topo, self.isl_index, t, k)
            if keys is None:
                continue
            shares = _iter_shares(keys)
            for key, _ in shares:
                residual.setdefault(key, _cap_bits(key))
            img_bits = C.PAYLOAD_B[k.depth] * 8
            room = [residual[key] / (img_bits * share)
                    for key, share in shares if share > 1e-12]
            y = min(y, k.encoded_by((t + 1) * C.SLOT_S) - k.delivered,
                    min(room) if room else 0.0)
            if y <= 1e-9:
                continue
            for key, share in shares:
                residual[key] -= y * img_bits * share
            delay = self.plan_delay.get(
                (k.kid, t), self.static.delay_s(t, k.dst_gs, k.src_sat))
            _record_delivery(k, t, y, route_delay_s=delay)
