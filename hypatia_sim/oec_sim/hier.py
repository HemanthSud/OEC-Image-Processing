"""
Hierarchical MPC (Dr. Liu: "hierarchical MPC ... use one MPC to select the
best path and use another one [for depth]") plus the rotting-backlog fix
(a task whose freshness has decayed to ~0 still holds backlog and still
generates MILP variables every re-plan, but earns ~0 reward, so it is never
served and never removed -- see config.PHI_DROP / T_ABANDON_S / AGING_ETA).

Two levels, coupled by an explicit Directive/Feedback contract:

  upper (CoordinatorMPC)  — slow (every HIER_MACRO_SLOTS), coarse, an LP
                             (depth relaxed to continuous x[k,q]) over the
                             GBS-aggregate budget only (ISL/GSL dropped --
                             see oracle.py's docstring for why that's a
                             reasonable, cheap relaxation at these rates).
                             Decides admission, explicit drop, and a
                             per-task bit budget for the next macro-step.
  lower (fast, per-slot)  — today's per-task MILP (schedulers.py), but with
                             a short horizon (HIER_LOW_HORIZON), a hard
                             per-task budget_bits[k] cap handed down by the
                             upper level, and an added backpressure
                             (Lyapunov drift-plus-penalty) term so large
                             backlogs get cleared before they rot.

Budgets are upper bounds only (y <= budget), never targets, so the lower
level always has a feasible y=0 fallback under model mismatch between the
coarse upper-level capacity forecast and the true per-slot topology.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

from . import config as C
from . import routing as R
from . import utility
from .schedulers import (_Base, _edge_keys, _edge_keys_from_path, _cap_bits,
                         _iter_shares, _record_delivery)


def _lb_var(n, umin_i):
    """Column lower bounds: 0 everywhere except the maximin floor, which must
    be free to go negative once penalties exceed the quality gain."""
    lb = np.zeros(n)
    if umin_i is not None:
        lb[umin_i] = -np.inf
    return lb


@dataclass
class Directive:
    epoch_slot: int
    budget_bits: dict = field(default_factory=dict)   # kid -> bits this macro-step
    admit: set = field(default_factory=set)            # kid -> may transmit
    priority: dict = field(default_factory=dict)        # kid -> aged weight
    # Route directives (mpc-hier-route only). Empty dicts reproduce today's
    # static-routing behaviour exactly, which is what keeps mpc-hier itself
    # bit-for-bit unchanged.
    routes: dict = field(default_factory=dict)          # kid -> {edge_key: share}
    route_delay: dict = field(default_factory=dict)     # kid -> effective delay s


def _macro_windows(t0, horizon, agg):
    return [(m, t0 + s, min(t0 + s + agg, t0 + horizon))
            for m, s in enumerate(range(0, horizon, agg))]


class CoordinatorMPC:
    """Upper level: admission, drop, and per-task GBS-budget allocation."""

    def __init__(self):
        pass

    def solve(self, t, active):
        horizon = min(C.HIER_MACRO_SLOTS * 6, C.N_SLOTS - t)  # ~6 macro-steps
        if horizon <= 0 or not active:
            return Directive(t), set()
        windows = _macro_windows(t, horizon, C.HIER_MACRO_SLOTS)

        admit, drop = set(), set()
        for k in active:
            phi_now = k.freshness(t * C.SLOT_S)
            rem_frac = (k.n_images - k.delivered) / k.n_images
            if (phi_now < C.PHI_DROP and rem_frac > 0.02) or \
               (t * C.SLOT_S - k.deadline_s > C.T_ABANDON_S and rem_frac > 0.02):
                drop.add(k.kid)
                continue
            if C.ADMISSION_ON and k.depth is None and not k.rejected:
                cheapest = min(C.DEPTHS)
                budget_to_deadline = C.GS_RATE_BPS * max(
                    k.deadline_s - t * C.SLOT_S, 0)
                need = C.HIER_THETA_ADMIT * k.n_images * C.PAYLOAD_B[cheapest] * 8
                if budget_to_deadline < need:
                    k.rejected = True
                    continue
            admit.add(k.kid)

        tasks_lp = [k for k in active if k.kid in admit]
        budget = {k.kid: 0.0 for k in tasks_lp}
        if tasks_lp:
            budget = self._lp_allocate(t, tasks_lp, windows)

        priority = {k.kid: k.weight * (1.0 + C.AGING_ETA * max(
            0.0, t * C.SLOT_S - k.arrival_slot * C.SLOT_S) / (horizon * C.SLOT_S))
            for k in active}
        for kid in drop:
            budget.pop(kid, None)
            admit.discard(kid)
        return Directive(t, budget_bits=budget, admit=admit, priority=priority), drop

    def _lp_allocate(self, t, tasks, windows):
        """Continuous-depth LP over the near horizon, GBS-aggregate budget
        only; returns kid -> bits it may use in the *first* macro window."""
        n_m = len(windows)
        yi, xi = {}, {}
        for k in tasks:
            for q in C.DEPTHS:
                for m in range(n_m):
                    yi[(k.kid, q, m)] = len(yi)
        off = len(yi)
        for k in tasks:
            for q in C.DEPTHS:
                xi[(k.kid, q)] = off + len(xi)
        task_by_id = {k.kid: k for k in tasks}
        seg = utility.SegmentBlock(yi, off + len(xi))
        n = off + len(xi) + len(seg)

        def _t_opt(key):
            return windows[key[2]][1] * C.SLOT_S

        c = np.zeros(n)
        for (kid, q, m), i in yi.items():
            c[i] = -utility.y_coeff(task_by_id[kid], q, _t_opt((kid, q, m)))
        seg.set_objective(c, task_by_id, _t_opt)

        rows, cols, vals, lbs, ubs = [], [], [], [], []
        r = 0

        def add(entries, lb, ub):
            nonlocal r
            for i, v in entries:
                rows.append(r); cols.append(i); vals.append(v)
            lbs.append(lb); ubs.append(ub)
            r += 1

        for k in tasks:
            rem = k.n_images - k.delivered
            for q in C.DEPTHS:
                ent = [(yi[(k.kid, q, m)], 1.0) for m in range(n_m)]
                ent.append((xi[(k.kid, q)], -rem))
                add(ent, -np.inf, 0.0)
            add([(xi[(k.kid, q)], 1.0) for q in C.DEPTHS], 1.0, 1.0)
            for m in range(n_m):
                _, _, s_end = windows[m]
                enc = max(k.encoded_by(s_end * C.SLOT_S) - k.delivered, 0.0)
                ent = [(yi[(k.kid, q, mm)], 1.0)
                      for mm in range(m + 1) for q in C.DEPTHS]
                add(ent, -np.inf, enc)

        for m, s_start, s_end in windows:
            wslots = s_end - s_start
            for g in range(C.N_GS):
                ent = [(yi[(k.kid, q, m)], C.PAYLOAD_B[q] * 8.0)
                      for k in tasks if k.dst_gs == g for q in C.DEPTHS]
                if ent:
                    add(ent, -np.inf, C.GS_RATE_BPS * C.SLOT_S * wslots)

        ub_var = np.full(n, np.inf)
        seg.add_rows(add, ub_var, yi, task_by_id)

        A = csr_matrix((vals, (rows, cols)), shape=(r, n))
        res = milp(c=c, constraints=LinearConstraint(A, np.array(lbs), np.array(ubs)),
                  integrality=np.zeros(n),
                  bounds=Bounds(np.zeros(n), ub_var))
        budget = {k.kid: 0.0 for k in tasks}
        if res.x is not None:
            for (kid, q, m), i in yi.items():
                if m == 0 and res.x[i] > 1e-6:
                    budget[kid] += res.x[i] * C.PAYLOAD_B[q] * 8.0
        return budget


class RouteCoordinatorMPC:
    """Upper level of the hierarchical routing/depth split.

    Chooses a small frozen path set per task once per macro-epoch (10 min)
    over a long horizon, aggregated into macro-windows, and hands it down.
    The lower level then solves depth+schedule inside those paths every 5
    slots. This is the cheap end of the trade-off against twolevel.py, which
    re-solves routing on every re-plan.

    The macro-window graph is taken from the topology at the window's MIDDLE
    slot -- a forecast, and documented as one. Intersecting isl_ok across the
    window would be needlessly conservative at Kuiper spacing, where all
    2,312 +Grid links are feasible essentially all of the time.
    """

    def __init__(self, topo, isl_index):
        self.topo = topo
        self.isl_index = isl_index
        self.predictor = R.PredictiveRouter(topo, isl_index)

    def solve(self, t, tasks, budget_bits):
        horizon = min(C.HIER_ROUTE_HORIZON, C.N_SLOTS - t)
        if horizon <= 0 or not tasks:
            return {}, {}
        wins = _macro_windows(t, horizon, C.HIER_ROUTE_WINDOW)
        mids = [min((a + b) // 2, C.N_SLOTS - 1) for _m, a, b in wins]

        cands = {}
        for wi, mid in enumerate(mids):
            sets = self.predictor.build_multi(mid, 1, {},
                                              K=C.HIER_ROUTE_NPATHS)
            for k in tasks:
                seen, cl = set(), []
                for rs in sets:
                    p = rs.path(0, k.dst_gs, k.src_sat)
                    if p is None or tuple(p) in seen:
                        continue
                    keys = _edge_keys_from_path(p, self.isl_index)
                    if keys is None:
                        continue
                    seen.add(tuple(p))
                    cl.append((tuple(p), keys,
                               rs.delay_s(0, k.dst_gs, k.src_sat)))
                if cl:
                    cands[(k.kid, wi)] = cl
        if not cands:
            return {}, {}

        fvars, shvars = [], []
        for (kid, wi), cl in cands.items():
            for pi in range(len(cl)):
                fvars.append((kid, wi, pi))
            shvars.append((kid, wi))
        fi = {v: i for i, v in enumerate(fvars)}
        si = {v: len(fvars) + i for i, v in enumerate(shvars)}
        n = len(fvars) + len(shvars)
        by_id = {k.kid: k for k in tasks}

        c = np.zeros(n)
        for (kid, wi, pi), i in fi.items():
            c[i] = cands[(kid, wi)][pi][2] / C.MPC2L_DELAY_REF_S
        for (kid, wi), i in si.items():
            # Unrouted bits cost more for tasks whose value is decaying, so
            # the coordinator routes the perishable traffic first.
            # NOTE the scaling: the flow coefficient above is delay/DELAY_REF
            # ~ O(1) per bit, so the shortfall must be >> 1 per bit to act as
            # a big-M. An earlier version divided this by 1e9 (bits -> Gbit)
            # without normalizing the flow term the same way, which made
            # shorting ~5e5x CHEAPER than routing: the LP shorted everything,
            # returned zero routes, and mpc-hier-route silently degenerated
            # into plain mpc-hier.
            k = by_id[kid]
            phi = k.freshness(wins[wi][1] * C.SLOT_S)
            c[i] = C.MPC2L_SHORTFALL_PENALTY * k.weight * max(phi, 1e-3)

        rows, cols, vals, lbs, ubs = [], [], [], [], []
        r = 0

        def add(entries, lb, ub):
            nonlocal r
            for i, v in entries:
                rows.append(r); cols.append(i); vals.append(v)
            lbs.append(lb); ubs.append(ub)
            r += 1

        for k in tasks:
            b = budget_bits.get(k.kid, 0.0)
            ent = [(fi[(k.kid, wi, pi)], 1.0)
                   for wi in range(len(wins))
                   for pi in range(len(cands.get((k.kid, wi), [])))]
            if ent:
                add(ent, -np.inf, b)
        for (kid, wi) in shvars:
            ent = [(fi[(kid, wi, pi)], 1.0)
                   for pi in range(len(cands[(kid, wi)]))]
            ent.append((si[(kid, wi)], 1.0))
            b = budget_bits.get(kid, 0.0) / max(len(wins), 1)
            add(ent, b, b)

        edge_users = {}
        for (kid, wi, pi), i in fi.items():
            for key in cands[(kid, wi)][pi][1]:
                edge_users.setdefault((wi, key), []).append(i)
        for (wi, key), idxs in edge_users.items():
            wslots = wins[wi][2] - wins[wi][1]
            add([(i, 1.0) for i in idxs], -np.inf, _cap_bits(key) * wslots)

        A = csr_matrix((vals, (rows, cols)), shape=(r, n))
        res = milp(c=c, constraints=LinearConstraint(A, np.array(lbs),
                                                     np.array(ubs)),
                   integrality=np.zeros(n),
                   bounds=Bounds(np.zeros(n), np.full(n, np.inf)))
        if res.x is None:
            return {}, {}

        flow = {}
        for (kid, wi, pi), i in fi.items():
            if res.x[i] > 1e-6:
                path, keys, d = cands[(kid, wi)][pi]
                e = flow.setdefault(kid, {})
                cur = e.get(path, [0.0, keys, d])
                cur[0] += float(res.x[i])
                e[path] = cur
        routes, delays = {}, {}
        for kid, per_path in flow.items():
            top = sorted(per_path.items(), key=lambda kv: -kv[1][0])
            top = top[:max(1, C.HIER_ROUTE_KEEP)]
            tot = sum(v[0] for _p, v in top)
            if tot <= 1e-9:
                continue
            share, delay = {}, 0.0
            for _p, (f, keys, d) in top:
                th = f / tot
                delay += th * d
                for key in keys:
                    share[key] = share.get(key, 0.0) + th
            routes[kid] = share
            delays[kid] = delay
        return routes, delays


class HierarchicalMPCScheduler(_Base):
    """Two-level MPC: CoordinatorMPC (admission/drop/budget, slow) feeding a
    short-horizon per-task MILP (depth + send schedule, fast)."""

    name = 'mpc-hier'

    def __init__(self, topo, tasks, low_horizon=None):
        super().__init__(topo, tasks)
        self.H = low_horizon or C.HIER_LOW_HORIZON
        self.coordinator = CoordinatorMPC()
        self.route_coordinator = None          # set by mpc-hier-route
        self.directive = Directive(0)
        self.n_route_fallbacks = 0
        self.n_route_lookups = 0
        self.n_route_exec = 0
        self.n_route_exec_fallbacks = 0
        self.plan = {}
        self.last_hi = -10**9
        self.last_lo = -10**9
        self.known = set()

    def decide(self, t, active):
        if t - self.last_hi >= C.HIER_MACRO_SLOTS:
            self.directive, dropped = self.coordinator.solve(t, active)
            for k in active:
                if k.kid in dropped and not k.dropped:
                    k.dropped = True
                    k.dropped_slot = t
            if self.route_coordinator is not None:
                live = [k for k in active
                        if k.kid not in dropped and k.kid in self.directive.admit]
                routes, delays = self.route_coordinator.solve(
                    t, live, self.directive.budget_bits)
                self.directive.routes = routes
                self.directive.route_delay = delays
            self.last_hi = t
        new = {k.kid for k in active} - self.known
        if new or t - self.last_lo >= C.MPC_RESOLVE_EVERY:
            self.known |= new
            self._solve_lower(t, [k for k in active if not k.dropped])
            self.last_lo = t
        self._execute(t, active)

    def _route_for(self, t, tau, k):
        """Edge shares for (task, slot): the frozen directive route if it is
        still feasible, otherwise the static geometric one.

        The fallback counter is not just defensive coding -- it is the
        quantitative price of freezing routes for a whole macro-epoch, and
        the number that most directly separates this coupling from the peer
        one in twolevel.py. It is reported.
        """
        share = self.directive.routes.get(k.kid)
        if share:
            # Count the EXECUTED slot separately from horizon lookaheads.
            # Aggregating them understates survival badly (4.9% vs 14.4%),
            # because most lookups are for slots up to H ahead where a frozen
            # route has no chance -- and the two numbers answer different
            # questions: "was the frozen route actually usable?" vs "did the
            # MILP plan against routes that will still exist?".
            self.n_route_lookups += 1
            if tau == 0:
                self.n_route_exec += 1
            if self._share_feasible(t + tau, share):
                return share
            self.n_route_fallbacks += 1
            if tau == 0:
                self.n_route_exec_fallbacks += 1
        return _edge_keys(self.topo, self.isl_index, t + tau, k)

    def _share_feasible(self, t, share):
        if t >= C.N_SLOTS:
            return False
        for key in share:
            if key[0] == 'isl':
                if not self.topo.isl_ok[t, key[1]]:
                    return False
            elif key[0] == 'gsl':
                if not self.topo.gsl_ok[t, key[1], key[2]]:
                    return False
        return True

    def _solve_lower(self, t, active):
        active = [k for k in active if k.kid in self.directive.admit]
        H = min(self.H, C.N_SLOTS - t)
        t0 = time.perf_counter()
        yvars, paths = [], {}
        for k in active:
            qs = [k.depth] if k.depth else C.DEPTHS
            for tau in range(H):
                keys = self._route_for(t, tau, k)
                if keys is None:
                    continue
                paths[(k.kid, tau)] = keys
                for q in qs:
                    yvars.append((k.kid, q, tau))
        if not yvars:
            self.plan = {}
            self.solve_log.append(dict(t_s=t * C.SLOT_S, level='lo',
                                       n_active=len(active), n_vars=0, n_rows=0,
                                       status='infeasible', objective=0.0,
                                       wall_s=time.perf_counter() - t0))
            return

        xvars = [(k.kid, q) for k in active if k.depth is None for q in C.DEPTHS]
        yi = {v: i for i, v in enumerate(yvars)}
        xi = {v: len(yvars) + i for i, v in enumerate(xvars)}
        Yi = {(k.kid, tau): len(yvars) + len(xvars) + i
              for i, (k, tau) in enumerate((k, tau) for k in active for tau in range(H))}
        seg = utility.SegmentBlock(yi, len(yvars) + len(xvars) + len(Yi),
                                   group_of=utility.stride_grouper())
        n = len(yvars) + len(xvars) + len(Yi) + len(seg)
        umin_i = n if C.UTIL_W_FAIR else None
        if umin_i is not None:
            n += 1
        task_by_id = {k.kid: k for k in active}

        c = np.zeros(n)
        val = np.zeros(n)      # pure reported-score coefficients
        for (kid, q, tau), i in yi.items():
            k = task_by_id[kid]
            delay_s = self.directive.route_delay.get(
                kid, self.static.delay_s(t + tau, k.dst_gs, k.src_sat))
            t_arrive = (t + tau + 1) * C.SLOT_S + delay_s
            w = self.directive.priority.get(kid, k.weight)
            util = utility.y_coeff(k, q, t_arrive, w=w)
            val[i] = utility.y_coeff(k, q, t_arrive)   # unaged, for fairness
            tardy = utility.objective_tardiness_coeff(k, t_arrive) * (
                w / k.weight if k.weight else 1.0)
            backlog_bits = (k.n_images - k.delivered) * C.PAYLOAD_B[q] * 8
            backpressure = C.AGING_ETA * backlog_bits / 1e11 / max(k.n_images, 1)
            c[i] = -(util - tardy + backpressure)
        seg.set_objective(
            c, task_by_id,
            lambda g: ((t + g[2] * C.UTIL_SEG_STRIDE + 1) * C.SLOT_S
                       + self.directive.route_delay.get(
                           g[0],
                           self.static.delay_s(t + g[2] * C.UTIL_SEG_STRIDE,
                                               task_by_id[g[0]].dst_gs,
                                               task_by_id[g[0]].src_sat))),
            w_of=lambda k: self.directive.priority.get(k.kid, k.weight))
        # the fairness row needs UNAGED score coefficients, so recompute them
        # separately from the aged ones the objective uses
        for (g, j), i in seg.li.items():
            kk = task_by_id[g[0]]
            val[i] = utility.seg_coeff(
                kk, g[1],
                (t + g[2] * C.UTIL_SEG_STRIDE + 1) * C.SLOT_S
                + self.directive.route_delay.get(
                    g[0], self.static.delay_s(t + g[2] * C.UTIL_SEG_STRIDE,
                                              kk.dst_gs, kk.src_sat)), j)
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

            # upper-level budget cap: total bits sent this low-level solve
            budget = self.directive.budget_bits.get(k.kid)
            if budget is not None:
                ent = [(yi[(k.kid, q, tau)], C.PAYLOAD_B[q] * 8.0)
                      for tau in range(H) for q in qs if (k.kid, q, tau) in yi]
                if ent:
                    add(ent, -np.inf, budget)

        seg.add_rows(add, ub_var, yi, task_by_id)

        if umin_i is not None:
            for k in active:
                const_k, scale_k = utility.fair_row_coeffs(k)
                ent = [(umin_i, 1.0)]
                ent += [(i, -scale_k * val[i])
                        for (kid, q, tau), i in yi.items() if kid == k.kid]
                ent += [(i, -scale_k * val[i]) for i in seg.columns_of(k.kid)]
                add(ent, -np.inf, const_k)

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
        res = milp(c=c, constraints=LinearConstraint(A, np.array(lbs), np.array(ubs)),
                  integrality=integrality,
                  bounds=Bounds(_lb_var(n, umin_i), ub_var))
        wall_s = time.perf_counter() - t0
        self.plan = {}
        status = 'optimal' if res.x is not None else 'infeasible'
        obj = -float(res.fun) if res.x is not None else 0.0
        if res.x is not None:
            for (kid, q, tau), i in yi.items():
                if res.x[i] > 1e-6:
                    self.plan[(kid, q, t + tau)] = float(res.x[i])
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
        self.solve_log.append(dict(t_s=t * C.SLOT_S, level='lo',
                                   n_active=len(active), n_vars=n, n_rows=r,
                                   status=status, objective=obj, wall_s=wall_s))

    def _execute(self, t, active):
        residual = {}
        for k in sorted(active, key=lambda k: k.deadline_s):
            if k.dropped:
                continue
            y = self.plan.pop((k.kid, k.depth, t), 0.0) if k.depth else 0.0
            if y <= 1e-9:
                continue
            keys = self._route_for(t, 0, k)
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
            # score on the route actually used, not the geometric one
            delay = self.directive.route_delay.get(
                k.kid, self.static.delay_s(t, k.dst_gs, k.src_sat))
            _record_delivery(k, t, y, route_delay_s=delay)


class HierRouteMPCScheduler(HierarchicalMPCScheduler):
    """Hierarchical routing/depth split: a slow routing MPC over a long
    horizon freezes a small path set per macro-epoch; the existing fast depth
    MILP then solves inside those paths.

    The counterpart to twolevel.TwoLevelMPCScheduler, which instead re-solves
    routing as a peer on every re-plan. Setting HIER_ROUTE_ON = False makes
    this reproduce mpc-hier exactly.
    """

    name = 'mpc-hier-route'

    def __init__(self, topo, tasks, low_horizon=None):
        super().__init__(topo, tasks, low_horizon=low_horizon)
        self.name = 'mpc-hier-route'
        if C.HIER_ROUTE_ON:
            self.route_coordinator = RouteCoordinatorMPC(topo, self.isl_index)
