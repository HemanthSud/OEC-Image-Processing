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
from .schedulers import _Base, _edge_keys, _cap_bits, _record_delivery


@dataclass
class Directive:
    epoch_slot: int
    budget_bits: dict = field(default_factory=dict)   # kid -> bits this macro-step
    admit: set = field(default_factory=set)            # kid -> may transmit
    priority: dict = field(default_factory=dict)        # kid -> aged weight


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
        n = off + len(xi)

        c = np.zeros(n)
        for (kid, q, m), i in yi.items():
            k = next(kk for kk in tasks if kk.kid == kid)
            _, s_start, _ = windows[m]
            t_opt = s_start * C.SLOT_S
            util = k.weight * k.freshness(t_opt) * C.UTILITY[q] / k.n_images
            c[i] = -util

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

        A = csr_matrix((vals, (rows, cols)), shape=(r, n))
        res = milp(c=c, constraints=LinearConstraint(A, np.array(lbs), np.array(ubs)),
                  integrality=np.zeros(n),
                  bounds=Bounds(np.zeros(n), np.full(n, np.inf)))
        budget = {k.kid: 0.0 for k in tasks}
        if res.x is not None:
            for (kid, q, m), i in yi.items():
                if m == 0 and res.x[i] > 1e-6:
                    budget[kid] += res.x[i] * C.PAYLOAD_B[q] * 8.0
        return budget


class HierarchicalMPCScheduler(_Base):
    """Two-level MPC: CoordinatorMPC (admission/drop/budget, slow) feeding a
    short-horizon per-task MILP (depth + send schedule, fast)."""

    name = 'mpc-hier'

    def __init__(self, topo, tasks, low_horizon=None):
        super().__init__(topo, tasks)
        self.H = low_horizon or C.HIER_LOW_HORIZON
        self.coordinator = CoordinatorMPC()
        self.directive = Directive(0)
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
            self.last_hi = t
        new = {k.kid for k in active} - self.known
        if new or t - self.last_lo >= C.MPC_RESOLVE_EVERY:
            self.known |= new
            self._solve_lower(t, [k for k in active if not k.dropped])
            self.last_lo = t
        self._execute(t, active)

    def _solve_lower(self, t, active):
        active = [k for k in active if k.kid in self.directive.admit]
        H = min(self.H, C.N_SLOTS - t)
        t0 = time.perf_counter()
        yvars, paths = [], {}
        for k in active:
            qs = [k.depth] if k.depth else C.DEPTHS
            for tau in range(H):
                keys = _edge_keys(self.topo, self.isl_index, t + tau, k)
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
        n = len(yvars) + len(xvars) + len(Yi)
        task_by_id = {k.kid: k for k in active}

        c = np.zeros(n)
        for (kid, q, tau), i in yi.items():
            k = task_by_id[kid]
            delay_s = self.static.delay_s(t + tau, k.dst_gs, k.src_sat)
            t_arrive = (t + tau + 1) * C.SLOT_S + delay_s
            phi = k.freshness(t_arrive)
            w = self.directive.priority.get(kid, k.weight)
            util = w * phi * C.UTILITY[q] / k.n_images
            late_s = max(0.0, t_arrive - k.deadline_s)
            tardy = C.MPC_LAMBDA_LATE * w * late_s / C.TARDINESS_REF_S / k.n_images
            backlog_bits = (k.n_images - k.delivered) * C.PAYLOAD_B[q] * 8
            backpressure = C.AGING_ETA * backlog_bits / 1e11 / max(k.n_images, 1)
            c[i] = -(util - tardy + backpressure)
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

        edge_users = {}
        for (kid, tau), keys in paths.items():
            for key in keys:
                edge_users.setdefault((tau, key), []).append(kid)
        for (tau, key), kids in edge_users.items():
            ent = []
            for kid in kids:
                k = task_by_id[kid]
                for q in ([k.depth] if k.depth else C.DEPTHS):
                    if (kid, q, tau) in yi:
                        ent.append((yi[(kid, q, tau)], C.PAYLOAD_B[q] * 8.0))
            if ent:
                add(ent, -np.inf, _cap_bits(key))

        A = csr_matrix((vals, (rows, cols)), shape=(r, n))
        integrality = np.zeros(n)
        for v, i in xi.items():
            integrality[i] = 1
            ub_var[i] = 1.0
        res = milp(c=c, constraints=LinearConstraint(A, np.array(lbs), np.array(ubs)),
                  integrality=integrality, bounds=Bounds(np.zeros(n), ub_var))
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
            keys = _edge_keys(self.topo, self.isl_index, t, k)
            if keys is None:
                continue
            for key in keys:
                residual.setdefault(key, _cap_bits(key))
            img_bits = C.PAYLOAD_B[k.depth] * 8
            y = min(y, k.encoded_by((t + 1) * C.SLOT_S) - k.delivered,
                    min(residual[key] for key in keys) / img_bits)
            if y <= 1e-9:
                continue
            for key in keys:
                residual[key] -= y * img_bits
            delay = self.static.delay_s(t, k.dst_gs, k.src_sat)
            _record_delivery(k, t, y, route_delay_s=delay)
