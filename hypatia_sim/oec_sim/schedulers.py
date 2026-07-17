"""
Schedulers for the OEC downlink problem (OEC_RQ_NAC.pdf eq. 8).

Shared engine model (per scheduling slot, duration SLOT_S):
  * task data sits at its source satellite; within a slot it can flow over
    the current shortest-delay path to its destination GBS (per-packet
    propagation+transmission time << slot length, so flow-through is valid);
  * images available for transmission are limited by the on-board encoder
    pipeline (ENC_IMGS_PER_S, starts at task arrival);
  * capacity: each ISL carries ISL_RATE_BPS; each GBS has an aggregate
    receive budget GS_RATE_BPS shared by all tasks landing there.

Schedulers:
  * GreedyScheduler(depth=q)      — fixed depth, earliest-deadline-first
  * GreedyScheduler(depth=None)   — myopic adaptive depth at arrival
  * MPCScheduler                  — rolling-horizon MILP, jointly picks the
                                    compression depth x_{k,q} and the
                                    transmission schedule y over H slots
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from . import config as C


# ── shared helpers ────────────────────────────────────────────────────────────

def _edge_keys(topo, isl_index, t, task):
    """Capacity-edge keys used by task's path at slot t, or None if cut off."""
    path = topo.path(t, task.dst_gs, task.src_sat)
    if path is None:
        return None
    keys = []
    for u, v in zip(path[:-1], path[1:]):
        if v >= C.N_SATS:
            keys.append(('gs', v - C.N_SATS))
        else:
            keys.append(('isl', isl_index[(min(u, v), max(u, v))]))
    return keys


def _cap_bits(key):
    rate = C.GS_RATE_BPS if key[0] == 'gs' else C.ISL_RATE_BPS
    return rate * C.SLOT_S


def _record_delivery(task, t, images):
    if images <= 1e-9:
        return
    t_end = (t + 1) * C.SLOT_S
    task.delivered += images
    task.delivery_slots[t] = task.delivery_slots.get(t, 0.0) + images
    task.delivered_utility += (task.weight * task.freshness(t_end)
                               * C.UTILITY[task.depth] * images / task.n_images)


class _Base:
    name = 'base'

    def __init__(self, topo, tasks):
        self.topo = topo
        self.tasks = tasks
        self.isl_index = {(min(a, b), max(a, b)): l
                          for l, (a, b) in enumerate(topo.isl_pairs)}

    def run(self):
        """Simulate all slots; returns per-slot aggregate history."""
        hist = {'t_s': [], 'delivered_images': [], 'utility': [],
                'backlog_bits': [], 'n_active': []}
        for t in range(C.N_SLOTS):
            active = [k for k in self.tasks
                      if k.arrival_slot <= t and k.delivered < k.n_images - 1e-6]
            if active:
                self.decide(t, active)
            backlog = sum((k.n_images - k.delivered)
                          * C.PAYLOAD_B[k.depth or max(C.DEPTHS)] * 8
                          for k in self.tasks if k.arrival_slot <= t)
            hist['t_s'].append(t * C.SLOT_S)
            hist['delivered_images'].append(sum(k.delivered for k in self.tasks))
            hist['utility'].append(sum(k.delivered_utility for k in self.tasks))
            hist['backlog_bits'].append(backlog)
            hist['n_active'].append(len(active))
        return hist


# ── greedy baselines ──────────────────────────────────────────────────────────

class GreedyScheduler(_Base):
    """EDF greedy.  depth=q fixes the depth; depth=None picks it myopically
    at arrival (largest q whose payload fits the GBS budget until deadline,
    split evenly with the other active tasks at that GBS)."""

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
            _record_delivery(task, t, y)


# ── rolling-horizon MPC (eq. 8) ───────────────────────────────────────────────

class MPCScheduler(_Base):
    """Deterministic MPC with predicted contact/route traces and fixed
    shortest-delay routing.  At each decision epoch it solves the MILP (8)
    over H slots, executes the first slot, and re-plans on new arrivals or
    at latest every MPC_RESOLVE_EVERY slots."""

    name = 'mpc'

    def __init__(self, topo, tasks, horizon=C.MPC_HORIZON_SLOTS):
        super().__init__(topo, tasks)
        self.H = horizon
        self.plan = {}          # (kid, q, abs_slot) -> images
        self.last_solve = -10**9
        self.known = set()

    def decide(self, t, active):
        new = {k.kid for k in active} - self.known
        if new or t - self.last_solve >= C.MPC_RESOLVE_EVERY:
            self.known |= new
            self._solve(t, active)
            self.last_solve = t
        self._execute(t, active)

    # -- MILP over the horizon ------------------------------------------------
    def _solve(self, t, active):
        H = min(self.H, C.N_SLOTS - t)
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
            return
        xvars = [(k.kid, q) for k in active if k.depth is None
                 for q in C.DEPTHS]
        yi = {v: i for i, v in enumerate(yvars)}
        xi = {v: len(yvars) + i for i, v in enumerate(xvars)}
        n = len(yvars) + len(xvars)
        task_by_id = {k.kid: k for k in active}

        # objective (milp minimizes -> negate): utility + early-delivery bonus
        c = np.zeros(n)
        for (kid, q, tau), i in yi.items():
            k = task_by_id[kid]
            t_end = (t + tau + 1) * C.SLOT_S
            util = k.weight * k.freshness(t_end) * C.UTILITY[q] / k.n_images
            early = (C.MPC_LAMBDA_QUEUE * (H - tau)
                     * C.PAYLOAD_B[q] * 8 / 1e9)          # backlog term, Gbit
            c[i] = -(util + early)
        # backlog term also penalizes committing to a big payload
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

        for k in active:
            rem = k.n_images - k.delivered
            qs = [k.depth] if k.depth else C.DEPTHS
            # volume: sum_tau y[k,q,.] <= rem * x[k,q]
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
            # exactly one depth
            if k.depth is None:
                add([(xi[(k.kid, q)], 1.0) for q in C.DEPTHS], 1.0, 1.0)
            # encoder pipeline: cumulative y <= encoded - already delivered
            for tau in range(H):
                ent = [(yi[(k.kid, q, tt)], 1.0) for tt in range(tau + 1)
                       for q in qs if (k.kid, q, tt) in yi]
                if ent:
                    enc = k.encoded_by((t + tau + 1) * C.SLOT_S) - k.delivered
                    add(ent, -np.inf, max(enc, 0.0))

        # link capacities per (slot, edge)
        edge_users = {}
        for (kid, tau), keys in paths.items():
            for key in keys:
                edge_users.setdefault((tau, key), []).append(kid)
        for (tau, key), kids in edge_users.items():
            ent = []
            for kid in kids:
                k = task_by_id[kid]
                for q in ([k.depth] if k.depth else C.DEPTHS):
                    ent.append((yi[(kid, q, tau)], C.PAYLOAD_B[q] * 8.0))
            add(ent, -np.inf, _cap_bits(key))

        from scipy.sparse import csr_matrix
        A = csr_matrix((vals, (rows, cols)), shape=(r, n))
        integrality = np.zeros(n)
        ub_var = np.full(n, np.inf)
        for v, i in xi.items():
            integrality[i] = 1
            ub_var[i] = 1.0
        res = milp(c=c,
                   constraints=LinearConstraint(A, np.array(lbs), np.array(ubs)),
                   integrality=integrality,
                   bounds=Bounds(np.zeros(n), ub_var))
        self.plan = {}
        if res.x is None:
            return
        for (kid, q, tau), i in yi.items():
            if res.x[i] > 1e-6:
                self.plan[(kid, q, t + tau)] = float(res.x[i])
        # commit depth for tasks the plan actually starts transmitting
        for k in active:
            if k.depth is None:
                planned = {q: sum(v for (kid, q2, _), v in self.plan.items()
                                  if kid == k.kid and q2 == q)
                           for q in C.DEPTHS}
                best = max(planned, key=planned.get)
                if planned[best] > 1e-6:
                    k.depth = best
                    for key in list(self.plan):
                        if key[0] == k.kid and key[1] != best:
                            del self.plan[key]

    # -- execute first-slot decisions, re-checking real capacity ---------------
    def _execute(self, t, active):
        residual = {}
        for k in sorted(active, key=lambda k: k.deadline_s):
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
            _record_delivery(k, t, y)
