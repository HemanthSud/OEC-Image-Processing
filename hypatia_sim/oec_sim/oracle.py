"""
Offline HiGHS upper bound for the OEC downlink problem ("look into the
optimization software -- the upper bound should be in HiGHS").

Ignores per-slot ISL/GSL routing entirely and keeps only the per-(window,
GBS) aggregate receive budget. This is always a *relaxation* of the real
feasible region (Explore-agent finding: at the committed rates the GBS
aggregate budget is the only constraint that ever binds -- ISL utilization
tops out around 0.08 -- so dropping ISL/GSL rows only widens the feasible
set further), so both bounds below are valid upper bounds on every
scheduler's realized utility, including under 'fabric-limited' where ISL
capacity can matter (dropping those rows there makes the bound looser, not
invalid).

Two bounds, in increasing tightness / cost:
  lp_bound()  — depth choice relaxed to continuous x[k,q] in [0,1], slots
                aggregated into windows of `agg` (default 5) so a solved LP
                (not MILP) is enough. Always run; seconds.
  mip_bound() — full-resolution (no slot aggregation), integral x[k,q],
                real MILP via scipy.optimize.milp / HiGHS, time-limited.
                A time-limited run still returns a valid dual bound even
                if it doesn't close the gap (res.mip_dual_bound).
"""

import time

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

from . import config as C


def _windows(H_total, agg):
    """[(m, slot_start, slot_end_exclusive), ...] covering [0, H_total)."""
    return [(m, s, min(s + agg, H_total))
            for m, s in enumerate(range(0, H_total, agg))]


def _build(tasks, windows, integral):
    """Shared LP/MILP builder: GBS-aggregate-only relaxation described above.
    Returns (c, A, lbs, ubs, ub_var, integrality, index) with index giving
    the (kid,q,m)/(kid,q)/(kid,m) -> column maps."""
    n_m = len(windows)
    yi, xi, Yi = {}, {}, {}
    for k in tasks:
        for q in C.DEPTHS:
            for m in range(n_m):
                yi[(k.kid, q, m)] = len(yi)
    off1 = len(yi)
    for k in tasks:
        for q in C.DEPTHS:
            xi[(k.kid, q)] = off1 + len(xi)
    off2 = off1 + len(xi)
    for k in tasks:
        for m in range(n_m):
            Yi[(k.kid, m)] = off2 + len(Yi)
    n = off2 + len(Yi)

    c = np.zeros(n)
    task_by_id = {k.kid: k for k in tasks}
    for (kid, q, m), i in yi.items():
        k = task_by_id[kid]
        # Utility must be evaluated at the *most optimistic* time in the
        # window (its start) to stay a valid upper bound: crediting a unit
        # of y as if delivered at window end (as the capacity/encoder rows
        # below permissively assume it *could* be) would UNDER-value early
        # delivery and could make the bound tighter than truly achievable,
        # i.e. invalid. Window-start freshness/tardiness, window-end
        # capacity -- both directions stay generous.
        _, s_start, _ = windows[m]
        t_opt = s_start * C.SLOT_S
        phi = k.freshness(t_opt)
        util = k.weight * phi * C.UTILITY[q] / k.n_images
        late_s = max(0.0, t_opt - k.deadline_s)
        tardy = C.MPC_LAMBDA_LATE * k.weight * late_s / C.TARDINESS_REF_S / k.n_images
        c[i] = -(util - tardy)

    rows, cols, vals, lbs, ubs = [], [], [], [], []
    r = 0

    def add(entries, lb, ub):
        nonlocal r
        for i, v in entries:
            rows.append(r); cols.append(i); vals.append(v)
        lbs.append(lb); ubs.append(ub)
        r += 1

    ub_var = np.full(n, np.inf)

    for k in tasks:
        rem = k.n_images
        for q in C.DEPTHS:
            ent = [(yi[(k.kid, q, m)], 1.0) for m in range(n_m)]
            ent.append((xi[(k.kid, q)], -rem))
            add(ent, -np.inf, 0.0)
        add([(xi[(k.kid, q)], 1.0) for q in C.DEPTHS], 1.0, 1.0)

        prev = None
        for m in range(n_m):
            _, _, s_end = windows[m]
            enc = max(k.encoded_by(s_end * C.SLOT_S), 0.0)
            ub_var[Yi[(k.kid, m)]] = enc
            ent = [(Yi[(k.kid, m)], 1.0)]
            if prev is not None:
                ent.append((prev, -1.0))
            for q in C.DEPTHS:
                ent.append((yi[(k.kid, q, m)], -1.0))
            add(ent, 0.0, 0.0)
            prev = Yi[(k.kid, m)]

    gs_of = {k.kid: k.dst_gs for k in tasks}
    for m, s_start, s_end in windows:
        window_slots = s_end - s_start
        for g in range(C.N_GS):
            ent = [(yi[(k.kid, q, m)], C.PAYLOAD_B[q] * 8.0)
                  for k in tasks if gs_of[k.kid] == g for q in C.DEPTHS]
            if ent:
                add(ent, -np.inf, C.GS_RATE_BPS * C.SLOT_S * window_slots)

    A = csr_matrix((vals, (rows, cols)), shape=(r, n))
    integrality = np.zeros(n)
    if integral:
        for v, i in xi.items():
            integrality[i] = 1
            ub_var[i] = 1.0
    return c, A, np.array(lbs), np.array(ubs), ub_var, integrality


def lp_bound(topo, tasks, agg=5):
    t0 = time.perf_counter()
    windows = _windows(C.N_SLOTS, agg)
    c, A, lbs, ubs, ub_var, integrality = _build(tasks, windows, integral=False)
    # constraints mix equalities (lb==ub, the cumulative-Y rows) and
    # inequalities; scipy.optimize.linprog wants those pre-split into
    # A_eq/A_ub, so route the (depth-)relaxed problem through milp with
    # integrality=0 instead, which accepts one combined LinearConstraint.
    res = milp(c=c, constraints=LinearConstraint(A, lbs, ubs),
              integrality=np.zeros(len(c)), bounds=Bounds(np.zeros(len(c)), ub_var))
    wall = time.perf_counter() - t0
    obj = -float(res.fun) if res.x is not None else float('nan')
    return dict(objective=obj, wall_s=wall, agg=agg, n_windows=len(windows),
               n_vars=len(c), n_rows=A.shape[0], status=res.status)


def mip_bound(topo, tasks, time_limit=300, mip_rel_gap=1e-3):
    t0 = time.perf_counter()
    windows = _windows(C.N_SLOTS, 1)     # full resolution, no aggregation
    c, A, lbs, ubs, ub_var, integrality = _build(tasks, windows, integral=True)
    res = milp(c=c, constraints=LinearConstraint(A, lbs, ubs),
              integrality=integrality, bounds=Bounds(np.zeros(len(c)), ub_var),
              options={'time_limit': time_limit, 'mip_rel_gap': mip_rel_gap})
    wall = time.perf_counter() - t0
    obj = -float(res.fun) if res.x is not None else float('nan')
    dual = -float(res.mip_dual_bound) if hasattr(res, 'mip_dual_bound') and \
           res.mip_dual_bound is not None else obj
    return dict(objective=obj, dual_bound=dual, wall_s=wall,
               n_vars=len(c), n_rows=A.shape[0], status=res.status,
               mip_gap=getattr(res, 'mip_gap', None))


def analytic_ceiling(tasks):
    """3-line sanity bound: everything delivered instantly at the best
    depth. Catches sign/normalization bugs in the LP/MIP bounds above."""
    umax = max(C.UTILITY.values())
    return sum(k.weight * umax for k in tasks)


def report(topo, tasks, results, agg=5, mip_time_limit=120):
    lp = lp_bound(topo, tasks, agg=agg)
    ceiling = analytic_ceiling(tasks)
    L = []
    L.append('=' * 74)
    L.append('Offline HiGHS upper bound (routing-relaxed: only the '
             'per-window GBS aggregate')
    L.append('budget is enforced; ISL/GSL rows dropped -- valid because '
             'they never bind at')
    L.append('the committed rates, and dropping rows can only raise the '
             'bound further)')
    L.append('=' * 74)
    L.append(f'analytic ceiling (everything at best depth, instantly): '
             f'{ceiling:8.2f}')
    L.append(f'LP bound   (continuous depth, {agg}-slot windows, '
             f'{lp["n_vars"]} vars): {lp["objective"]:8.2f}   '
             f'({lp["wall_s"]:.1f}s, status={lp["status"]})')
    try:
        mip = mip_bound(topo, tasks, time_limit=mip_time_limit)
        gap_txt = (f'  mip_gap={mip["mip_gap"]:.3f}'
                   if mip.get('mip_gap') is not None else '')
        L.append(f'MILP bound (integral depth, full resolution, '
                 f'time_limit={mip_time_limit}s): '
                 f'incumbent {mip["objective"]:8.2f}  dual {mip["dual_bound"]:8.2f}'
                 f'{gap_txt}   ({mip["wall_s"]:.1f}s, status={mip["status"]})')
        bound = mip['dual_bound']
    except Exception as e:                          # pragma: no cover
        L.append(f'MILP bound: skipped ({e})')
        bound = lp['objective']
    L.append('')
    L.append(f'{"scheduler":16s} {"utility":>9s} {"gap to bound":>13s}')
    for name, res in results.items():
        u = res['hist']['utility'][-1]
        gap = (1 - u / bound) * 100 if bound > 0 else float('nan')
        L.append(f'  {name:16s} {u:9.2f} {gap:12.1f}%')
    L.append('=' * 74)
    return '\n'.join(L)
