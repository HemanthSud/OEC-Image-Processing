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
from . import utility


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
    # Coverage segments, indexed on windows ORACLE_SEG_AGG times coarser than
    # y. Coarsening only gives the relaxation MORE freedom (it loosens which
    # window a unit of coverage is credited in), so the result is still a
    # valid upper bound -- and it keeps the full-resolution MILP tractable.
    agg = max(1, int(C.ORACLE_SEG_AGG))
    seg = utility.SegmentBlock(yi, off2 + len(Yi),
                               group_of=lambda key: (key[0], key[1], key[2] // agg))
    n = off2 + len(Yi) + len(seg)
    # The maximin fairness floor is exactly LP-representable, which is the
    # whole reason it -- not Jain -- is what gets optimized: the bound can
    # carry it, so the gap table stays meaningful. Adding a non-negative term
    # to a maximization can only raise the optimum, so this stays a bound.
    umin_i = n if C.UTIL_W_FAIR else None
    if umin_i is not None:
        n += 1

    c = np.zeros(n)
    val = np.zeros(n)
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
        util = utility.y_coeff(k, q, t_opt)
        val[i] = util
        tardy = utility.objective_tardiness_coeff(k, t_opt)
        c[i] = -(util - tardy)
    # freshness at the *group* start: the most optimistic instant in the
    # group, keeping the bound generous in the same direction as above
    seg.set_objective(c, task_by_id,
                      lambda g: windows[min(g[2] * agg, n_m - 1)][1] * C.SLOT_S,
                      val=val)
    if umin_i is not None:
        c[umin_i] = -(C.UTIL_W_FAIR * max(len(tasks), 1))

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

    # the offline bound plans from scratch, so no coverage is pre-consumed
    seg.add_rows(add, ub_var, yi, task_by_id, rho_of=lambda k: 0.0)

    if umin_i is not None:
        for k in tasks:
            const_k, scale_k = utility.fair_row_coeffs(k, realized=0.0)
            ent = [(umin_i, 1.0)]
            ent += [(i, -scale_k * val[i])
                    for (kid, q, m), i in yi.items() if kid == k.kid]
            ent += [(i, -scale_k * val[i]) for i in seg.columns_of(k.kid)]
            add(ent, -np.inf, const_k)

    A = csr_matrix((vals, (rows, cols)), shape=(r, n))
    integrality = np.zeros(n)
    if integral:
        for v, i in xi.items():
            integrality[i] = 1
            ub_var[i] = 1.0
    lb_var = np.zeros(n)
    if umin_i is not None:
        lb_var[umin_i] = -np.inf
    return c, A, np.array(lbs), np.array(ubs), ub_var, integrality, lb_var


def lp_bound(topo, tasks, agg=5):
    t0 = time.perf_counter()
    windows = _windows(C.N_SLOTS, agg)
    c, A, lbs, ubs, ub_var, integrality, lb_var = _build(tasks, windows,
                                                        integral=False)
    # constraints mix equalities (lb==ub, the cumulative-Y rows) and
    # inequalities; scipy.optimize.linprog wants those pre-split into
    # A_eq/A_ub, so route the (depth-)relaxed problem through milp with
    # integrality=0 instead, which accepts one combined LinearConstraint.
    res = milp(c=c, constraints=LinearConstraint(A, lbs, ubs),
              integrality=np.zeros(len(c)), bounds=Bounds(lb_var, ub_var))
    wall = time.perf_counter() - t0
    obj = -float(res.fun) if res.x is not None else float('nan')
    return dict(objective=obj, wall_s=wall, agg=agg, n_windows=len(windows),
               n_vars=len(c), n_rows=A.shape[0], status=res.status)


def mip_bound(topo, tasks, time_limit=300, mip_rel_gap=1e-3):
    t0 = time.perf_counter()
    windows = _windows(C.N_SLOTS, 1)     # full resolution, no aggregation
    c, A, lbs, ubs, ub_var, integrality, lb_var = _build(tasks, windows,
                                                        integral=True)
    res = milp(c=c, constraints=LinearConstraint(A, lbs, ubs),
              integrality=integrality, bounds=Bounds(lb_var, ub_var),
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
    umax = utility.quality_max()
    per = sum(utility.weight(k) * C.UTIL_W_QUALITY * umax for k in tasks)
    # the fairness bonus is bounded by omega_F |K| * 1 (every task at its own
    # ceiling), so adding it keeps this a ceiling too
    return per + C.UTIL_W_FAIR * len(tasks) * 1.0


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
    if C.UTIL_W_FAIR:
        # How much of the bound is the fairness bonus rather than deliverable
        # value? Worth separating: omega_F |K| u_min can be up to omega_F |K|,
        # a fixed offset that would otherwise silently inflate every gap.
        # Dropping the u_min column can leave HiGHS numerically unhappy on the
        # ~52k-variable LP (status 4), so fall back to a coarser aggregation
        # and, failing that, say it was not obtainable rather than print a nan.
        lp0 = None
        for a in (agg, agg * 2, agg * 4):
            with C.config_override(UTIL_W_FAIR=0.0):
                cand = lp_bound(topo, tasks, agg=a)
            if cand['status'] == 0 and np.isfinite(cand['objective']):
                lp0 = cand
                break
        if lp0 is not None:
            # compare at the SAME aggregation, or the difference would mix
            # "fairness removed" with "relaxation coarsened"
            ref = lp if lp0['agg'] == agg else lp_bound(topo, tasks,
                                                        agg=lp0['agg'])
            note = ('' if lp0['agg'] == agg
                    else f", both at {lp0['agg']}-slot windows")
            L.append(f'LP bound   (fairness term removed{note}): '
                     f'{lp0["objective"]:8.2f}   '
                     f'(fairness contributes '
                     f'{ref["objective"] - lp0["objective"]:+.2f} of '
                     f'{ref["objective"]:.2f})')
        else:
            L.append('LP bound   (fairness term removed): not obtainable '
                     '(HiGHS numerical failure without the u_min column)')
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
    violations = []
    for name, res in results.items():
        u = res['hist']['utility'][-1]
        gap = (1 - u / bound) * 100 if bound > 0 else float('nan')
        L.append(f'  {name:16s} {u:9.2f} {gap:12.1f}%')
        if u > bound + 1e-6:
            violations.append((name, u))
    # A realized score above the bound means the bound is not a bound. Every
    # term added to the reported utility must also appear in _build's
    # objective, relaxed optimistically -- this single check catches
    # essentially any sign or normalization mistake in the utility terms.
    if violations:
        L.append('')
        L.append('!! BOUND VIOLATED -- the offline relaxation is below a '
                 'realized schedule.')
        L.append('!! Some term in the reported utility is missing from '
                 'oracle._build (see FORMULATION.md).')
        for name, u in violations:
            L.append(f'!!   {name}: realized {u:.4f} > bound {bound:.4f}')
    L.append('=' * 74)
    return '\n'.join(L)
