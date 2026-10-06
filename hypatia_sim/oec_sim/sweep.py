"""
Systematic congestion sweep across GS_RATE_BPS regimes x task seeds x every
scheduler -- closes the "MPC's advantage shown only by a one-off manual run
that isn't saved in the repo" gap (meeting_script_oec.txt section 6). The
crossover this produces (best FIXED depth flips with load; MPC/MPC-hier
track the upper envelope in every regime) is the headline figure.

Run from hypatia_sim/:
    python3 -m oec_sim.sweep                  # default grid, ~10-20 min
    python3 -m oec_sim.sweep --quick           # 2 rates x 2 seeds, smoke test
"""

import csv
import os
import time

import numpy as np

from . import config as C
from . import topology as T
from . import utility
from . import tasks as TK
from .schedulers import GreedyScheduler, MPCScheduler

RATES_MBPS = [0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
SEEDS = [42, 43, 44, 45, 46]

# Routing only has leverage where the FABRIC is scarce. Under gbs-limited the
# per-GBS aggregate budget is >=12x tighter than any ISL a route could cross,
# so ISL contention is mathematically unreachable and every routing variant
# reproduces the static one bit-for-bit -- we verified exactly that. The
# coupling grid therefore sweeps the ISL rate under fabric-limited instead.
ISL_RATES_MBPS = [0.5, 0.75, 1.0, 1.5, 2.0]
COUPLING_SEEDS = [42, 43, 44]

OUT_DIR = os.path.join(C.OUT_DIR, 'sweep')


def _row(rate_mbps, seed, sched, tasks, hist, tot_img, wall_s, obj=None,
         rate_key='gs_rate_mbps'):
    img = sum(k.delivered for k in tasks)
    n_viol = sum(1 for k in tasks if k.dropped or k.completion_slot is None
                or (k.completion_slot + 1) * C.SLOT_S > k.deadline_s)
    total, terms, jain, umin = utility.run_utility(tasks)
    solve_s = sum(x['wall_s'] for x in getattr(obj, 'solve_log', []) or [])
    return {
        rate_key: rate_mbps, 'seed': seed, 'scheduler': sched,
        'images_delivered': round(img, 1),
        'delivery_frac': round(img / tot_img, 4),
        'utility': round(hist['utility'][-1], 4),
        'utility_quality': round(terms['quality'], 4),
        'utility_tardy': round(terms['tardiness'], 4),
        'utility_cost': round(terms['cost'], 4),
        'u_min': round(umin, 4), 'jain': round(jain, 4),
        'n_dropped': sum(1 for k in tasks if k.dropped),
        'violation_frac': round(n_viol / len(tasks), 4),
        'n_route_fallbacks': getattr(obj, 'n_route_fallbacks', 0),
        'solve_wall_s': round(solve_s, 2),
        'wall_s': round(wall_s, 2),
    }


def run(rates_mbps=RATES_MBPS, seeds=SEEDS, quick=False):
    if quick:
        rates_mbps, seeds = rates_mbps[:2], seeds[:2]
    os.makedirs(OUT_DIR, exist_ok=True)
    topo = T.build_topology()          # independent of rate/seed -- build once
    rows = []

    for rate in rates_mbps:
        with C.config_override(GS_RATE_BPS=rate * 1e6):
            for seed in seeds:
                with C.config_override(RNG_SEED=seed):
                    makers = ([('Flat-MPC', lambda tp, tk: MPCScheduler(tp, tk)),
                              ('greedy-adaptive',
                               lambda tp, tk: GreedyScheduler(tp, tk, depth=None))]
                              + [(f'greedy-fixed-{q}',
                                 lambda tp, tk, q=q: GreedyScheduler(tp, tk, depth=q))
                                for q in C.DEPTHS])
                    for name, make in makers:
                        tk = TK.generate_tasks(topo)
                        tot_img = sum(k.n_images for k in tk)
                        sched = make(topo, tk)
                        t0 = time.time()
                        hist = sched.run()
                        wall_s = time.time() - t0
                        rows.append(_row(rate, seed, name, tk, hist,
                                         tot_img, wall_s, obj=sched))
                        print(f'  rate={rate:4.2f}Mbps seed={seed:3d} '
                             f'{name:16s} util={hist["utility"][-1]:7.2f} '
                             f'({wall_s:5.1f}s)', flush=True)

    # --quick is a 2x2 smoke test, not a result. Writing it to results.csv
    # silently replaces the committed 180-row grid with 24 rows, which is
    # exactly the kind of loss you only notice much later.
    path = os.path.join(OUT_DIR,
                        'results_quick.csv' if quick else 'results.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {os.path.relpath(path)}')

    if not quick:
        try:
            _plot(rows)
        except ImportError:
            print('matplotlib not available, skipping sweep plot')
    return rows


def _plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .plots import SCHED_COLOR, GRID

    scheds = sorted({r['scheduler'] for r in rows})
    rates = sorted({r['gs_rate_mbps'] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in scheds:
        util_by_rate = []
        for rate in rates:
            vals = [r['utility'] for r in rows
                    if r['scheduler'] == s and r['gs_rate_mbps'] == rate]
            util_by_rate.append((np.mean(vals), np.std(vals)))
        means = np.array([u for u, _ in util_by_rate])
        stds = np.array([sd for _, sd in util_by_rate])
        col = SCHED_COLOR.get(s, '#6f6e66')
        lw = 2.4 if s == 'OrbitFlow' else 1.4
        ax.plot(rates, means, color=col, lw=lw, marker='o', ms=4, label=s)
        ax.fill_between(rates, means - stds, means + stds, color=col, alpha=0.15)
    ax.set_xlabel('GBS aggregate rate (Mbps per station)')
    ax.set_ylabel('utility (mean over seeds, shaded = ±1 std)')
    ax.set_title('MPC vs. best fixed depth across load regimes -- the '
                 'crossover is the headline result', fontsize=10)
    ax.grid(**GRID)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, '..', 'plots', 'sweep_utility_vs_load.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'wrote {os.path.relpath(out)}')


def run_couplings(isl_rates_mbps=ISL_RATES_MBPS, seeds=COUPLING_SEEDS,
                  quick=False):
    """HISTORICAL coupling grid under `fabric-limited` (Mbps-scale, pre-
    2026-09-21 scheduler set). Kept as-is for provenance/comparison against
    `results_routing.csv`'s already-committed numbers. mpc-hier/mpc-hier-
    route are archived (Part 13) -- this now runs Flat-MPC and Two-Level-MPC
    only; OrbitFlow isn't included here since this grid predates its rename
    and was never run with it. For the new Gbps-scale 'realistic' sensitivity
    sweep (Xuanhao's ask, all three of Flat-MPC/OrbitFlow/Two-Level-MPC), see
    run_realistic_isl/gsl/gbs() below instead.
    """
    from .twolevel import TwoLevelMPCScheduler
    if quick:
        isl_rates_mbps, seeds = isl_rates_mbps[:2], seeds[:1]
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    out_name = 'results_routing_quick.csv' if quick else 'results_routing.csv'
    with C.config_override(scenario='fabric-limited'):
        topo = T.build_topology()
        for rate in isl_rates_mbps:
            with C.config_override(ISL_RATE_BPS=rate * 1e6):
                for seed in seeds:
                    with C.config_override(RNG_SEED=seed):
                        makers = [
                            ('Flat-MPC', lambda tp, tk: MPCScheduler(tp, tk)),
                            ('Two-Level-MPC',
                             lambda tp, tk: TwoLevelMPCScheduler(tp, tk)),
                        ]
                        for name, make in makers:
                            tk = TK.generate_tasks(topo)
                            tot_img = sum(k.n_images for k in tk)
                            sched = make(topo, tk)
                            t0 = time.time()
                            hist = sched.run()
                            rows.append(_row(rate, seed, name, tk, hist,
                                             tot_img, time.time() - t0,
                                             obj=sched,
                                             rate_key='isl_rate_mbps'))
                            print(f'  isl={rate:4.2f}Mbps seed={seed:3d} '
                                  f'{name:16s} util={hist["utility"][-1]:7.2f} '
                                  f'({time.time() - t0:5.1f}s)', flush=True)
    path = os.path.join(OUT_DIR, out_name)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {os.path.relpath(path)}')
    return rows


# ── Part 11: three-way capacity sensitivity under 'realistic' ───────────────
# Xuanhao's draft (OrbitFlow, "Comments for Hemanth" #2): fix two of
# {ISL, GSL, GBS} at the 'realistic' scenario's nominal point and vary the
# third across his own grids. Only the three surviving MPC schedulers
# (matches his sensitivity-table headers exactly -- greedy/PPO aren't in
# these tables). "We do not need to run all of these settings... select a
# few representative scenarios" -- so these default to a single seed; pass
# more via `seeds=` once a representative subset is chosen. MAX_TASKS=16000
# under 'realistic' makes Two-Level-MPC expensive (~99s per run at just 150
# tasks, superlinear) -- expect these to need the NCSU server and real
# wall-clock time, not a quick local run.
REALISTIC_ISL_GRID_GBPS = [1, 5, 10, 20]
REALISTIC_GSL_GRID_GBPS = [0.5, 1, 2, 5]
REALISTIC_GBS_GRID_GBPS = [2, 5, 10, 20]
REALISTIC_SWEEP_SEEDS = [42]


def _realistic_sensitivity(rate_key, grid_gbps, override_key, seeds, quick):
    from .twolevel import TwoLevelMPCScheduler
    if quick:
        grid_gbps, seeds = grid_gbps[:2], seeds[:1]
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    out_name = f'results_realistic_{rate_key}{"_quick" if quick else ""}.csv'
    with C.config_override(scenario='realistic'):
        topo = T.build_topology()
        for rate in grid_gbps:
            with C.config_override(**{override_key: rate * 1e9}):
                for seed in seeds:
                    with C.config_override(RNG_SEED=seed):
                        makers = [
                            ('Flat-MPC', lambda tp, tk: MPCScheduler(tp, tk)),
                            ('OrbitFlow', lambda tp, tk: MPCScheduler(
                                tp, tk, route_mode='predictive')),
                            ('Two-Level-MPC',
                             lambda tp, tk: TwoLevelMPCScheduler(tp, tk)),
                        ]
                        for name, make in makers:
                            tk = TK.generate_tasks(topo)
                            tot_img = sum(k.n_images for k in tk)
                            sched = make(topo, tk)
                            t0 = time.time()
                            hist = sched.run()
                            wall_s = time.time() - t0
                            rows.append(_row(rate, seed, name, tk, hist,
                                             tot_img, wall_s, obj=sched,
                                             rate_key=f'{rate_key}_gbps'))
                            print(f'  {rate_key}={rate:5.2f}Gbps seed={seed:3d} '
                                  f'{name:14s} util={hist["utility"][-1]:9.2f} '
                                  f'({wall_s:7.1f}s)', flush=True)
                            path = os.path.join(OUT_DIR, out_name)
                            with open(path, 'w', newline='') as f:
                                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                                w.writeheader(); w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {os.path.relpath(path)}')
    return rows


def run_realistic_isl(grid_gbps=REALISTIC_ISL_GRID_GBPS,
                      seeds=REALISTIC_SWEEP_SEEDS, quick=False):
    """Vary ISL, GSL/GBS fixed at the 'realistic' nominal point."""
    return _realistic_sensitivity('isl_rate', grid_gbps, 'ISL_RATE_BPS',
                                  seeds, quick)


def run_realistic_gsl(grid_gbps=REALISTIC_GSL_GRID_GBPS,
                      seeds=REALISTIC_SWEEP_SEEDS, quick=False):
    """Vary GSL, ISL/GBS fixed at the 'realistic' nominal point."""
    return _realistic_sensitivity('gsl_rate', grid_gbps, 'GSL_RATE_BPS',
                                  seeds, quick)


def run_realistic_gbs(grid_gbps=REALISTIC_GBS_GRID_GBPS,
                      seeds=REALISTIC_SWEEP_SEEDS, quick=False):
    """Vary aggregate GBS, ISL/GSL fixed at the 'realistic' nominal point."""
    return _realistic_sensitivity('gbs_rate', grid_gbps, 'GS_RATE_BPS',
                                  seeds, quick)


# ── Part 12: per-task data-volume sensitivity under 'realistic-bigdata' ────
# Xuanhao's ask (pasted email thread, 2026-10-04): instead of growing task
# COUNT to force ISL congestion (infeasible for the MPC family -- see Part
# 11's docstring), grow each task's DATA VOLUME directly and see whether
# that creates enough backlog/concurrency to congest the fabric and finally
# separate Flat-MPC/OrbitFlow/Two-Level-MPC. Includes greedy-fixed-16 too,
# so the "MPC beats the simplest fixed-depth policy" comparison carries over
# into the congested regime, not just scheduler-vs-scheduler separation.
# MAX_TASKS is far below 'realistic''s 16000 -- see config.py's
# 'realistic-bigdata' comment for why (active-set growth from tasks that
# never finish, not just task count, drives MILP cost here).
REALISTIC_DATASIZE_GRID_GB = [
    (0.1, 0.3),   # control point: close to today's 140-422MB baseline
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 5.0),   # expected to start crossing into partial-non-completion
]
REALISTIC_DATASIZE_SWEEP_SEEDS = [42]
REALISTIC_DATASIZE_MAX_TASKS = 2000

# Any grid point with delivery_frac below this, or violation_frac above the
# companion threshold, across all three MPC schedulers, is degenerate (same
# precedent as the LOAD_SCALE=20 failure: 4.6% delivery, 100% violations) --
# exclude it from the "pick a reasonable range" decision rather than
# averaging it in.
DEGENERATE_DELIVERY_FRAC_MAX = 0.10
DEGENERATE_VIOLATION_FRAC_MIN = 0.90


def _realistic_datasize_sensitivity(grid_gb, seeds, max_tasks, quick):
    from .twolevel import TwoLevelMPCScheduler
    if quick:
        grid_gb, seeds = grid_gb[:2], seeds[:1]
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    out_name = f'results_realistic_datasize{"_quick" if quick else ""}.csv'
    # scenario applied in its own `with` so apply_scenario() (called inside
    # config_override when scenario= is passed) doesn't clobber the
    # MAX_TASKS/TASK_DATA_GB_* overrides nested below it -- same pattern
    # Part 11 uses for its ISL/GSL/GBS rate overrides.
    with C.config_override(scenario='realistic-bigdata'):
        with C.config_override(MAX_TASKS=max_tasks):
            topo = T.build_topology()
            for lo, hi in grid_gb:
                with C.config_override(TASK_DATA_GB_MIN=lo,
                                       TASK_DATA_GB_MAX=hi):
                    for seed in seeds:
                        with C.config_override(RNG_SEED=seed):
                            makers = [
                                ('Flat-MPC',
                                 lambda tp, tk: MPCScheduler(tp, tk)),
                                ('OrbitFlow', lambda tp, tk: MPCScheduler(
                                    tp, tk, route_mode='predictive')),
                                ('Two-Level-MPC',
                                 lambda tp, tk: TwoLevelMPCScheduler(tp, tk)),
                                ('greedy-fixed-16', lambda tp, tk:
                                 GreedyScheduler(tp, tk, depth=16)),
                            ]
                            for name, make in makers:
                                tk = TK.generate_tasks(topo)
                                tot_img = sum(k.n_images for k in tk)
                                sched = make(topo, tk)
                                t0 = time.time()
                                hist = sched.run()
                                wall_s = time.time() - t0
                                row = _row(hi, seed, name, tk, hist, tot_img,
                                          wall_s, obj=sched,
                                          rate_key='data_gb_max')
                                row['data_gb_min'] = lo
                                rows.append(row)
                                print(f'  gb=[{lo},{hi}] seed={seed:3d} '
                                     f'{name:14s} '
                                     f'util={hist["utility"][-1]:9.2f} '
                                     f'delivery={row["delivery_frac"]:.3f} '
                                     f'viol={row["violation_frac"]:.3f} '
                                     f'({wall_s:7.1f}s)', flush=True)
                                path = os.path.join(OUT_DIR, out_name)
                                with open(path, 'w', newline='') as f:
                                    w = csv.DictWriter(
                                        f, fieldnames=list(rows[0].keys()))
                                    w.writeheader(); w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {os.path.relpath(path)}')
    return rows


def run_realistic_datasize(grid_gb=REALISTIC_DATASIZE_GRID_GB,
                           seeds=REALISTIC_DATASIZE_SWEEP_SEEDS,
                           max_tasks=REALISTIC_DATASIZE_MAX_TASKS,
                           quick=False):
    """Vary per-task data volume under 'realistic-bigdata' (Xuanhao's ask)."""
    return _realistic_datasize_sensitivity(grid_gb, seeds, max_tasks, quick)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--couplings', action='store_true',
                    help='historical: run the routing-coupling grid (ISL '
                         'rates under fabric-limited) instead of the GBS-rate '
                         'grid')
    ap.add_argument('--realistic-isl', action='store_true',
                    help="Part 11: vary ISL under 'realistic', GSL/GBS fixed")
    ap.add_argument('--realistic-gsl', action='store_true',
                    help="Part 11: vary GSL under 'realistic', ISL/GBS fixed")
    ap.add_argument('--realistic-gbs', action='store_true',
                    help="Part 11: vary GBS under 'realistic', ISL/GSL fixed")
    ap.add_argument('--realistic-datasize', action='store_true',
                    help="Part 12: vary per-task data volume under "
                         "'realistic-bigdata' (Xuanhao's ask)")
    ap.add_argument('--utility', default=C.UTIL_MODE,
                    choices=('legacy', 'unified'))
    args = ap.parse_args()
    C.apply_utility_mode(args.utility)
    if args.realistic_isl:
        run_realistic_isl(quick=args.quick)
    elif args.realistic_gsl:
        run_realistic_gsl(quick=args.quick)
    elif args.realistic_gbs:
        run_realistic_gbs(quick=args.quick)
    elif args.realistic_datasize:
        run_realistic_datasize(quick=args.quick)
    elif args.couplings:
        run_couplings(quick=args.quick)
    else:
        run(quick=args.quick)
