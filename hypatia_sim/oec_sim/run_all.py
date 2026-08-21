"""
End-to-end driver:  topology -> tasks -> schedulers (MPC + baselines)
-> CSVs + summary + plots + interactive viewer.

Run from hypatia_sim/:
    python3 -m oec_sim.run_all
"""

import csv
import os
import time

import numpy as np

from . import config as C
from . import topology as T
from . import tasks as TK
from .schedulers import GreedyScheduler, MPCScheduler


def _write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'  wrote {len(rows):6d} rows -> {os.path.relpath(path)}')


def run_schedulers(topo, extra_makers=()):
    """Run every scheduler on a fresh copy of the task set. extra_makers
    is a list of (topo, tasks) -> scheduler callables appended to the
    default set (e.g. mpc-congestion, mpc-hier — see run_all.main())."""
    results = {}
    makers = ([lambda tp, tk: MPCScheduler(tp, tk),
               lambda tp, tk: GreedyScheduler(tp, tk, depth=None)]
              + [lambda tp, tk, q=q: GreedyScheduler(tp, tk, depth=q)
                 for q in C.DEPTHS]
              + list(extra_makers))
    for make in makers:
        tasks = TK.generate_tasks(topo)          # deterministic, fresh state
        sched = make(topo, tasks)
        t0 = time.time()
        hist = sched.run()
        print(f'  {sched.name:16s}  {time.time() - t0:6.1f} s  '
              f'delivered {hist["delivered_images"][-1]:12,.0f} images  '
              f'utility {hist["utility"][-1]:8.2f}')
        results[sched.name] = {'hist': hist, 'tasks': tasks,
                               'solve_log': getattr(sched, 'solve_log', [])}
    return results


def write_outputs(topo, results):
    os.makedirs(C.OUT_DIR, exist_ok=True)

    wins = T.contact_windows(topo)
    _write_csv(os.path.join(C.OUT_DIR, 'contact_windows.csv'), wins,
               ['gs', 'sat', 'start_s', 'end_s', 'duration_s'])

    # per-slot topology state (aggregates; full per-link table is ~1.4M rows)
    rows = []
    intra = np.array([abs(a - b) in (1, C.SATS_PER_PLANE - 1)
                      and a // C.SATS_PER_PLANE == b // C.SATS_PER_PLANE
                      for a, b in topo.isl_pairs])
    for t in range(C.N_SLOTS):
        rows.append({
            't_s': int(topo.times[t]),
            'isl_feasible': int(topo.isl_ok[t].sum()),
            'isl_feasible_intra': int(topo.isl_ok[t][intra].sum()),
            'isl_feasible_inter': int(topo.isl_ok[t][~intra].sum()),
            'isl_dist_min_km': round(float(topo.isl_dist[t][topo.isl_ok[t]].min()), 1),
            'isl_dist_max_km': round(float(topo.isl_dist[t][topo.isl_ok[t]].max()), 1),
            **{f'sats_visible_{g["name"]}': int(topo.gsl_ok[t, :, gi].sum())
               for gi, g in enumerate(C.GROUND_STATIONS)},
        })
    _write_csv(os.path.join(C.OUT_DIR, 'topology_state.csv'), rows,
               list(rows[0].keys()))

    # tasks + per-scheduler outcomes
    any_tasks = next(iter(results.values()))['tasks']
    rows = [{'kid': k.kid, 'aoi': k.aoi, 'src_sat': k.src_sat,
             'dst_gs': C.GROUND_STATIONS[k.dst_gs]['name'],
             'arrival_s': k.arrival_slot * C.SLOT_S, 'n_images': k.n_images,
             'weight': k.weight, 'deadline_s': k.deadline_s}
            for k in any_tasks]
    _write_csv(os.path.join(C.OUT_DIR, 'tasks.csv'), rows, list(rows[0].keys()))

    for name, res in results.items():
        rows = []
        for k in res['tasks']:
            completion_s = ((k.completion_slot + 1) * C.SLOT_S
                            if k.completion_slot is not None else None)
            rows.append({
                'kid': k.kid, 'depth': k.depth,
                'arrival_s': k.arrival_slot * C.SLOT_S,
                'deadline_s': k.deadline_s,
                'delivered_images': round(k.delivered, 1),
                'delivery_fraction': round(k.delivered / k.n_images, 4),
                'utility': round(k.delivered_utility, 4),
                'first_delivery_s': ((k.first_delivery_slot + 1) * C.SLOT_S
                                     if k.first_delivery_slot is not None else None),
                'completion_s': completion_s,
                'completion_delay_s': (completion_s - k.arrival_slot * C.SLOT_S
                                       if completion_s is not None else None),
                'lateness_s': (completion_s - k.deadline_s
                               if completion_s is not None else None),
                'late_image_fraction': round(k.late_images / k.n_images, 4),
                'deadline_violated': int(
                    k.dropped or completion_s is None or completion_s > k.deadline_s),
                'dropped': int(k.dropped),
                'rejected': int(k.rejected),
            })
        _write_csv(os.path.join(C.OUT_DIR, f'task_outcomes_{name}.csv'),
                   rows, list(rows[0].keys()))
        if res.get('solve_log'):
            _write_csv(os.path.join(C.OUT_DIR, f'solve_log_{name}.csv'),
                       res['solve_log'], list(res['solve_log'][0].keys()))
        hist = res['hist']
        rows = [{'t_s': hist['t_s'][i],
                 'delivered_images': round(hist['delivered_images'][i], 1),
                 'utility': round(hist['utility'][i], 4),
                 'backlog_bits': round(hist['backlog_bits'][i]),
                 'n_active': hist['n_active'][i],
                 'n_dropped': hist['n_dropped'][i]}
                for i in range(len(hist['t_s']))]
        _write_csv(os.path.join(C.OUT_DIR, f'timeline_{name}.csv'),
                   rows, list(rows[0].keys()))


def build_summary(topo, results):
    L = []
    L.append('=' * 74)
    L.append('OEC RQ-NAC Network Simulation — Summary')
    L.append('=' * 74)
    L.append(f'Constellation   {C.CONSTELLATION_NAME}: Walker '
             f'{C.N_SATS}/{C.N_PLANES}/{C.PHASING_F}  '
             f'({C.N_PLANES} planes x {C.SATS_PER_PLANE} sats)')
    L.append(f'Altitude        {C.ALT_KM:.0f} km   inclination {C.INCL_DEG}°  '
             f'orbital period {C.T_ORB:.0f} s')
    L.append(f'ISLs            +Grid, {len(topo.isl_pairs)} candidate links, '
             f'LOS-gated (graze > {C.LOS_GRAZE_KM:.0f} km) '
             f'+ range <= {C.ISL_MAX_KM:.0f} km')
    feas = topo.isl_ok.mean(axis=0)
    L.append(f'                feasible on average: '
             f'{topo.isl_ok.sum(axis=1).mean():.0f}/{len(topo.isl_pairs)} '
             f'({100 * feas.mean():.1f}%)')
    L.append(f'GBSs            ' + ', '.join(g['name'] for g in C.GROUND_STATIONS)
             + f'   (min elevation {C.MIN_ELEV_DEG}°)')
    L.append(f'Link rates      ISL {C.ISL_RATE_BPS / 1e6:.0f} Mbps, '
             f'GBS aggregate {C.GS_RATE_BPS / 1e6:.0f} Mbps')
    L.append(f'Simulation      {C.SIM_S} s ({C.SIM_S / 3600:.1f} h, '
             f'{C.SIM_S / C.T_ORB:.1f} orbits) at dt = {C.SLOT_S} s '
             f'-> {C.N_SLOTS} slots')
    L.append(f'Depths D        {C.DEPTHS}  payload b_q = '
             + ', '.join(f'{C.PAYLOAD_B[q]} B' for q in C.DEPTHS))
    L.append('')

    wins = T.contact_windows(topo)
    L.append('── Contact windows ──────────────────────────────────────────')
    for g in C.GROUND_STATIONS:
        gw = [w for w in wins if w['gs'] == g['name']]
        cov = topo.gsl_ok[:, :, C.GROUND_STATIONS.index(g)].any(axis=1).mean()
        L.append(f'  {g["name"]:10s}  {len(gw):3d} windows   '
                 f'avg {np.mean([w["duration_s"] for w in gw]):5.0f} s   '
                 f'>=1 sat visible {100 * cov:5.1f}% of slots')
    L.append('')

    any_tasks = next(iter(results.values()))['tasks']
    tot_img = sum(k.n_images for k in any_tasks)
    q_max = max(C.DEPTHS)
    offered16 = tot_img * C.PAYLOAD_B[q_max] * 8
    cap = C.N_GS * C.GS_RATE_BPS * C.SIM_S
    L.append('── Task load K ──────────────────────────────────────────────')
    L.append(f'  {len(any_tasks)} tasks, {tot_img:,} images total, '
             f'N_k ~ U({C.TASK_IMAGES_MIN:,}, {C.TASK_IMAGES_MAX:,})')
    L.append(f'  offered load at depth-{q_max}: {offered16 / 1e9:.1f} Gbit  vs  '
             f'total GBS capacity {cap / 1e9:.1f} Gbit  '
             f'(utilization {offered16 / cap:.2f})')
    L.append(f'  encoder pipeline: {C.ENC_IMGS_PER_S:.0f} img/s per satellite '
             f'({C.ENC_S_PER_IMG * 1e3:.2f} ms/img measured)')
    L.append('')

    L.append('── Scheduler comparison (maximize images transmitted / utility) ─')
    L.append(f'  {"scheduler":16s} {"images":>14s} {"delivery%":>10s} '
             f'{"utility":>9s} {"on-time%":>9s} {"viol%":>7s} {"dropped":>8s}')
    for name, res in results.items():
        tk = res['tasks']
        img = sum(k.delivered for k in tk)
        frac = img / tot_img
        ontime = sum(v for k in tk for t, v in k.delivery_slots.items()
                     if (t + 1) * C.SLOT_S <= k.deadline_s)
        n_viol = sum(1 for k in tk if k.dropped or k.completion_slot is None
                    or (k.completion_slot + 1) * C.SLOT_S > k.deadline_s)
        n_dropped = sum(1 for k in tk if k.dropped)
        L.append(f'  {name:16s} {img:14,.0f} {100 * frac:9.1f}% '
                 f'{res["hist"]["utility"][-1]:9.2f} {100 * ontime / tot_img:8.1f}% '
                 f'{100 * n_viol / len(tk):6.1f}% {n_dropped:8d}')
    L.append('')

    L.append('── Delay / depth mix ────────────────────────────────────────')
    for name, res in results.items():
        tk = res['tasks']
        delays = [k.completion_slot * C.SLOT_S - k.arrival_slot * C.SLOT_S
                 for k in tk if k.completion_slot is not None]
        mean_d = np.mean(delays) if delays else float('nan')
        p95_d = np.percentile(delays, 95) if delays else float('nan')
        mix = ' '.join(f'{q}:{sum(1 for k in tk if k.depth == q)}' for q in C.DEPTHS)
        L.append(f'  {name:16s} mean delay {mean_d:7.0f}s  p95 {p95_d:7.0f}s  '
                 f'depth mix {mix}')
    L.append('')

    L.append('── MPC solve-time instrumentation ───────────────────────────')
    for name, res in results.items():
        log = res.get('solve_log') or []
        if not log:
            continue
        ms = [r['wall_s'] * 1e3 for r in log]
        L.append(f'  {name:16s} {len(log):4d} solves   '
                 f'mean {np.mean(ms):7.1f} ms   p95 {np.percentile(ms, 95):7.1f} ms   '
                 f'max {np.max(ms):7.1f} ms   total {sum(ms) / 1e3:6.1f} s')
    L.append('')

    L.append('Outputs in oec_scenario/: contact_windows.csv, topology_state.csv,')
    L.append('tasks.csv, timeline_*.csv, task_outcomes_*.csv, solve_log_*.csv,')
    L.append('plots/*.png, viewer.html (interactive simulator)')
    L.append('=' * 74)
    return '\n'.join(L)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', default=C.SCENARIO,
                   choices=list(C._SCENARIOS))
    p.add_argument('--congestion', action='store_true',
                   help='also run mpc-congestion (predictive routing, '
                        'Dr. Liu\'s directive) alongside the flat mpc')
    p.add_argument('--hier', action='store_true',
                   help='also run mpc-hier (hierarchical MPC)')
    p.add_argument('--oracle', action='store_true',
                   help='compute the HiGHS offline upper bound and append '
                        'the optimality-gap table to summary.txt')
    args = p.parse_args(argv)
    C.apply_scenario(args.scenario)

    print(f'Building topology... (scenario={args.scenario})')
    t0 = time.time()
    topo = T.build_topology()
    print(f'  done in {time.time() - t0:.1f} s '
          f'({C.N_SATS} sats, {len(topo.isl_pairs)} ISLs, {C.N_SLOTS} slots)')

    extra_makers = []
    if args.congestion:
        extra_makers.append(lambda tp, tk: MPCScheduler(tp, tk, route_mode='predictive'))
    if args.hier:
        from .hier import HierarchicalMPCScheduler
        extra_makers.append(lambda tp, tk: HierarchicalMPCScheduler(tp, tk))

    print('Running schedulers...')
    results = run_schedulers(topo, extra_makers=extra_makers)

    oracle_txt = None
    if args.oracle:
        from . import oracle as OR
        any_tasks = next(iter(results.values()))['tasks']
        print('Solving oracle upper bound...')
        oracle_txt = OR.report(topo, any_tasks, results)
        with open(os.path.join(C.OUT_DIR, 'upper_bound.txt'), 'w') as f:
            f.write(oracle_txt)
        print(oracle_txt)

    print('Writing outputs...')
    write_outputs(topo, results)

    summary = build_summary(topo, results)
    if oracle_txt:
        summary += '\n\n' + oracle_txt
    with open(os.path.join(C.OUT_DIR, 'summary.txt'), 'w') as f:
        f.write(summary)
    print()
    print(summary)

    try:
        from . import plots
        plots.make_all(topo, results)
    except ImportError:
        print('plots.py not available, skipping')
    try:
        from . import viewer
        viewer.write_viewer(os.path.join(C.OUT_DIR, 'viewer.html'))
    except ImportError:
        print('viewer.py not available, skipping')


if __name__ == '__main__':
    main()
