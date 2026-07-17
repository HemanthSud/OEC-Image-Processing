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


def run_schedulers(topo):
    """Run every scheduler on a fresh copy of the task set."""
    results = {}
    makers = ([lambda tp, tk: MPCScheduler(tp, tk),
               lambda tp, tk: GreedyScheduler(tp, tk, depth=None)]
              + [lambda tp, tk, q=q: GreedyScheduler(tp, tk, depth=q)
                 for q in C.DEPTHS])
    for make in makers:
        tasks = TK.generate_tasks(topo)          # deterministic, fresh state
        sched = make(topo, tasks)
        t0 = time.time()
        hist = sched.run()
        print(f'  {sched.name:16s}  {time.time() - t0:6.1f} s  '
              f'delivered {hist["delivered_images"][-1]:12,.0f} images  '
              f'utility {hist["utility"][-1]:8.2f}')
        results[sched.name] = {'hist': hist, 'tasks': tasks}
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
        rows = [{'kid': k.kid, 'depth': k.depth,
                 'delivered_images': round(k.delivered, 1),
                 'delivery_fraction': round(k.delivered / k.n_images, 4),
                 'utility': round(k.delivered_utility, 4)}
                for k in res['tasks']]
        _write_csv(os.path.join(C.OUT_DIR, f'task_outcomes_{name}.csv'),
                   rows, list(rows[0].keys()))
        hist = res['hist']
        rows = [{'t_s': hist['t_s'][i],
                 'delivered_images': round(hist['delivered_images'][i], 1),
                 'utility': round(hist['utility'][i], 4),
                 'backlog_bits': round(hist['backlog_bits'][i]),
                 'n_active': hist['n_active'][i]}
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
             f'{"utility":>9s} {"on-time%":>9s}')
    for name, res in results.items():
        tk = res['tasks']
        img = sum(k.delivered for k in tk)
        frac = img / tot_img
        ontime = sum(v for k in tk for t, v in k.delivery_slots.items()
                     if (t + 1) * C.SLOT_S <= k.deadline_s)
        L.append(f'  {name:16s} {img:14,.0f} {100 * frac:9.1f}% '
                 f'{res["hist"]["utility"][-1]:9.2f} {100 * ontime / tot_img:8.1f}%')
    L.append('')
    L.append('Outputs in oec_scenario/: contact_windows.csv, topology_state.csv,')
    L.append('tasks.csv, timeline_*.csv, task_outcomes_*.csv, plots/*.png,')
    L.append('viewer.html (interactive simulator)')
    L.append('=' * 74)
    return '\n'.join(L)


def main():
    print('Building topology...')
    t0 = time.time()
    topo = T.build_topology()
    print(f'  done in {time.time() - t0:.1f} s '
          f'({C.N_SATS} sats, {len(topo.isl_pairs)} ISLs, {C.N_SLOTS} slots)')

    print('Running schedulers...')
    results = run_schedulers(topo)

    print('Writing outputs...')
    write_outputs(topo, results)

    summary = build_summary(topo, results)
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
