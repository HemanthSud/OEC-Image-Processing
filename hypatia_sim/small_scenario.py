"""
Small but complete satellite network scenario for the OEC RQ-VAE downlink study.

Constellation: Walker 6/2/1  (2 orbital planes × 3 satellites, 550 km, 53°)
Ground stations: Tokyo and Sao Paulo  (from Hypatia top-100 list)
Duration: 6 000 s  (≈ one full orbital period, ~95.6 min)
Time step: 10 s

Outputs (all in small_scenario/):
  contact_windows.csv    — every visibility window per (satellite, ground station)
  link_availability.csv  — ISL and GSL state at every time step
  path_info.csv          — serving satellite + end-to-end routing path per GS
  summary.txt            — human-readable report

No ns-3 or Hypatia state generation needed.  Pure NumPy orbital mechanics.
Run from hypatia_sim/:
    python3 small_scenario.py
"""

import csv
import math
import os
from collections import defaultdict

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
R_E     = 6_371.0          # Earth radius, km
MU      = 398_600.4418     # Gravitational parameter, km³ s⁻²
OMEGA_E = 7.2921150e-5     # Earth rotation rate, rad s⁻¹
C_KM_S  = 299_792.458      # Speed of light, km s⁻¹
DEG     = math.pi / 180

# ── Constellation ─────────────────────────────────────────────────────────────
ALT_KM         = 550.0
INCL_DEG       = 53.0
N_PLANES       = 2
SATS_PER_PLANE = 3
N_SATS         = N_PLANES * SATS_PER_PLANE    # 6

_a_sma = R_E + ALT_KM                         # semi-major axis, km
_n_mm  = math.sqrt(MU / _a_sma**3)            # mean motion, rad s⁻¹
T_ORB  = 2 * math.pi / _n_mm                  # orbital period, s  (~5737 s)
_incl  = INCL_DEG * DEG

# Walker 6/2/1: RAAN spacing = 180°, inter-plane phase offset = 60° (F=1)
_RAAN_SPACING  = 2 * math.pi / N_PLANES        # π  (180°)
_PHASE_OFFSET  = 2 * math.pi / N_SATS          # π/3 (60°)

SATELLITES = []
for _p in range(N_PLANES):
    _raan = _p * _RAAN_SPACING
    for _s in range(SATS_PER_PLANE):
        _anom0 = _s * (2 * math.pi / SATS_PER_PLANE) + _p * _PHASE_OFFSET
        SATELLITES.append({
            'id':           _p * SATS_PER_PLANE + _s,
            'plane':        _p,
            'idx_in_plane': _s,
            'raan':         _raan,
            'anom0':        _anom0,
        })

# ── ISL topology (+ring intra-plane, nearest inter-plane) ────────────────────
# Intra-plane rings: 0↔1, 1↔2, 2↔0  and  3↔4, 4↔5, 5↔3
# Inter-plane pairs: 0↔3, 1↔4, 2↔5
ISLS = []
for _p in range(N_PLANES):
    _base = _p * SATS_PER_PLANE
    for _s in range(SATS_PER_PLANE):
        ISLS.append((_base + _s, _base + (_s + 1) % SATS_PER_PLANE))
for _s in range(SATS_PER_PLANE):
    ISLS.append((_s, SATS_PER_PLANE + _s))

# ── Ground stations (Tokyo + Sao Paulo, both in Hypatia top-100) ─────────────
GROUND_STATIONS = [
    {'id': N_SATS,     'name': 'Tokyo',     'lat':  35.6895, 'lon':  139.6917},
    {'id': N_SATS + 1, 'name': 'Sao-Paulo', 'lat': -23.5475, 'lon': -46.6361},
]
MIN_EL_DEG = 10.0
MIN_EL_RAD = MIN_EL_DEG * DEG

# ── Simulation window ─────────────────────────────────────────────────────────
SIM_DURATION_S = 6_000
TIME_STEP_S    = 10
OUT_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'small_scenario')


# ── Orbital mechanics ─────────────────────────────────────────────────────────

def sat_eci(sat, t):
    """ECI position of satellite at time t (km)."""
    theta = sat['anom0'] + _n_mm * t     # true anomaly (circular orbit)
    raan  = sat['raan']
    x_op  = _a_sma * math.cos(theta)
    y_op  = _a_sma * math.sin(theta)
    # Rz(raan) · Rx(incl) · [x_op, y_op, 0]
    cr, sr = math.cos(raan), math.sin(raan)
    ci, si = math.cos(_incl), math.sin(_incl)
    return np.array([
        cr * x_op - sr * ci * y_op,
        sr * x_op + cr * ci * y_op,
        si * y_op,
    ])


def eci_to_ecef(pos, t):
    """Rotate ECI → ECEF (simplified: GMST = OMEGA_E * t)."""
    theta = OMEGA_E * t
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
         ct * pos[0] + st * pos[1],
        -st * pos[0] + ct * pos[1],
        pos[2],
    ])


def gs_ecef(gs):
    """Ground station ECEF position (km)."""
    lat = gs['lat'] * DEG
    lon = gs['lon'] * DEG
    return R_E * np.array([
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    ])


def elevation_rad(sat_ecef_pos, gs_pos_ecef):
    """Elevation angle of satellite above ground station horizon (radians)."""
    delta  = sat_ecef_pos - gs_pos_ecef
    n_hat  = gs_pos_ecef / np.linalg.norm(gs_pos_ecef)
    dist   = float(np.linalg.norm(delta))
    if dist == 0:
        return 0.0
    return math.asin(float(np.dot(delta, n_hat)) / dist)


# ── Graph routing (BFS by hops) ───────────────────────────────────────────────

def bfs_path(graph, src, dst):
    """Return (node_list, total_dist_km) or (None, inf)."""
    if src == dst:
        return [src], 0.0
    from collections import deque
    queue   = deque([(src, [src], 0.0)])
    visited = {src}
    while queue:
        node, path, dist = queue.popleft()
        for nbr, edge_dist in graph.get(node, []):
            if nbr not in visited:
                new_dist = dist + edge_dist
                new_path = path + [nbr]
                if nbr == dst:
                    return new_path, new_dist
                visited.add(nbr)
                queue.append((nbr, new_path, new_dist))
    return None, float('inf')


# ── Main simulation ───────────────────────────────────────────────────────────

def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    gs_ecef_cache = {gs['id']: gs_ecef(gs) for gs in GROUND_STATIONS}

    times = list(range(0, SIM_DURATION_S + TIME_STEP_S, TIME_STEP_S))

    # contact[gs_id][sat_id] = sorted list of time steps with visibility
    contact = defaultdict(lambda: defaultdict(list))

    path_records = []    # one entry per time step
    link_rows    = []    # one row per (time step, link)

    for t in times:
        # Satellite ECEF positions
        sat_pos = {}
        for sat in SATELLITES:
            sat_pos[sat['id']] = eci_to_ecef(sat_eci(sat, t), t)

        # Build routing graph
        graph = defaultdict(list)

        # ISL edges (always available — laser links between satellites)
        for link_id, (a_id, b_id) in enumerate(ISLS):
            dist = float(np.linalg.norm(sat_pos[a_id] - sat_pos[b_id]))
            graph[a_id].append((b_id, dist))
            graph[b_id].append((a_id, dist))
            link_rows.append({
                't': t, 'link_id': link_id, 'type': 'ISL',
                'node_a': a_id, 'node_b': b_id,
                'dist_km': round(dist, 2), 'available': 1,
                'elevation_deg': '',
            })

        # GSL edges (elevation-gated)
        for gs in GROUND_STATIONS:
            gs_id  = gs['id']
            gs_pos = gs_ecef_cache[gs_id]
            for sat in SATELLITES:
                sat_id = sat['id']
                el     = elevation_rad(sat_pos[sat_id], gs_pos)
                avail  = 1 if el >= MIN_EL_RAD else 0
                dist   = float(np.linalg.norm(sat_pos[sat_id] - gs_pos))
                if avail:
                    graph[sat_id].append((gs_id, dist))
                    graph[gs_id].append((sat_id, dist))
                    contact[gs_id][sat_id].append(t)
                link_rows.append({
                    't': t,
                    'link_id': len(ISLS) + gs_id * N_SATS + sat_id,
                    'type': 'GSL',
                    'node_a': sat_id, 'node_b': gs_id,
                    'dist_km': round(dist, 2), 'available': avail,
                    'elevation_deg': round(el / DEG, 2),
                })

        # Per-GS: serving satellite + path from sat 0
        t_rec = {'t': t, 'gs': {}}
        for gs in GROUND_STATIONS:
            gs_id  = gs['id']
            gs_pos = gs_ecef_cache[gs_id]

            # Serving satellite = highest-elevation visible satellite
            best_sat, best_el = None, -999.0
            for sat in SATELLITES:
                el = elevation_rad(sat_pos[sat['id']], gs_pos)
                if el >= MIN_EL_RAD and el > best_el:
                    best_sat, best_el = sat['id'], el

            # End-to-end path from satellite 0 (imaging/compute sat) to GS
            path, total_dist = bfs_path(graph, 0, gs_id)
            delay_ms = total_dist / C_KM_S * 1000.0 if path else None

            t_rec['gs'][gs_id] = {
                'serving_sat':    best_sat,
                'serving_el_deg': round(best_el / DEG, 2) if best_sat is not None else None,
                'path':           path,
                'hops':           len(path) - 1 if path else None,
                'prop_delay_ms':  round(delay_ms, 3) if delay_ms is not None else None,
                'reachable':      path is not None,
            }
        path_records.append(t_rec)

    # ── Contact windows ───────────────────────────────────────────────────────
    windows = []
    for gs in GROUND_STATIONS:
        gs_id = gs['id']
        for sat in SATELLITES:
            sat_id = sat['id']
            ts = contact[gs_id][sat_id]
            if not ts:
                continue
            win_start = ts[0]
            prev      = ts[0]
            for cur in ts[1:]:
                if cur - prev > TIME_STEP_S:
                    windows.append({
                        'gs': gs['name'], 'sat_id': sat_id,
                        'start_s': win_start, 'end_s': prev,
                        'duration_s': prev - win_start,
                    })
                    win_start = cur
                prev = cur
            windows.append({
                'gs': gs['name'], 'sat_id': sat_id,
                'start_s': win_start, 'end_s': prev,
                'duration_s': prev - win_start,
            })

    # ── Write CSVs ────────────────────────────────────────────────────────────
    cw_path = os.path.join(OUT_DIR, 'contact_windows.csv')
    with open(cw_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['gs', 'sat_id', 'start_s', 'end_s', 'duration_s'])
        w.writeheader()
        w.writerows(windows)
    print(f'Wrote {len(windows)} contact windows → {cw_path}')

    la_path = os.path.join(OUT_DIR, 'link_availability.csv')
    with open(la_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['t', 'link_id', 'type', 'node_a', 'node_b',
                                           'dist_km', 'available', 'elevation_deg'])
        w.writeheader()
        w.writerows(link_rows)
    print(f'Wrote {len(link_rows)} link records → {la_path}')

    pi_path = os.path.join(OUT_DIR, 'path_info.csv')
    with open(pi_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_s', 'ground_station', 'serving_sat', 'serving_el_deg',
                    'src_sat', 'path', 'hops', 'prop_delay_ms', 'reachable'])
        for rec in path_records:
            for gs in GROUND_STATIONS:
                gs_id = gs['id']
                g = rec['gs'][gs_id]
                w.writerow([
                    rec['t'], gs['name'],
                    g['serving_sat'], g['serving_el_deg'],
                    0,
                    '→'.join(str(x) for x in g['path']) if g['path'] else 'UNREACHABLE',
                    g['hops'], g['prop_delay_ms'], g['reachable'],
                ])
    print(f'Wrote {len(path_records) * len(GROUND_STATIONS)} path records → {pi_path}')

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = _build_summary(windows, path_records)
    print()
    print(summary)
    sm_path = os.path.join(OUT_DIR, 'summary.txt')
    with open(sm_path, 'w') as f:
        f.write(summary)
    print(f'\nSummary written → {sm_path}')


def _build_summary(windows, path_records):
    L = []
    L.append('=' * 65)
    L.append('Small Satellite Network Scenario — Summary')
    L.append('=' * 65)
    L.append(f'Constellation  Walker {N_SATS}/{N_PLANES}/1  '
             f'({N_PLANES} planes × {SATS_PER_PLANE} sats)')
    L.append(f'Altitude       {ALT_KM} km    Inclination {INCL_DEG}°')
    L.append(f'Orbital period {T_ORB:.0f} s  ({T_ORB/60:.1f} min)')
    L.append(f'ISLs ({len(ISLS)} total): '
             + '  '.join(f'{a}↔{b}' for a, b in ISLS))
    L.append(f'Ground stations  '
             + ',  '.join(f'{g["name"]} ({g["lat"]}°, {g["lon"]}°)'
                          for g in GROUND_STATIONS))
    L.append(f'Simulation     {SIM_DURATION_S} s  ({TIME_STEP_S} s steps)')
    L.append(f'Min elevation  {MIN_EL_DEG}°')
    L.append('')

    L.append('── Contact Windows (satellite visibility) ───────────────────')
    for gs in GROUND_STATIONS:
        gs_wins = [w for w in windows if w['gs'] == gs['name']]
        if not gs_wins:
            L.append(f'  {gs["name"]:12s}  no coverage in simulation window')
            continue
        durations   = [w['duration_s'] for w in gs_wins]
        total_cov   = sum(durations)
        pct         = total_cov / SIM_DURATION_S * 100
        sats_seen   = sorted({w['sat_id'] for w in gs_wins})
        L.append(f'  {gs["name"]:12s}  '
                 f'{len(gs_wins)} windows  '
                 f'total {total_cov} s ({pct:.1f}%)  '
                 f'avg {sum(durations)/len(durations):.0f} s  '
                 f'sats seen: {sats_seen}')
        for w in gs_wins:
            L.append(f'    sat {w["sat_id"]}  '
                     f't={w["start_s"]}–{w["end_s"]} s  '
                     f'({w["duration_s"]} s)')
    L.append('')

    L.append('── End-to-End Path from Sat 0 (imaging satellite) ──────────')
    for gs in GROUND_STATIONS:
        gs_id  = gs['id']
        delays = [r['gs'][gs_id]['prop_delay_ms']
                  for r in path_records if r['gs'][gs_id]['reachable']]
        hops_l = [r['gs'][gs_id]['hops']
                  for r in path_records if r['gs'][gs_id]['reachable']]
        reach  = len(delays)
        total  = len(path_records)
        if delays:
            L.append(f'  {gs["name"]:12s}  '
                     f'reachable {reach}/{total} steps  |  '
                     f'prop delay  avg {sum(delays)/len(delays):.2f} ms  '
                     f'min {min(delays):.2f} ms  '
                     f'max {max(delays):.2f} ms  |  '
                     f'avg {sum(hops_l)/len(hops_l):.1f} hops')
        else:
            L.append(f'  {gs["name"]:12s}  never reachable from sat 0 in this window')

    L.append('')
    L.append('── ISL Distances at sample times ────────────────────────────')
    L.append(f'  {"Link":<8} {"t=0 km":>10} {"t=1000 km":>12} {"t=3000 km":>12}')
    L.append('  ' + '-' * 44)
    for a_id, b_id in ISLS:
        dists = []
        for t_s in [0, 1000, 3000]:
            pa = eci_to_ecef(sat_eci(SATELLITES[a_id], t_s), t_s)
            pb = eci_to_ecef(sat_eci(SATELLITES[b_id], t_s), t_s)
            dists.append(float(np.linalg.norm(pa - pb)))
        L.append(f'  {a_id}↔{b_id:<5} '
                 f'{dists[0]:>10.1f} {dists[1]:>12.1f} {dists[2]:>12.1f}')

    L.append('')
    L.append('Files in small_scenario/:')
    L.append('  contact_windows.csv    satellite-GS visibility windows')
    L.append('  link_availability.csv  ISL and GSL state at every time step')
    L.append('  path_info.csv          routing path + delay per GS per step')
    L.append('  summary.txt            this report')
    L.append('=' * 65)
    return '\n'.join(L)


if __name__ == '__main__':
    run()
