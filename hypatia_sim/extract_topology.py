"""
Extract and summarise the real communication topology from a generated
Hypatia satellite-network state.

Run on the server after the satellite state has been generated:
    python3 extract_topology.py --gen-dir <path to gen_data/<scenario>>

It reports, using the actual generated data (not assumptions):
  - Constellation summary (satellites, orbital planes, ISL count/degree)
  - Inter-satellite link (ISL) topology: per-satellite degree distribution
  - Ground stations (AC nodes)
  - Per-ground-station serving satellite over time  -> the "compute" satellite
    actually overhead at each instant
  - Handoff events (when the serving satellite changes)
  - Communication windows (how long each satellite serves a station)
  - How many distinct satellites do downlink "computation" for each station

Writes a human-readable topology_summary.txt next to this script.
"""

import argparse
import glob
import os
import re
from collections import defaultdict

STEP_NS = 100_000_000  # 100 ms dynamic-state interval


def find_scenario_dir(gen_dir):
    """Accept either a scenario dir or a gen_data dir with one scenario inside."""
    if os.path.isdir(os.path.join(gen_dir, "dynamic_state_100ms_for_200s")):
        return gen_dir
    subs = [d for d in glob.glob(os.path.join(gen_dir, "*")) if os.path.isdir(d)]
    for d in subs:
        if os.path.isdir(os.path.join(d, "dynamic_state_100ms_for_200s")):
            return d
    return gen_dir


def read_constellation(scenario_dir):
    """Read tles.txt / description.txt for constellation parameters."""
    info = {}
    tles = os.path.join(scenario_dir, "tles.txt")
    if os.path.exists(tles):
        with open(tles) as f:
            first = f.readline().strip()
        # First line is "<num_orbits> <num_sats_per_orbit>"
        parts = first.split()
        if len(parts) == 2 and parts[0].isdigit():
            info["num_planes"] = int(parts[0])
            info["sats_per_plane"] = int(parts[1])
            info["num_satellites"] = int(parts[0]) * int(parts[1])
    return info


def read_isls(scenario_dir):
    """Read isls.txt -> list of (sat_a, sat_b) and per-satellite degree."""
    path = os.path.join(scenario_dir, "isls.txt")
    edges = []
    degree = defaultdict(int)
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a, b = (int(x) for x in line.split())
                edges.append((a, b))
                degree[a] += 1
                degree[b] += 1
    return edges, degree


def read_ground_stations(scenario_dir, num_satellites):
    """Read ground_stations.txt -> list of (node_id, name, lat, lon)."""
    path = os.path.join(scenario_dir, "ground_stations.txt")
    stations = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                gid = int(parts[0])
                stations.append((num_satellites + gid, parts[1],
                                 float(parts[2]), float(parts[3])))
    return stations


def serving_satellite_timeline(scenario_dir, gs_node_ids):
    """
    Walk the forwarding-state files in time order and record, for each ground
    station node, the satellite it is connected to (its first hop) at each step.

    Returns dict: gs_node -> list of (time_ns, serving_sat_or_None)
    """
    dyn = os.path.join(scenario_dir, "dynamic_state_100ms_for_200s")
    files = glob.glob(os.path.join(dyn, "fstate_*.txt"))

    def ts_of(p):
        m = re.search(r"fstate_(\d+)\.txt", p)
        return int(m.group(1)) if m else 0

    files.sort(key=ts_of)
    gs_set = set(gs_node_ids)
    timeline = {g: [] for g in gs_node_ids}

    for fp in files:
        t = ts_of(fp)
        serving = {g: None for g in gs_node_ids}
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                cur = int(parts[0])
                if cur in gs_set and serving[cur] is None:
                    nxt = int(parts[2])
                    if nxt != -1:
                        serving[cur] = nxt
        for g in gs_node_ids:
            timeline[g].append((t, serving[g]))

    return timeline


def windows_from_timeline(tl):
    """
    Collapse a (time, serving_sat) timeline into contiguous windows.
    Returns list of (serving_sat, start_ns, end_ns, duration_ns).
    """
    windows = []
    if not tl:
        return windows
    cur_sat, cur_start = tl[0][1], tl[0][0]
    last_t = tl[0][0]
    for t, sat in tl[1:]:
        if sat != cur_sat:
            windows.append((cur_sat, cur_start, t, t - cur_start))
            cur_sat, cur_start = sat, t
        last_t = t
    windows.append((cur_sat, cur_start, last_t + STEP_NS, last_t + STEP_NS - cur_start))
    return windows


def main():
    ap = argparse.ArgumentParser(description="Summarise Hypatia topology")
    ap.add_argument("--gen-dir", required=True,
                    help="Path to gen_data/<scenario> (or its parent gen_data)")
    args = ap.parse_args()

    scenario = find_scenario_dir(args.gen_dir)
    out_lines = []

    def emit(s=""):
        print(s)
        out_lines.append(s)

    emit("=" * 64)
    emit("COMMUNICATION TOPOLOGY SUMMARY")
    emit(f"Scenario: {os.path.basename(scenario)}")
    emit("=" * 64)

    # --- Constellation ---
    const = read_constellation(scenario)
    edges, degree = read_isls(scenario)
    num_sats = const.get("num_satellites",
                         (max(max(a, b) for a, b in edges) + 1) if edges else 0)

    emit("\n--- Constellation ---")
    if const:
        emit(f"  Orbital planes      : {const.get('num_planes', '?')}")
        emit(f"  Satellites per plane: {const.get('sats_per_plane', '?')}")
    emit(f"  Total satellites    : {num_sats}")
    emit(f"  Total ISLs          : {len(edges)}")
    if degree:
        degs = list(degree.values())
        emit(f"  ISLs per satellite  : min {min(degs)}, max {max(degs)}, "
             f"avg {sum(degs)/len(degs):.2f}")
        emit(f"  (+Grid expectation  : 4 ISLs/sat)")

    # --- Ground stations ---
    stations = read_ground_stations(scenario, num_sats)
    emit("\n--- Ground Stations (AC nodes) ---")
    for node_id, name, lat, lon in stations:
        emit(f"  node {node_id:<6} {name:<14} ({lat:.4f}, {lon:.4f})")

    # --- Serving satellites over time (compute satellites) ---
    gs_nodes = [s[0] for s in stations]
    timeline = serving_satellite_timeline(scenario, gs_nodes)

    emit("\n--- Communication Windows & Serving (compute) Satellites ---")
    name_by_node = {s[0]: s[1] for s in stations}
    for g in gs_nodes:
        wins = windows_from_timeline(timeline[g])
        served = [w for w in wins if w[0] is not None]
        distinct_sats = sorted({w[0] for w in served})
        gap = [w for w in wins if w[0] is None]
        emit(f"\n  {name_by_node[g]} (node {g}):")
        emit(f"    Handoffs (serving-sat changes): {max(len(served) - 1, 0)}")
        emit(f"    Distinct serving satellites    : {len(distinct_sats)}")
        if served:
            durs = [w[3] / 1e9 for w in served]
            emit(f"    Window duration (s)            : "
                 f"min {min(durs):.1f}, max {max(durs):.1f}, avg {sum(durs)/len(durs):.1f}")
        if gap:
            total_gap = sum(w[3] for w in gap) / 1e9
            emit(f"    Out-of-coverage time (s)       : {total_gap:.1f}")
        sample = ", ".join(str(s) for s in distinct_sats[:8])
        emit(f"    Serving satellite IDs          : {sample}"
             f"{' ...' if len(distinct_sats) > 8 else ''}")

    emit("\n--- Interpretation ---")
    emit("  'Serving satellite' = the satellite overhead a ground station at a")
    emit("  given instant. Because all satellites are equal, this is exactly the")
    emit("  satellite that would image the region, run RQ-VAE compression, and")
    emit("  downlink. 'Distinct serving satellites' is how many satellites do")
    emit("  downlink computation for that station over the simulation window.")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "topology_summary.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    emit(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
