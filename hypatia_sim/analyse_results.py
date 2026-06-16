"""
Analyse Hypatia UDP burst results for RQ-VAE compression models.

Reads the udp_bursts_incoming.csv output from ns-3 and computes:
  - Mean / median / p99 delivery latency per model
  - Packet delivery rate (drop rate) per model
  - Throughput per model
  - RQ node computing delay contribution
  - ILS (inter-satellite link) utilization
  - Communication window duration (satellite visibility over ground station)
  - Communication topology summary

Run after the ns-3 simulation completes:
    python3 analyse_results.py
"""

import csv
import os
import math
import statistics
from collections import defaultdict

RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run")

BITS_PER_CODE = math.ceil(math.log2(2048))  # 11

MODELS = [
    {"name": "8x8x1",  "codes": 8 * 8 * 1},
    {"name": "8x8x2",  "codes": 8 * 8 * 2},
    {"name": "8x8x4",  "codes": 8 * 8 * 4},
    {"name": "8x8x8",  "codes": 8 * 8 * 8},
    {"name": "8x8x16", "codes": 8 * 8 * 16},
]
N_IMAGES = 100

for m in MODELS:
    m["bits"]  = m["codes"] * BITS_PER_CODE
    m["bytes"] = math.ceil(m["bits"] / 8)


def load_compute_delays():
    """Load per-model RQ node compute delays from profiling output."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compute_delays.txt")
    delays = {}
    if not os.path.exists(path):
        return delays
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "," in line:
                name, ns = line.split(",", 1)
                for m in MODELS:
                    if m["name"] in name:
                        delays[m["name"]] = int(ns) / 1e6  # convert to ms
    return delays


def load_burst_schedule():
    """Map burst_id -> model name."""
    path = os.path.join(RUN_DIR, "udp_burst_schedule.csv")
    burst_to_model = {}
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            burst_id = int(row[0])
            metadata = row[7] if len(row) > 7 else ""
            for part in metadata.split(","):
                if part.startswith("model="):
                    burst_to_model[burst_id] = part.split("=")[1]
    return burst_to_model


def load_incoming():
    path = os.path.join(RUN_DIR, "logs_ns3", "udp_bursts_incoming.csv")
    if not os.path.exists(path):
        print(f"Results file not found: {path}")
        print("Run the ns-3 simulation first.")
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split(","))
    return rows


def load_isl_utilization():
    """
    Load ILS (inter-satellite link) utilization from Hypatia output.
    Returns list of (interval_start_ns, link_id, utilization_fraction).
    """
    path = os.path.join(RUN_DIR, "logs_ns3", "isl_utilization.csv")
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    entries.append((int(parts[0]), int(parts[1]), float(parts[2])))
                except ValueError:
                    continue
    return entries


def load_communication_window():
    """
    Parse ground station satellite visibility files to find communication
    window duration (how long the satellite is over the ground station).
    Returns total visible time in seconds.
    """
    import glob
    pattern = os.path.join(
        RUN_DIR, "logs_ns3",
        "ground_station_satellites_in_range_time_step_*.txt"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        return None

    visible_intervals = []
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                # Format: time_ns, gs_id, sat_id, in_range (0/1)
                if len(parts) >= 4 and parts[3].strip() == "1":
                    try:
                        visible_intervals.append(int(parts[0]))
                    except ValueError:
                        continue

    if not visible_intervals:
        return None

    window_ns = max(visible_intervals) - min(visible_intervals)
    return window_ns / 1e9  # seconds


def print_topology():
    """Print communication topology: nodes and active ISL links."""
    path = os.path.join(RUN_DIR, "logs_ns3", "isl_utilization.csv")
    if not os.path.exists(path):
        print("  Topology data not available (isl_utilization.csv missing)")
        return

    link_ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    link_ids.add(int(parts[1]))
                except ValueError:
                    continue

    print(f"  RQ node (satellite) : node 0")
    print(f"  AC node (ground)    : node 1")
    print(f"  ILS links tracked   : {len(link_ids)}")
    print(f"  Link IDs            : {sorted(link_ids)[:10]}{'...' if len(link_ids) > 10 else ''}")


def analyse():
    burst_to_model = load_burst_schedule()
    incoming       = load_incoming()
    compute_delays = load_compute_delays()

    if not incoming:
        return

    latencies = defaultdict(list)
    delivered = defaultdict(int)
    total     = defaultdict(int)

    for parts in incoming:
        try:
            burst_id     = int(parts[0])
            recv_pkts    = int(parts[1])
            exp_pkts     = int(parts[2])
            avg_delay_ns = float(parts[5]) if len(parts) > 5 else 0.0
        except (ValueError, IndexError):
            continue

        model = burst_to_model.get(burst_id, "unknown")
        total[model]     += exp_pkts
        delivered[model] += recv_pkts
        if avg_delay_ns > 0:
            latencies[model].append(avg_delay_ns / 1e6)

    original_bytes = 512 * 512 * 3

    # --- Communication Topology ---
    print("=== Communication Topology ===\n")
    print_topology()

    # --- Communication Window ---
    print("\n=== Communication Window (Satellite Visibility over NCSU) ===\n")
    window_s = load_communication_window()
    if window_s is not None:
        print(f"  Visible window: {window_s:.1f} s")
    else:
        print("  Visibility data not available (simulation may not have generated it)")

    # --- ILS Utilization ---
    print("\n=== ILS (Inter-Satellite Link) Utilization ===\n")
    isl = load_isl_utilization()
    if isl:
        utils = [u for _, _, u in isl]
        print(f"  Mean utilization : {sum(utils)/len(utils)*100:.1f}%")
        print(f"  Peak utilization : {max(utils)*100:.1f}%")
        print(f"  Intervals logged : {len(isl)}")
    else:
        print("  ISL data not available")

    # --- Per-Model Results ---
    print("\n=== Per-Model Results (RQ Node -> AC Node) ===\n")
    print(f"{'Model':<10} {'Bytes':>8} {'Ratio':>8} {'Delivered':>10} "
          f"{'Drop%':>7} {'Net_lat':>9} {'Compute':>9} {'E2E_lat':>9}")
    print("-" * 78)

    for m in MODELS:
        name  = m["name"]
        ratio = (original_bytes * 8) / m["bits"]
        tot   = total.get(name, 0)
        dlv   = delivered.get(name, 0)
        drop  = (1 - dlv / tot) * 100 if tot > 0 else 0
        lats  = latencies.get(name, [])
        net_lat  = statistics.mean(lats) if lats else float("nan")
        comp_ms  = compute_delays.get(name, 0.0)
        e2e_ms   = net_lat + comp_ms if lats else float("nan")

        print(f"{name:<10} {m['bytes']:>8} {ratio:>7.1f}x {dlv:>5}/{tot:<5} "
              f"{drop:>6.1f}% {net_lat:>8.2f}ms {comp_ms:>8.2f}ms {e2e_ms:>8.2f}ms")

    print("\nColumns:")
    print("  Net_lat  = network latency (propagation + queuing, from Hypatia)")
    print("  Compute  = RQ node encoding time (from profile_compute_delay.py)")
    print("  E2E_lat  = total end-to-end = Compute + Net_lat")
    print("\n  Lower E2E + 0% drop = model viable for this orbital configuration")
    print("  High drop%           = compressed payload too large for available bandwidth")


if __name__ == "__main__":
    analyse()
