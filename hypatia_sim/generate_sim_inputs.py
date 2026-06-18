"""
Generate Hypatia simulation inputs to evaluate RQ-VAE compression models
under realistic LEO satellite downlink constraints.

This script produces:
  - udp_burst_schedule.csv  : one burst per RQ-VAE model per image
  - config_ns3.properties   : ns-3 simulation config
  - ground_stations.txt     : ground station definition (NCSU)

Usage (after installing Hypatia and building ns-3):
    python3 generate_sim_inputs.py
    # then run from ns3-sat-sim/simulator:
    # ./waf --run="main_satnet --run_dir='<abs_path_to_run_dir>'"

Simulation design
-----------------
Each RQ-VAE model compresses one 512x512 RGB aerial image into a different
number of integer codes. We model each compressed image as a single UDP burst
from a satellite node to a ground station. The simulator reports delivery
latency and queue drops for each compression level, letting us compare which
compression depth is actually viable under real orbital bandwidth/latency.

Model parameters
----------------
Codebook size : 2048 entries  -> 11 bits per code  (log2(2048))
Image size    : 512 x 512 x 3 x 8 = 6,291,456 bits (original)

Satellite configuration (Starlink-like)
---------------------------------------
GSL bandwidth : 100 Mbps  (conservative real-world LEO downlink)
ISL bandwidth : 10 Gbps   (laser cross-links)
Altitude      : ~550 km   -> ~3.6 ms one-way propagation to ground
"""

import csv
import os
import math

# ---------------------------------------------------------------------------
# RQ-VAE model definitions
# ---------------------------------------------------------------------------
BITS_PER_CODE = math.ceil(math.log2(2048))  # 11 bits

MODELS = [
    {"name": "8x8x1",  "codes": 8 * 8 * 1},
    {"name": "8x8x2",  "codes": 8 * 8 * 2},
    {"name": "8x8x4",  "codes": 8 * 8 * 4},
    {"name": "8x8x8",  "codes": 8 * 8 * 8},
    {"name": "8x8x16", "codes": 8 * 8 * 16},
]

ORIGINAL_BITS = 512 * 512 * 3 * 8  # 6,291,456

for m in MODELS:
    m["bits"]             = m["codes"] * BITS_PER_CODE
    m["bytes"]            = math.ceil(m["bits"] / 8)
    m["compression_ratio"] = ORIGINAL_BITS / m["bits"]

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
# Number of images to simulate per model
N_IMAGES = 100

# Satellite node ID (source) — Starlink 550km constellation has 1584 satellites
# (72 orbits x 22 sats). Node 0 is a valid satellite; Hypatia routes dynamically
# through whatever satellite is overhead at each time step.
SAT_NODE_ID = 0

# Ground station node ID — in Hypatia, ground stations follow satellites:
# nodes 1584..1683 for the top-100 city list.
# We use New York (city ID 9, the closest top-100 city to NCSU Raleigh NC).
# Node ID = 1584 + 9 = 1593.
GS_NODE_ID = 1593

# GSL bandwidth (Mbps) — 100 Mbps is a realistic Starlink Gen2 downlink
GSL_DATA_RATE_MBPS = 100.0

# ISL bandwidth (Mbps)
ISL_DATA_RATE_MBPS = 10000.0

# Queue sizes (packets)
QUEUE_SIZE_PKTS = 1000

# Simulation duration — enough to send all bursts with margin
# Longest model: 8x8x16 = 1024 codes * 11 bits = 11264 bits = 1408 bytes
# At 100 Mbps: 1408 bytes / (100e6/8) = ~0.11 ms per image
# 100 images = ~11 ms; use 10 seconds for plenty of margin
SIM_DURATION_NS = 10_000_000_000  # 10 seconds

# Inter-burst gap (ns) — space bursts 1 ms apart so they don't pile up
BURST_GAP_NS = 1_000_000  # 1 ms

# Dynamic state update interval
DYNAMIC_STATE_UPDATE_NS = 100_000_000  # 100 ms

# Output directory
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run")

# Path to satellite network state (relative to run_dir, pointing at Hypatia
# satellite_networks_state generated data). Update this once Hypatia is set up.
SAT_NETWORK_DIR = "../../../../hypatia/paper/satellite_networks_state/gen_data/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
SAT_ROUTES_DIR  = SAT_NETWORK_DIR + "/dynamic_state_100ms_for_200s"


def bits_to_mbps(bits, duration_ns):
    """Convert a total bit count and duration (ns) to Mbps."""
    duration_s = duration_ns / 1e9
    return bits / duration_s / 1e6


def load_compute_delays():
    """
    Load per-model compute delays (ns) from profile_compute_delay.py output.
    Returns dict: model_name -> delay_ns. Falls back to 0 if file missing.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compute_delays.txt")
    delays = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if "," in line:
                    name, ns = line.split(",", 1)
                    # Map output dir name to model name (e.g. flair-rqvae-8x8x1 -> 8x8x1)
                    for m in MODELS:
                        if m["name"] in name:
                            delays[m["name"]] = int(ns)
    return delays


def generate_udp_burst_schedule():
    """
    Generate udp_burst_schedule.csv.

    Each burst represents one compressed image being downlinked from the
    RQ (satellite) node to the AC (ground/access) node after on-orbit
    compression. Burst start times are offset by the RQ node computing
    delay so the simulation reflects true end-to-end latency.

    UDP burst format:
        burst_id, from_node, to_node, target_rate_mbps,
        start_time_ns, duration_ns, additional_params, metadata
    """
    compute_delays = load_compute_delays()
    rows = []
    burst_id = 0
    cumulative_ns = 0

    for model in MODELS:
        duration_ns   = int((model["bytes"] * 8 / (GSL_DATA_RATE_MBPS * 1e6)) * 1e9)
        duration_ns   = max(duration_ns, 1000)
        compute_delay = compute_delays.get(model["name"], 0)

        for i in range(N_IMAGES):
            # Start = cumulative offset + RQ node computing delay
            start_ns = cumulative_ns + compute_delay
            rows.append([
                burst_id,
                SAT_NODE_ID,   # RQ node
                GS_NODE_ID,    # AC node
                GSL_DATA_RATE_MBPS,
                start_ns,
                duration_ns,
                "",
                f"model={model['name']},image={i},rq_node={SAT_NODE_ID},ac_node={GS_NODE_ID},compute_delay_ns={compute_delay}"
            ])
            burst_id    += 1
            cumulative_ns = start_ns + duration_ns + BURST_GAP_NS

    path = os.path.join(OUT_DIR, "udp_burst_schedule.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} bursts to {path}")
    return rows


def generate_ground_stations():
    """
    ground_stations.txt — NCSU coordinates as the receiving ground station.

    Format: id,name,latitude_deg,longitude_deg,elevation_m
    (Hypatia auto-computes Cartesian coordinates from lat/lon/elev)
    """
    # NCSU Raleigh, NC
    stations = [
        (0, "NCSU-Raleigh", 35.7796, -78.6382, 0),
    ]
    path = os.path.join(OUT_DIR, "ground_stations.txt")
    with open(path, "w") as f:
        for s in stations:
            f.write(f"{s[0]},{s[1]},{s[2]},{s[3]},{s[4]}\n")
    print(f"Wrote {len(stations)} ground stations to {path}")


def generate_config():
    """
    config_ns3.properties — ns-3 simulation configuration.
    """
    total_bursts = len(MODELS) * N_IMAGES
    # Estimate end time: last burst start + duration + 1 second margin
    # rough upper bound
    config = f"""simulation_end_time_ns={SIM_DURATION_NS}
simulation_seed=123456789

satellite_network_dir="{SAT_NETWORK_DIR}"
satellite_network_routes_dir="{SAT_ROUTES_DIR}"
dynamic_state_update_interval_ns={DYNAMIC_STATE_UPDATE_NS}

isl_data_rate_megabit_per_s={int(ISL_DATA_RATE_MBPS)}
gsl_data_rate_megabit_per_s={int(GSL_DATA_RATE_MBPS)}
isl_max_queue_size_pkts={QUEUE_SIZE_PKTS}
gsl_max_queue_size_pkts={QUEUE_SIZE_PKTS}

enable_isl_utilization_tracking=true
isl_utilization_tracking_interval_ns=1000000000

tcp_socket_type=TcpNewReno

enable_udp_burst_scheduler=true
udp_burst_schedule_filename="udp_burst_schedule.csv"
"""
    path = os.path.join(OUT_DIR, "config_ns3.properties")
    with open(path, "w") as f:
        f.write(config)
    print(f"Wrote config to {path}")


def print_summary():
    print("\n=== RQ-VAE Compression Model Summary ===")
    print(f"{'Model':<10} {'Codes':>6} {'Bits':>10} {'Bytes':>8} {'Ratio':>8}")
    print("-" * 48)
    for m in MODELS:
        print(f"{m['name']:<10} {m['codes']:>6} {m['bits']:>10} {m['bytes']:>8} {m['compression_ratio']:>7.1f}x")
    print(f"\nOriginal image: {ORIGINAL_BITS:,} bits ({ORIGINAL_BITS//8:,} bytes)")
    print(f"\nSimulation: {N_IMAGES} images per model, {len(MODELS)} models")
    print(f"Total bursts: {N_IMAGES * len(MODELS)}")
    print(f"GSL bandwidth: {GSL_DATA_RATE_MBPS} Mbps")
    print(f"Simulation duration: {SIM_DURATION_NS/1e9:.1f} seconds")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "logs_ns3"), exist_ok=True)

    print_summary()
    print()
    generate_udp_burst_schedule()
    generate_ground_stations()
    generate_config()

    print(f"\n=== Files written to {OUT_DIR} ===")
    print("\nNext steps:")
    print("1. Install Hypatia: https://github.com/snkas/hypatia")
    print("2. Generate Starlink satellite network state:")
    print("   cd hypatia/paper/satellite_networks_state")
    print("   python3 main_helper.py")
    print("3. Update SAT_NETWORK_DIR in this script to point at the generated state")
    print("4. Build ns-3:")
    print("   cd hypatia/ns3-sat-sim/simulator")
    print("   ./waf configure --build-profile=optimized")
    print("   ./waf")
    print("5. Run simulation:")
    abs_run_dir = os.path.abspath(OUT_DIR)
    print(f"   ./waf --run=\"main_satnet --run_dir='{abs_run_dir}'\"")
    print("6. Analyse results:")
    print("   python3 analyse_results.py")


if __name__ == "__main__":
    main()
