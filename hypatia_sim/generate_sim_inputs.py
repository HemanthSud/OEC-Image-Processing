"""
Generate Hypatia simulation inputs to evaluate RQ-VAE compression models
under realistic LEO satellite downlink constraints.

This script produces:
  - udp_burst_schedule.csv  : one burst per RQ-VAE model per image per station
  - config_ns3.properties   : ns-3 simulation config
  - ground_stations.txt      : reference list of the selected AC nodes

Usage (after the satellite state is generated and ns-3 is built):
    python3 generate_sim_inputs.py
    # then run from ns3-sat-sim/simulator:
    # ./waf --run="main_satnet --run_dir='<abs_path_to_run_dir>'"

Simulation design
-----------------
Each RQ-VAE model compresses one 512x512 RGB aerial image into a different
number of integer codes. We model each compressed image as a single UDP burst
from an imaging/compute satellite (RQ node) to a ground station (AC node). The
simulator reports delivery latency and queue drops for each compression level,
per ground station, letting us compare which compression depth is viable under
real orbital bandwidth/latency and how that varies by location.

Topology (see topology_config.py for the single source of truth)
----------------------------------------------------------------
- Constellation : Starlink-550, 1584 satellites, +Grid ISLs (4/sat, 3168 total)
- Ground stations: 5 chosen from Hypatia's EXISTING top-100 city list (no
                   state regeneration), spread by longitude
- RQ nodes      : all satellites are equal; one imaging/compute satellite is
                  assigned per ground station as the burst source
- Links         : GSL 100 Mbps, ISL 10 Gbps
"""

import csv
import os
import math

import topology_config as topo

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
    m["bits"]              = m["codes"] * BITS_PER_CODE
    m["bytes"]             = math.ceil(m["bits"] / 8)
    m["compression_ratio"] = ORIGINAL_BITS / m["bits"]

# ---------------------------------------------------------------------------
# Simulation parameters (topology pulled from topology_config)
# ---------------------------------------------------------------------------
N_IMAGES = 100   # images per (ground station, model)

SOURCE_SATELLITES  = topo.SOURCE_SATELLITES
GSL_DATA_RATE_MBPS = topo.GSL_DATA_RATE_MBPS
ISL_DATA_RATE_MBPS = topo.ISL_DATA_RATE_MBPS
QUEUE_SIZE_PKTS    = topo.QUEUE_SIZE_PKTS
DYNAMIC_STATE_UPDATE_NS = topo.DYNAMIC_STATE_UPDATE_NS

SIM_DURATION_NS = 10_000_000_000  # 10 seconds (ample for all bursts)
BURST_GAP_NS    = 1_000_000       # 1 ms spacing between a station's bursts

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run")

# Existing top-100 satellite state (already generated — no regeneration needed).
SAT_NETWORK_DIR = "../../../../hypatia/paper/satellite_networks_state/gen_data/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
SAT_ROUTES_DIR  = SAT_NETWORK_DIR + "/dynamic_state_100ms_for_200s"


def load_top100_stations():
    """
    Read the existing Hypatia top-100 ground station file.
    Returns list of (index, name, lat, lon). Falls back to a built-in list if
    the file is not reachable (e.g. running off-server for a dry test).
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), topo.TOP100_PATH)
    stations = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                stations.append((int(parts[0]), parts[1],
                                 float(parts[2]), float(parts[3])))
        return stations
    print(f"[warn] {path} not found — using fallback station list")
    return list(topo.FALLBACK_STATIONS)


def select_stations():
    """
    Choose NUM_GROUND_STATIONS AC nodes from the existing top-100 list.
    Returns list of (index, name, lat, lon) in selection order.
    """
    allst = load_top100_stations()
    n = topo.NUM_GROUND_STATIONS
    sel = topo.GROUND_STATION_SELECTION

    if isinstance(sel, (list, tuple)):
        chosen = [s for s in allst if s[0] in set(sel)]
    elif sel == "spread":
        # Even spacing across stations sorted by longitude -> global spread.
        by_lon = sorted(allst, key=lambda s: s[3])
        if len(by_lon) <= n:
            chosen = by_lon
        else:
            step = len(by_lon) / n
            chosen = [by_lon[int(i * step)] for i in range(n)]
    else:
        chosen = allst[:n]

    return chosen[:n]


# Resolve the AC node set once at import time.
STATIONS = select_stations()   # list of (index, name, lat, lon)


def load_compute_delays():
    """Load per-model compute delays (ns) from profile_compute_delay.py output."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compute_delays.txt")
    delays = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if "," in line:
                    name, ns = line.split(",", 1)
                    for m in MODELS:
                        if m["name"] in name:
                            delays[m["name"]] = int(ns)
    return delays


def generate_udp_burst_schedule():
    """
    Generate udp_burst_schedule.csv.

    For every ground station, every model sends N_IMAGES bursts from that
    station's assigned imaging/compute satellite (RQ node) down to the station
    (AC node). Burst start times are offset by the RQ node computing delay so
    the simulation reflects true end-to-end (compute + network) latency.

    UDP burst format:
        burst_id, from_node, to_node, target_rate_mbps,
        start_time_ns, duration_ns, additional_params, metadata
    """
    compute_delays = load_compute_delays()
    rows = []
    burst_id = 0

    for si, (st_index, st_name, _, _) in enumerate(STATIONS):
        ac_node = topo.gs_node_id(st_index)
        rq_node = SOURCE_SATELLITES[si % len(SOURCE_SATELLITES)]
        cumulative_ns = 0

        for model in MODELS:
            duration_ns   = int((model["bytes"] * 8 / (GSL_DATA_RATE_MBPS * 1e6)) * 1e9)
            duration_ns   = max(duration_ns, 1000)
            compute_delay = compute_delays.get(model["name"], 0)

            for i in range(N_IMAGES):
                start_ns = cumulative_ns + compute_delay
                rows.append([
                    burst_id,
                    rq_node,   # RQ node (imaging + compute satellite)
                    ac_node,   # AC node (ground station)
                    GSL_DATA_RATE_MBPS,
                    start_ns,
                    duration_ns,
                    "",
                    f"gs={st_name},model={model['name']},image={i},"
                    f"rq_node={rq_node},ac_node={ac_node},compute_delay_ns={compute_delay}"
                ])
                burst_id    += 1
                cumulative_ns = start_ns + duration_ns + BURST_GAP_NS

    path = os.path.join(OUT_DIR, "udp_burst_schedule.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} bursts to {path}")
    return rows


def generate_ground_stations_reference():
    """
    Write a reference list of the selected AC nodes. The simulation itself uses
    the ground stations baked into the generated satellite state; this file is
    only for the analysis scripts and human reference. Indices match the state.

    Format: top100_index,name,lat,lon,node_id
    """
    path = os.path.join(OUT_DIR, "ground_stations.txt")
    with open(path, "w") as f:
        for st_index, name, lat, lon in STATIONS:
            f.write(f"{st_index},{name},{lat},{lon},{topo.gs_node_id(st_index)}\n")
    print(f"Wrote {len(STATIONS)} selected ground stations to {path}")


def generate_config():
    """config_ns3.properties — ns-3 simulation configuration."""
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

    print("\n=== Topology (existing top-100 stations) ===")
    print(f"Constellation : {topo.CONSTELLATION['name']} "
          f"({topo.CONSTELLATION['num_satellites']} sats, "
          f"{topo.CONSTELLATION['total_isls']} ISLs)")
    print(f"{'Ground Station':<16} {'idx':>4} {'AC node':>8} {'RQ sat':>7} {'plane':>6}")
    print("-" * 46)
    for si, (st_index, name, _, _) in enumerate(STATIONS):
        rq = SOURCE_SATELLITES[si % len(SOURCE_SATELLITES)]
        plane, _ = topo.sat_plane_index(rq)
        print(f"{name:<16} {st_index:>4} {topo.gs_node_id(st_index):>8} {rq:>7} {plane:>6}")

    total = len(STATIONS) * len(MODELS) * N_IMAGES
    print(f"\nBursts: {len(STATIONS)} stations x {len(MODELS)} models "
          f"x {N_IMAGES} images = {total}")
    print(f"GSL bandwidth: {GSL_DATA_RATE_MBPS} Mbps  |  ISL bandwidth: {ISL_DATA_RATE_MBPS/1000:.0f} Gbps")
    print(f"Simulation duration: {SIM_DURATION_NS/1e9:.1f} seconds")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "logs_ns3"), exist_ok=True)

    print_summary()
    print()
    generate_udp_burst_schedule()
    generate_ground_stations_reference()
    generate_config()

    print(f"\n=== Files written to {OUT_DIR} ===")
    print("\nNext steps:")
    print("1. Summarise topology: python3 extract_topology.py --gen-dir \\")
    print("   ../hypatia/paper/satellite_networks_state/gen_data/"
          "starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls")
    print("2. Build + run ns-3:")
    abs_run_dir = os.path.abspath(OUT_DIR)
    print(f"   ./waf --run=\"main_satnet --run_dir='{abs_run_dir}'\"")
    print("3. Analyse: python3 analyse_results.py")


if __name__ == "__main__":
    main()
