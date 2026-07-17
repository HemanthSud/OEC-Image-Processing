"""
Central configuration for the OEC RQ-NAC network simulation.

All symbols map 1:1 to the Overleaf formulation (OEC_RQ_NAC.pdf):
  S  satellites            -> CONSTELLATION (Hypatia Kuiper-630 shell 1)
  G  ground base stations  -> GROUND_STATIONS
  T, dt                    -> N_SLOTS, SLOT_S
  E_t, C_ij(t)             -> topology.py (LOS + range gated ISLs, elevation
                              gated GSLs), ISL_RATE_BPS / GS_RATE_BPS
  K  tasks                 -> tasks.py (AOI-triggered arrivals)
  D  = {2, 4, 8, 16}       -> DEPTHS, with b_q, t_enc_q, u_q per depth
"""

import math

# ── Physical constants ────────────────────────────────────────────────────────
R_E     = 6_371.0          # Earth radius, km
MU      = 398_600.4418     # gravitational parameter, km^3 s^-2
OMEGA_E = 7.2921150e-5     # Earth rotation rate, rad s^-1
C_KM_S  = 299_792.458      # speed of light, km s^-1
DEG     = math.pi / 180.0

# ── Constellations (parameters as used by Hypatia's satgenpy) ─────────────────
# name: (n_planes, sats_per_plane, altitude_km, inclination_deg, phasing_F)
CONSTELLATIONS = {
    'kuiper-630':   (34, 34, 630.0, 51.9, 1),   # Kuiper shell 1, 1156 sats
    'starlink-550': (72, 22, 550.0, 53.0, 1),   # Starlink shell 1, 1584 sats
    'telesat-1015': (27, 13, 1015.0, 98.98, 1), # Telesat T1, 351 sats
    'small-walker': (2, 3, 550.0, 53.0, 1),     # legacy 6-sat debug scenario
}
CONSTELLATION_NAME = 'kuiper-630'

N_PLANES, SATS_PER_PLANE, ALT_KM, INCL_DEG, PHASING_F = \
    CONSTELLATIONS[CONSTELLATION_NAME]
N_SATS = N_PLANES * SATS_PER_PLANE

A_SMA  = R_E + ALT_KM                       # semi-major axis, km
N_MM   = math.sqrt(MU / A_SMA ** 3)         # mean motion, rad s^-1
T_ORB  = 2.0 * math.pi / N_MM               # orbital period, s

# ── Link feasibility (Xuanhao's line-of-sight + distance constraints) ─────────
LOS_GRAZE_KM   = 80.0      # ISL blocked if ray grazes below R_E + 80 km
ISL_MAX_KM     = 5016.0    # max laser ISL range (Hypatia Starlink figure)
MIN_ELEV_DEG   = 20.0      # GSL minimum elevation angle
AOI_ELEV_DEG   = 40.0      # satellite can image an AOI above this elevation

# ── Link capacities ───────────────────────────────────────────────────────────
ISL_RATE_BPS   = 100e6     # per-ISL laser link rate
GS_RATE_BPS    = 2e6       # per-GBS aggregate receive rate (S-band budget,
                           # single receive chain shared over visible sats)

# ── Simulation window ─────────────────────────────────────────────────────────
SLOT_S      = 30           # scheduling slot dt (s)
SIM_S       = 18_000       # 5 h  (~3.1 orbital periods; old scenario was 1)
N_SLOTS     = SIM_S // SLOT_S + 1

# ── Ground base stations G (from Hypatia's top-100 city list) ────────────────
GROUND_STATIONS = [
    {'name': 'Tokyo',     'lat':  35.6895, 'lon':  139.6917},
    {'name': 'New-York',  'lat':  40.7127, 'lon':  -74.0059},
    {'name': 'Sao-Paulo', 'lat': -23.5475, 'lon':  -46.6361},
    {'name': 'Sydney',    'lat': -33.8678, 'lon':  151.2073},
]
N_GS = len(GROUND_STATIONS)

# ── Areas of interest (fixed observation targets that generate tasks) ─────────
AOIS = [
    {'name': 'California-fires', 'lat':  37.0, 'lon': -120.0},
    {'name': 'Amazon-basin',     'lat':  -3.0, 'lon':  -60.0},
    {'name': 'Sahel',            'lat':  15.0, 'lon':   10.0},
    {'name': 'Ganges-delta',     'lat':  23.0, 'lon':   90.0},
    {'name': 'Barrier-reef',     'lat': -18.0, 'lon':  147.0},
    {'name': 'Dnipro-basin',     'lat':  49.0, 'lon':   32.0},
    {'name': 'Tohoku-coast',     'lat':  38.3, 'lon':  141.0},
    {'name': 'Central-Andes',    'lat': -33.0, 'lon':  -70.0},
]

# ── Task model K ──────────────────────────────────────────────────────────────
AOI_COOLDOWN_S   = 1_800   # min gap between two tasks from the same AOI
TASK_IMAGES_MIN  = 100_000 # N_k lower bound (512x512 tiles of a large scene)
TASK_IMAGES_MAX  = 300_000 # N_k upper bound
TASK_DEADLINE_MIN_S = 1_800
TASK_DEADLINE_MAX_S = 3_600
TASK_WEIGHTS     = [1, 2, 3]        # importance w_k, drawn uniformly
FRESHNESS_ALPHA  = 1.0 / 300.0      # phi_k decay rate after soft deadline
MAX_TASKS        = 64               # cap for MILP tractability
RNG_SEED         = 42

# ── RQ-VAE depth table D = {1,2,4,8} ─────────────────────────────────────────
# (the Overleaf doc writes D = {2,4,8,16}; set to {1,2,4,8} per Hemanth,
#  matching the FLAIR 8x8-latent truncation evaluation)
# b_q  : payload per image = 88 B per residual stage (measured: depth-1 88 B,
#        depth-8 704 B on FLAIR-1 512x512, 8x8 latent grid)
# t_enc: measured single-image encoder forward 12.34 ms (constant across q —
#        quantization stages are negligible vs the conv encoder)
# u_q  : MEASURED from the FLAIR depth-16 model truncated to the first q
#        stages (truncation_eval.txt, 7050 val images): u_q = 1 - LPIPS_q.
#        LPIPS 0.5948 / 0.5570 / 0.5314 / 0.5171 at q = 1 / 2 / 4 / 8.
#        Swap for downstream task metrics (mIoU / F1) once evaluated.
DEPTHS = [1, 2, 4, 8]
PAYLOAD_B  = {q: 88 * q for q in DEPTHS}
ENC_S_PER_IMG = 12.34e-3
ENC_IMGS_PER_S = 1.0 / ENC_S_PER_IMG        # ~81 img/s per satellite
UTILITY = {1: 0.4052, 2: 0.4430, 4: 0.4686, 8: 0.4829}

# ── MPC scheduler ─────────────────────────────────────────────────────────────
MPC_HORIZON_SLOTS = 60     # H = 60 slots x 30 s = 30 min lookahead
MPC_LAMBDA_QUEUE  = 1e-4   # lambda_1, backlog penalty weight
MPC_RESOLVE_EVERY = 5      # re-solve at least every 5 slots (and on arrivals)

# ── Output ────────────────────────────────────────────────────────────────────
import os
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'oec_scenario')
OUT_DIR = os.path.abspath(OUT_DIR)
