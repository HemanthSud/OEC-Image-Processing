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
# Two named regimes (Explore-agent finding, 2026-08-20): at the original
# gbs-limited rates, the per-GBS aggregate budget is >=12x tighter than any
# ISL segment a route could cross, so ISL contention is mathematically
# unreachable (max measured ISL utilization ~0.08) and link-cost-aware
# routing (routing.py) has nothing to act on. 'fabric-limited' rebalances
# rates so ISL segments can genuinely saturate, while keeping the identical
# topology/task model, so routing decisions have a measurable effect.
_SCENARIOS = {
    'gbs-limited':    dict(ISL_RATE_BPS=100e6, GSL_RATE_BPS=100e6,
                            GS_RATE_BPS=2e6,    LOAD_SCALE=1.0),
    'fabric-limited': dict(ISL_RATE_BPS=100e6, GSL_RATE_BPS=20e6,
                            GS_RATE_BPS=200e6,  LOAD_SCALE=20.0),
}


def apply_scenario(name):
    """Set the module-level capacity/load globals for a named regime.
    Callable at runtime (sweep.py, ablations) as well as at import — every
    other module reads these as C.<NAME> attribute lookups, so mutating
    them here propagates everywhere without a reimport."""
    global SCENARIO, ISL_RATE_BPS, GSL_RATE_BPS, GS_RATE_BPS, LOAD_SCALE
    global TASK_IMAGES_MIN, TASK_IMAGES_MAX
    sc = _SCENARIOS[name]
    SCENARIO = name
    ISL_RATE_BPS = sc['ISL_RATE_BPS']   # per-ISL laser link rate
    GSL_RATE_BPS = sc['GSL_RATE_BPS']   # per (satellite, GBS) downlink rate
    GS_RATE_BPS  = sc['GS_RATE_BPS']    # per-GBS aggregate receive budget
                                         # (sum over all sats downlinking to it)
    LOAD_SCALE   = sc['LOAD_SCALE']     # multiplies TASK_IMAGES_* so offered
                                         # load keeps the same utilization
                                         # target under 'fabric-limited' rates
    TASK_IMAGES_MIN = int(100_000 * LOAD_SCALE)
    TASK_IMAGES_MAX = int(300_000 * LOAD_SCALE)


SCENARIO = 'gbs-limited'    # 'gbs-limited' (published FORMULATION.md numbers)
                             # or 'fabric-limited' (routing.py ablations)
apply_scenario(SCENARIO)


import contextlib


@contextlib.contextmanager
def config_override(**kw):
    """Temporarily mutate module-level config globals (e.g. GS_RATE_BPS,
    RNG_SEED, MPC_ROUTE_ITERS) and restore them on exit. Used by sweep.py
    and by ablations that need several regimes in one process. 'scenario'
    is a special key that calls apply_scenario() instead of a raw setattr.
    """
    import sys
    mod = sys.modules[__name__]
    scenario = kw.pop('scenario', None)
    saved = {k: getattr(mod, k) for k in kw}
    saved_scenario_state = None
    if scenario is not None:
        saved_scenario_state = {k: getattr(mod, k) for k in
                                (list(_SCENARIOS[scenario]) +
                                 ['SCENARIO', 'TASK_IMAGES_MIN', 'TASK_IMAGES_MAX'])}
    try:
        for k, v in kw.items():
            setattr(mod, k, v)
        if scenario is not None:
            apply_scenario(scenario)
        yield mod
    finally:
        for k, v in saved.items():
            setattr(mod, k, v)
        if saved_scenario_state is not None:
            for k, v in saved_scenario_state.items():
                setattr(mod, k, v)

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
# TASK_IMAGES_MIN/MAX are set by apply_scenario() above (scaled by LOAD_SCALE)
AOI_COOLDOWN_S   = 1_800   # min gap between two tasks from the same AOI
TASK_DEADLINE_MIN_S = 1_800
TASK_DEADLINE_MAX_S = 3_600
TASK_WEIGHTS     = [1, 2, 3]        # importance w_k, drawn uniformly
FRESHNESS_ALPHA  = 1.0 / 300.0      # phi_k decay rate after soft deadline
MAX_TASKS        = 64               # cap for MILP tractability
RNG_SEED         = 42

# ── Timeliness objective (Xuanhao's ask: delay-related terms, not just the
#    freshness multiplier) ─────────────────────────────────────────────────
DEADLINE_HARD     = False   # if True, no y[k,q,tau] created past d_k at all
MPC_LAMBDA_LATE   = 5e-2    # weight on the explicit tardiness penalty
TARDINESS_REF_S   = 300.0   # normalizes lateness to "deadline units"

# ── Rotting / anti-starvation (Phase 4 upper level; also used by the flat
#    MPC's backpressure term) ────────────────────────────────────────────────
# Tuned by a small grid search (2026-08-20) against flat-MPC utility on the
# default scenario: PHI_DROP 0.05->0.01, HIER_THETA_ADMIT 0.9->0.5 closed
# ~1 utility point of mpc-hier's gap to the flat scheduler (60.71->61.4,
# 93.4%->95.1% delivered) without materially changing solve time. Still a
# first-pass tuning, not an exhaustive search -- see FORMULATION.md.
PHI_DROP    = 0.01    # freshness floor: below this with backlog left, drop
T_ABANDON_S = 3_600    # hard abandonment horizon past deadline
AGING_ETA   = 2.0      # anti-starvation backpressure weight (Lyapunov style)
ADMISSION_ON = True    # reject at arrival if not achievable at cheapest depth
HIER_THETA_ADMIT = 0.5

# ── MPC-predicted link costs + per-tau Dijkstra (routing.py) ────────────────
MPC_ROUTE_ITERS   = 3       # outer fixed-point rounds (0 => today's static)
MPC_ROUTE_BETA    = 1.0     # congestion-penalty weight
MPC_ROUTE_D_REF_KM = 1000.0 # "one nominal hop" of congestion penalty
MPC_ROUTE_RHO_MAX = 0.99    # clip utilization (never delete an edge)
MPC_ROUTE_TOL     = 0.02    # convergence: max |delta rho| across edges
MPC_ROUTE_STRIDE  = 1       # recompute routes every N slots inside horizon
MPC_ROUTE_NPATHS  = 1       # candidate paths per (task, tau) — 1 matches
                             # the literal advisor ask (predict cost, re-run
                             # Dijkstra); multipath (>1) is a documented
                             # future extension, not implemented here
MPC_REPLAN_TIME_BUDGET_S = 5.0

# ── Hierarchical MPC (Phase 4) ───────────────────────────────────────────────
HIER_MACRO_SLOTS   = 20     # upper-level aggregation (10 min)
HIER_LOW_HORIZON   = 20     # lower-level horizon, slots
HIER_GAMMA_EWMA     = 0.9   # aggregation-correction learning rate


# ── RQ-VAE depth table D = {2,4,8,16} ────────────────────────────────────────
# (matches the Overleaf doc's D = {2,4,8,16}; depth-16 finished training
#  2026-08 and its truncation metrics are measured, so the {1,2,4,8} stand-in
#  used while depth-16 was still training is retired.)
# b_q  : payload per image = 88 B per residual stage (measured on FLAIR-1
#        512x512, 8x8 latent grid)
# t_enc: measured single-image encoder forward 12.34 ms (constant across q —
#        quantization stages are negligible vs the conv encoder)
# u_q  : MEASURED from the FLAIR depth-16 model truncated to the first q
#        stages (truncation_eval.txt, 7050 val images): u_q = 1 - LPIPS_q.
#        LPIPS 0.5570 / 0.5314 / 0.5171 / 0.5039 at q = 2 / 4 / 8 / 16.
#        Swap for downstream task metrics (mIoU / F1) once evaluated — with
#        this reconstruction-based u_q the spread is only +12% (q=2 -> q=16)
#        against an 8x payload spread, so depth choice is driven mainly by
#        deliverability, not quality; flagged in the research-risk note.
DEPTHS = [2, 4, 8, 16]
PAYLOAD_B  = {q: 88 * q for q in DEPTHS}
ENC_S_PER_IMG = 12.34e-3
ENC_IMGS_PER_S = 1.0 / ENC_S_PER_IMG        # ~81 img/s per satellite
UTILITY = {2: 0.4430, 4: 0.4686, 8: 0.4829, 16: 0.4961}

# ── MPC scheduler ─────────────────────────────────────────────────────────────
MPC_HORIZON_SLOTS = 60     # H = 60 slots x 30 s = 30 min lookahead
MPC_LAMBDA_QUEUE  = 1e-4   # lambda_1, early-delivery backlog weight
MPC_RESOLVE_EVERY = 5      # re-solve at least every 5 slots (and on arrivals)

# ── Output ────────────────────────────────────────────────────────────────────
import os
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'oec_scenario')
OUT_DIR = os.path.abspath(OUT_DIR)
