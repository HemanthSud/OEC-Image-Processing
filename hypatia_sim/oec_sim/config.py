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
    # Retuned 2026-08: the previous setting (ISL 100 Mbps, GSL 20, GS 200,
    # LOAD_SCALE 20) did NOT make the fabric the bottleneck. Measured ISL
    # utilization under it was 0.01 -- its ISL rate was 5x its GSL rate, so
    # the GSL bound first, and LOAD_SCALE=20 pushed the run into being
    # ENCODER-limited (81 img/s/sat cannot even encode one task inside its
    # deadline at that scale), leaving 4.6% delivery, 100% deadline
    # violations and undefined delays -- degenerate, not congested.
    # Now: scarce ISLs, generous GSL/GBS, nominal load. LOAD_SCALE stays 1
    # because 81 img/s/sat means anything above ~1 cannot encode a task
    # inside its deadline, which is an encoder limit, not a fabric one.
    # Rate picked from the measured offered per-ISL demand over the full
    # 601-slot window (busiest 3.43 Mbps, p95 1.57, mean 0.89):
    #   2.0 Mbps ->  1.8% of (slot, ISL) pairs oversubscribed  (too loose --
    #                both routing couplings reproduced their baselines
    #                bit-for-bit, i.e. nothing to route around)
    #   1.0 Mbps -> 29.6% oversubscribed  <-- chosen: real hot spots, and
    #                ~70% of links still hold slack for a router to move on to
    #   0.5 Mbps -> 87.2% oversubscribed  (too uniform -- no alternative has
    #                capacity either, so routing again cannot help)
    'fabric-limited': dict(ISL_RATE_BPS=1e6,   GSL_RATE_BPS=50e6,
                            GS_RATE_BPS=500e6, LOAD_SCALE=1.0),
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
    is a special key that calls apply_scenario() instead of a raw setattr;
    'utility_mode' likewise routes to apply_utility_mode().
    """
    import sys
    mod = sys.modules[__name__]
    scenario = kw.pop('scenario', None)
    utility_mode = kw.pop('utility_mode', None)
    saved = {k: getattr(mod, k) for k in kw}
    saved_utility_state = None
    if utility_mode is not None:
        saved_utility_state = {k: getattr(mod, k) for k in _UTILITY_MODE_KEYS}
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
        if utility_mode is not None:
            apply_utility_mode(utility_mode)
        yield mod
    finally:
        for k, v in saved.items():
            setattr(mod, k, v)
        if saved_scenario_state is not None:
            for k, v in saved_scenario_state.items():
                setattr(mod, k, v)
        if saved_utility_state is not None:
            for k, v in saved_utility_state.items():
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
MPC_ROUTE_NPATHS  = 3       # candidate paths per (task, tau), generated by
                            # PredictiveRouter.build_multi (edge-penalized
                            # re-Dijkstra). Only the two-level schedulers read
                            # it; the single-path flat MPC is unaffected.
ROUTE_DIVERSITY_PENALTY = 0.5   # each earlier round's use of an edge
                                # multiplies its cost by (1 + this)
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


# ── Unified utility (utility.py; see FORMULATION.md "Unified utility") ────────
# The simulator historically scored every scheduler with a single factor,
#     U = sum_k sum_t w_k phi_k(t) u_q dr/N_k,
# while the MPC objective, the hierarchical levels and the oracle each
# optimized a *different* function. utility.py is now the single source of
# truth for all five; these weights select which terms are live.
#
#   UTIL_MODE = 'legacy'  reproduces the committed numbers exactly
#                         (mpc 64.95, greedy-fixed-8 64.15, ... ) and is the
#                         default so old results stay verifiable forever.
#   UTIL_MODE = 'unified' turns on the four-factor score: downstream
#                         segmentation quality, timeliness, concave coverage,
#                         and resource cost + a fairness floor.
import os as _os

MPC_LAMBDA_LATE_LEGACY = MPC_LAMBDA_LATE   # objective-only tardiness regularizer

UTIL_MODE           = 'legacy'
UTIL_W_QUALITY      = 1.0     # omega_Q
UTIL_W_TARDY        = 0.0     # omega_T
UTIL_W_COST         = 0.0     # omega_E
UTIL_W_FAIR         = 0.0     # omega_F
UTIL_WEIGHT_NORM    = False   # divide w_k by max(TASK_WEIGHTS)?
UTIL_QUALITY_SOURCE = 'lpips' # 'lpips' -> UTILITY, 'miou' -> quality table
UTIL_QUALITY_ANCHOR = 'floor' # 'floor' | 'ratio'  (see utility.quality_table)
UTIL_COVERAGE_BREAKS = [(1.0, 1.0)]   # [(width, slope)], concave: slopes down
UTIL_SEG_STRIDE     = 1       # index the coverage-segment columns every N
                              # horizon steps instead of every step. The
                              # freshness of a group is taken at its first
                              # step. 1 is exact; larger values trade a little
                              # objective fidelity for a much smaller MILP,
                              # which matters because the lambda columns
                              # multiply the branch-and-bound cost of the
                              # binary depth variables (measured: the solves
                              # where a depth is still undecided are ~30x the
                              # cost of ones where every depth is committed).
UTIL_W_COST_TX      = 0.5     # split of the cost term: payload-proportional
UTIL_W_COST_ENC     = 0.5     #                         ... and per-image fixed

# Energy accounting. Reported in absolute Joules; never used as an objective
# (the objective uses the normalized, unit-free cost_coeff instead).
# NOTE the finding this makes explicit: ENC_S_PER_IMG was measured at BOTH
# 8x8x1 and 8x8x8 as 12.34 ms (sigma ~0.03-0.05 ms), so encode time does not
# vary with depth, and encode energy (0.370 J/img) dominates downlink energy
# (2.25e-3 J/img at q=16) by ~164x. A literal Joule-denominated term is
# therefore nearly constant in q and cannot drive depth choice: depth
# selection in this system is a bandwidth decision, not an energy one.
ENC_S_PER_STAGE = 0.0      # MEASURED-as-zero: below the 0.05 ms noise floor
ENC_POWER_W     = 30.0     # ASSUMED: Jetson AGX Orin mid-band (15-60 W).
                           #   The A6000 the 12.34 ms was measured on (300 W)
                           #   is not a flight part.
TX_POWER_W      = 20.0     # ASSUMED: smallsat Ka-band SSPA + RF chain
TX_RATE_REF_BPS = 100e6    # reference rate the TX_POWER_W is drawn at
E_PER_BIT_J     = TX_POWER_W / TX_RATE_REF_BPS      # DERIVED: 2.0e-7 J/bit

# Downstream-task quality table. utility.load_quality_table() prefers this
# JSON (written by rq-vae/downstream/harvest_metrics.py on the server); the
# dict below is a clearly-marked PROVISIONAL stand-in so run_all works on a
# laptop. Any run using it says so in summary.txt -- never quote these as
# measured.
QUALITY_TABLE_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                   'quality_table.json')
QUALITY_TABLE_FALLBACK = {
    'source': 'PROVISIONAL - NOT MEASURED (placeholder until the FLAIR '
              'segmentation sweep lands; see rq-vae/downstream/)',
    'miou_ref': None, 'miou_floor': None, 'anchor': 'provisional',
    'miou': {}, 's': {2: 0.65, 4: 0.77, 8: 0.87, 16: 0.94},
}

ORACLE_SEG_AGG = 10        # coverage segments indexed on coarser windows than
                           # y in the offline bound (coarser => more freedom
                           # => still a valid upper bound), keeps it tractable


def apply_utility_mode(name):
    """Select the utility term weights. 'legacy' reproduces the committed
    single-factor score bit-for-bit; 'unified' turns on all four factors."""
    global UTIL_MODE, UTIL_W_QUALITY, UTIL_W_TARDY, UTIL_W_COST, UTIL_W_FAIR
    global UTIL_COVERAGE_BREAKS, UTIL_WEIGHT_NORM, UTIL_QUALITY_SOURCE
    global MPC_LAMBDA_LATE, UTIL_SEG_STRIDE
    if name not in ('legacy', 'unified'):
        raise ValueError(f'unknown utility mode {name!r}')
    UTIL_MODE = name
    if name == 'legacy':
        UTIL_W_QUALITY, UTIL_W_TARDY, UTIL_W_COST, UTIL_W_FAIR = 1.0, 0.0, 0.0, 0.0
        UTIL_COVERAGE_BREAKS = [(1.0, 1.0)]
        UTIL_WEIGHT_NORM = False
        UTIL_QUALITY_SOURCE = 'lpips'
        UTIL_SEG_STRIDE = 1          # inert at J = 1 anyway
        MPC_LAMBDA_LATE = MPC_LAMBDA_LATE_LEGACY
    else:
        UTIL_W_QUALITY, UTIL_W_TARDY, UTIL_W_COST, UTIL_W_FAIR = 1.0, 0.25, 0.05, 0.10
        # 4-segment discretization of g(rho) = (1-e^-3rho)/(1-e^-3):
        # widths sum to 1, sum(w*m) == 1, slopes strictly decreasing.
        UTIL_COVERAGE_BREAKS = [(.25, 2.00), (.25, 1.12), (.25, .60), (.25, .28)]
        UTIL_WEIGHT_NORM = True
        UTIL_QUALITY_SOURCE = 'miou'
        # Measured on the default scenario: stride 1 -> 33.1 s / utility
        # 24.248, stride 3 -> 14.0 s / 24.795, stride 5 -> 15.1 s / 24.780,
        # stride 10 -> 12.8 s / 24.780 (legacy baseline 10.1 s). Coarser
        # indexing is both faster AND slightly better here -- the smaller MILP
        # solves more reliably inside HiGHS's default effort -- so exact
        # per-step lambda columns buy nothing.
        UTIL_SEG_STRIDE = 5
        # the unified omega_T tardiness term subsumes the old objective-only
        # regularizer -- keeping both would double-count lateness.
        MPC_LAMBDA_LATE = 0.0


# globals apply_utility_mode() touches, so config_override can restore them
_UTILITY_MODE_KEYS = (
    'UTIL_MODE', 'UTIL_W_QUALITY', 'UTIL_W_TARDY', 'UTIL_W_COST',
    'UTIL_W_FAIR', 'UTIL_COVERAGE_BREAKS', 'UTIL_WEIGHT_NORM',
    'UTIL_QUALITY_SOURCE', 'MPC_LAMBDA_LATE', 'UTIL_SEG_STRIDE',
)


# ── Two-level MPC, peer coupling (twolevel.py, scheduler `mpc-2level`) ───────
MPC2L_ITERS        = 3      # routing-LP / depth-MILP exchange rounds
MPC2L_TOL          = 0.02   # convergence: relative L1 change in bit demand
MPC2L_ROUTE_HORIZON = 30    # routing horizon, slots (<= MPC_HORIZON_SLOTS;
                            # steps beyond it keep the static single path)
MPC2L_SHORTFALL_PENALTY = 1e3   # big-M on unrouted demand, keeps the LP feasible
MPC2L_DELAY_REF_S  = 0.05   # normalizes propagation delay in the routing cost


# ── Two-level MPC, hierarchical coupling (hier.py, `mpc-hier-route`) ─────────
HIER_ROUTE_ON          = True   # False makes mpc-hier-route == mpc-hier
HIER_ROUTE_MACRO_SLOTS = 20     # re-solve routing every 20 slots (10 min)
HIER_ROUTE_HORIZON     = 60     # routing lookahead, slots
HIER_ROUTE_WINDOW      = 10     # macro-window size for the routing LP
HIER_ROUTE_NPATHS      = 3      # candidate paths per (task, window)
HIER_ROUTE_KEEP        = 2      # paths frozen per task for the epoch
