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

# ── Areas of interest (fixed observation targets that generate tasks) ─────────
# _BASE_AOIS is what gbs-limited/fabric-limited use (unchanged, so their
# golden-file regression numbers don't drift). 'realistic' needs many more
# AOIs -- see _SCENARIOS['realistic']'s comment: with only these 8, task
# generation hard-caps at 4,808 regardless of AOI_COOLDOWN_S (a geometric
# visibility ceiling, not a cooldown one). _REALISTIC_EXTRA_AOIS adds 56
# synthetic points spread across the latitudes Kuiper-630 covers
# (-51.9..51.9 deg inclination) so 'realistic' alone can reach MAX_TASKS.
# AOIS itself is set by apply_scenario() below, scenario-scoped like
# MAX_TASKS/AOI_COOLDOWN_S.
_BASE_AOIS = [
    {'name': 'California-fires', 'lat':  37.0, 'lon': -120.0},
    {'name': 'Amazon-basin',     'lat':  -3.0, 'lon':  -60.0},
    {'name': 'Sahel',            'lat':  15.0, 'lon':   10.0},
    {'name': 'Ganges-delta',     'lat':  23.0, 'lon':   90.0},
    {'name': 'Barrier-reef',     'lat': -18.0, 'lon':  147.0},
    {'name': 'Dnipro-basin',     'lat':  49.0, 'lon':   32.0},
    {'name': 'Tohoku-coast',     'lat':  38.3, 'lon':  141.0},
    {'name': 'Central-Andes',    'lat': -33.0, 'lon':  -70.0},
]
_REALISTIC_EXTRA_AOIS = [
    {'name': f'synthetic-{i}',
     'lat': -50.0 + (i % 8) * 14.0, 'lon': -170.0 + (i // 8) * 34.0}
    for i in range(56)
]

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
                            GS_RATE_BPS=2e6,    LOAD_SCALE=1.0,
                            MAX_TASKS=64, AOI_COOLDOWN_S=1_800),
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
                            GS_RATE_BPS=500e6, LOAD_SCALE=1.0,
                            MAX_TASKS=64, AOI_COOLDOWN_S=1_800),
    # 'realistic' (2026-09-21, Xuanhao's ask): gbs-limited/fabric-limited were
    # both deliberately tuned to force a specific congestion pattern, not
    # drawn from real hardware -- a reviewer would reasonably ask why ISL is
    # 1 Mbps when real LEO laser ISLs run orders of magnitude higher. Values
    # below are Xuanhao's own nominal order-of-magnitude point, cross-checked
    # against public figures: Starlink/Kuiper laser ISL terminals are rated
    # up to ~100 Gbps each (SpaceX-reported network-wide realized average is
    # much lower, ~600 Mbps/link); Starlink Gen2's Ka-band satellite<->gateway
    # feeder link is reported around ~20 Gbps; a real deployed Starlink
    # gateway (FCC filing, Unalaska AK) is documented at 10 Gbps symmetric.
    # So 10 Gbps ISL / 1 Gbps GSL / 10 Gbps GBS is a defensible *conservative*
    # modern-LEO floor (real hardware peaks 10-20x higher on ISL/GSL), not a
    # literature-matched ceiling -- stated here explicitly rather than implied
    # as precise. GSL here means one satellite<->one-ground-station feeder
    # link, not an end-user terminal (topology_config.py's ~100 Mbps GSL
    # figure models the unrelated end-user downlink product, not this link).
    #
    # Workload (LOAD_SCALE/MAX_TASKS/AOI_COOLDOWN_S): calibration probe run
    # 2026-09-21 confirmed by measurement that MAX_TASKS=64 CANNOT produce
    # any link contention at Gbps-scale capacities, no matter how large
    # LOAD_SCALE gets -- at 81 img/s/sat (~0.91 Mbps/sat at depth 16), the
    # absolute ceiling on aggregate demand with only 64 tasks, even in the
    # limiting case where ALL of them are simultaneously active, is
    # 64 * 0.91 Mbps =~ 58 Mbps -- below even the *low* end of the ISL/GSL/
    # GBS sensitivity grid below (GSL as low as 0.5 Gbps). Confirmed
    # empirically: a probe run at LOAD_SCALE=1.0 produced greedy-adaptive
    # and greedy-fixed-16 as BIT-FOR-BIT IDENTICAL (both 100% delivery,
    # depth 16 for all 64 tasks, u_min/Jain identical) -- i.e. zero
    # scarcity anywhere. Raising LOAD_SCALE only lengthens each task's
    # encode duration, not its instantaneous per-slot rate, so it cannot
    # raise the aggregate-demand ceiling above ~58 Mbps regardless of how
    # high it goes. MAX_TASKS and AOI_COOLDOWN_S are therefore made
    # scenario-scoped (this dict), independent of gbs-limited/fabric-limited
    # (which keep their existing, separately-calibrated 64/1800 values so
    # neither regresses).
    #
    # Follow-up pass (also 2026-09-21): raising MAX_TASKS alone hit a SECOND
    # ceiling -- tasks.py's `cooldown_slots = AOI_COOLDOWN_S // SLOT_S` (30 s)
    # floors to 0 once AOI_COOLDOWN_S < 30 s, making shorter cooldowns a
    # no-op; the true limit becomes how many of the 601 slots each AOI has a
    # satellite in view at >= AOI_ELEV_DEG, a fixed geometric quantity per
    # AOI. With the original 8 AOIs this capped generation at 4,808 tasks no
    # matter how low AOI_COOLDOWN_S went. AOIS below therefore adds 56
    # synthetic points (spread across the latitudes Kuiper-630 actually
    # covers) so MAX_TASKS=16000 fully generates. Confirmed by measurement at
    # that point: peak concurrent encoder-bound demand ~4.9 Gbps system-wide,
    # ~42% (~2 Gbps) concentrated at the busiest of the 4 GBS -- comfortably
    # under the 10 Gbps nominal GBS budget (by design: nominal should NOT be
    # contended, that's a real result, not a gap) but right at the *low* end
    # of the GBS sensitivity grid below. Verified directly: at GS_RATE_BPS
    # forced to 2 Gbps, greedy-adaptive (3302.31) and greedy-fixed-16
    # (3627.62) diverge for the first time; at the 10 Gbps nominal point
    # they're still identical, as expected. MPC-family schedulers
    # (Flat-MPC/OrbitFlow/Two-Level-MPC) at MAX_TASKS=16000 are expensive --
    # Two-Level-MPC alone took ~99 s per run at just 150 tasks with heavy
    # superlinear growth -- so the actual multi-seed/sensitivity runs under
    # this scenario are expected to need the NCSU server, not this laptop,
    # and real wall-clock hours, not minutes (accepted tradeoff, 2026-09-21).
    'realistic':      dict(ISL_RATE_BPS=10e9,  GSL_RATE_BPS=1e9,
                            GS_RATE_BPS=10e9,  LOAD_SCALE=1.0,
                            MAX_TASKS=16000, AOI_COOLDOWN_S=8),
    # 'realistic-bigdata' (2026-10-05, Xuanhao's ask in reply to the ISL/GSL/
    # GBS sensitivity sweep above showing Flat-MPC/OrbitFlow/Two-Level-MPC
    # never separate, within 0.03% of each other): instead of growing task
    # COUNT to force ISL congestion (infeasible -- MILP solve cost ~n^2.65,
    # Two-Level-MPC alone was ~99s/run at just 150 tasks), grow each task's
    # DATA VOLUME directly via TASK_SIZE_MODE='data_gb'. A sibling of
    # 'realistic', not a mutation of it, so 'realistic' and its already-
    # committed results_realistic_{isl,gsl,gbs}_rate.csv stay untouched.
    # MAX_TASKS is deliberately far below 'realistic''s 16000: no MPC or
    # greedy scheduler here ever sets task.dropped (only rl_env.py's PPO path
    # does), so an oversized task never leaves the active set once it can't
    # finish inside SIM_S -- the active set grows monotonically, and active-
    # set size is exactly what multiplies MILP variable/row count at every
    # resolve. Running 16000 tasks AND multi-GB tasks together would be the
    # worst-of-both case (count-driven and sojourn-driven blowup at once).
    # TASK_DATA_GB_MIN/MAX are deliberately far below the literal "tens of
    # GB" Xuanhao suggested: at the reference depth used to invert GB into
    # n_images (TASK_SIZE_REF_DEPTH = max(DEPTHS) = 16, chosen because it
    # *minimizes* n_images, and therefore encode time, for a given target
    # size), PAYLOAD_B[16] * ENC_IMGS_PER_S ~= 1408 B * 81.07 img/s ~=
    # 114,106 B/s ~= 0.913 Mbps per source satellite, so a task's own encode
    # time ~= V_GB * 8,764 s ~= V_GB * 2.44 h. Even 10 GB (24.4 h) never
    # finishes inside SIM_S=18000s (5h), let alone a 1800-3600s soft deadline
    # -- the same degenerate shape already measured at LOAD_SCALE=20 above
    # (4.6% delivery, 100% deadline violations). 0.1-5.0 GB keeps the grid
    # (see sweep.py Part 12) biased toward the non-degenerate end where a
    # clean separation result is actually gettable, while still well above
    # today's 'realistic' baseline (100k-300k images * 1408 B = 140-422 MB
    # per task at depth 16).
    # KNOWN ISSUE (validation probe, 2026-10-06), unresolved -- do not treat
    # the MPC-vs-MPC comparison from this scenario's congested grid points as
    # trustworthy without revisiting this: at the congested end of the GB
    # range, Flat-MPC/OrbitFlow/Two-Level-MPC collapse to IDENTICAL,
    # worse-than-greedy behavior (16.1% delivery vs. greedy-fixed-16's 86%,
    # same ~4.83 utility to 3 decimals across all three MPC variants). Every
    # solve reports status='optimal' (not a solver bug). First hypothesis
    # tested -- the 30-min MPC_HORIZON_SLOTS window being too short to see a
    # multi-hour task's encode progress -- was tried at 240 slots (2h, 4x the
    # default) and REJECTED: delivery/utility came out bit-for-bit identical
    # to the 60-slot run, just 3x slower to solve, so horizon length isn't
    # the cause. At the grid's CONTROL point (0.1-0.3GB, tasks that fully
    # complete), Flat-MPC and greedy-fixed-16 are bit-for-bit identical too --
    # so the breakdown is specific to the scarcity regime, not a general MPC
    # bug. Leading hypothesis, not yet tested: the MILP objective
    # (formulation.tex's Flat-MPC objective, schedulers.py's y_{k,q,tau}/N_k
    # normalization) divides each task's reward by its own N_k -- harmless
    # when N_k is roughly uniform across tasks (the ~100k-300k regime this
    # was designed for), but with this scenario's ~48x size spread (73k-3.5M
    # images), it may structurally bias the solver away from ever
    # prioritizing a huge task's backlog once capacity is scarce, regardless
    # of what the horizon can see. This is a core-formulation question, not
    # a parameter to silently retune -- needs Hemanth's/Xuanhao's input
    # before the MPC-separation question this scenario was built to answer
    # can be considered resolved. The congestion-exists finding itself
    # (greedy-fixed-16's delivery dropping with GB range) is unaffected by
    # this and is still valid.
    'realistic-bigdata': dict(ISL_RATE_BPS=10e9,  GSL_RATE_BPS=1e9,
                               GS_RATE_BPS=10e9,  LOAD_SCALE=1.0,
                               MAX_TASKS=2000, AOI_COOLDOWN_S=8,
                               TASK_SIZE_MODE='data_gb',
                               TASK_DATA_GB_MIN=0.1, TASK_DATA_GB_MAX=5.0),
    # 'realistic-bigdata-routing' (2026-10-20, NORMALIZATION_FINDINGS.md
    # Finding 2): 'realistic-bigdata' can never show routing differentiation
    # because at Gbps rates a task's own ISL/GSL/GBS transmission time
    # (~0.8s/GB) is ~4 orders of magnitude faster than its encode time
    # (~2.44h/GB) -- every task is encoder-dead long before the network
    # could matter. This scenario fixes that by (a) scaling each task's
    # deadline to its OWN encode time plus slack (TASK_DEADLINE_MODE=
    # 'scaled', see tasks.py) instead of a fixed 30-60min window, so
    # encoding finishes with room to spare, and (b) dropping ISL/GSL/GBS to
    # 'fabric-limited'-style scarce rates so many tasks' encoded backlogs
    # genuinely compete for capacity instead of draining instantly. GB
    # range and MAX_TASKS: calibrated 2026-10-20 via a greedy-fixed-16 probe
    # (cheap, no MILP) over ISL/GSL/GBS candidates, same process used for
    # 'fabric-limited'/'realistic-bigdata'. Confirmed deadline scaling works
    # as intended: deadlines land well inside the 18,030s sim horizon
    # (3,032-10,061s observed, mean 6,573s) and 0/500 tasks exceed their own
    # encode ceiling (vs. ~100% at 'realistic-bigdata' sizes/deadlines).
    # Rate sweep at these deadlines: 100/200/300 Mbps -> 100% delivered (no
    # contention at all); 10/20/30 Mbps -> 73.2% (chosen: real contention,
    # not degenerate); 5/10/15 -> 41.0%; 2/5/8 -> 21.6%; 1/2/4 -> 10.8% (the
    # last two are too tight -- same shape as 'fabric-limited's own
    # 0.5 Mbps rejection, no alternative has capacity either).
    'realistic-bigdata-routing': dict(
        ISL_RATE_BPS=10e6, GSL_RATE_BPS=20e6, GS_RATE_BPS=30e6,
        LOAD_SCALE=1.0, MAX_TASKS=500, AOI_COOLDOWN_S=8,
        TASK_SIZE_MODE='data_gb',
        TASK_DATA_GB_MIN=0.3, TASK_DATA_GB_MAX=1.0,
        TASK_DEADLINE_MODE='scaled'),
}


def apply_scenario(name):
    """Set the module-level capacity/load globals for a named regime.
    Callable at runtime (sweep.py, ablations) as well as at import — every
    other module reads these as C.<NAME> attribute lookups, so mutating
    them here propagates everywhere without a reimport."""
    global SCENARIO, ISL_RATE_BPS, GSL_RATE_BPS, GS_RATE_BPS, LOAD_SCALE
    global TASK_IMAGES_MIN, TASK_IMAGES_MAX, MAX_TASKS, AOI_COOLDOWN_S, AOIS
    global TASK_SIZE_MODE, TASK_DATA_GB_MIN, TASK_DATA_GB_MAX
    global TASK_DEADLINE_MODE
    global MPC_HORIZON_SLOTS
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
    MAX_TASKS      = sc['MAX_TASKS']       # cap for MILP tractability
    AOI_COOLDOWN_S = sc['AOI_COOLDOWN_S']  # min gap between two same-AOI tasks
    AOIS = _BASE_AOIS + (_REALISTIC_EXTRA_AOIS
                         if name in ('realistic', 'realistic-bigdata') else [])
    # TASK_SIZE_MODE='data_gb' (only 'realistic-bigdata' sets this; every
    # other scenario falls back to 'images', today's bit-for-bit behavior)
    # samples each task's size directly as a data volume instead of an image
    # count -- see tasks.py's generate_tasks() and the 'realistic-bigdata'
    # comment above for the full rationale.
    TASK_SIZE_MODE   = sc.get('TASK_SIZE_MODE', 'images')
    TASK_DATA_GB_MIN = sc.get('TASK_DATA_GB_MIN')
    TASK_DATA_GB_MAX = sc.get('TASK_DATA_GB_MAX')
    # TASK_DEADLINE_MODE='scaled' (only 'realistic-bigdata-routing' sets
    # this; every other scenario falls back to 'fixed', today's bit-for-bit
    # behavior: deadline drawn uniformly from [TASK_DEADLINE_MIN_S,
    # TASK_DEADLINE_MAX_S] independent of task size). 'scaled' instead sets
    # each task's deadline from its OWN encode time (see tasks.py
    # generate_tasks() and the 'realistic-bigdata-routing' comment above).
    TASK_DEADLINE_MODE = sc.get('TASK_DEADLINE_MODE', 'fixed')
    # MPC planning horizon, scenario-scoped (see 'realistic-bigdata' comment
    # above for why) -- every scenario that doesn't define it falls back to
    # 60 slots/30min, today's bit-for-bit behavior.
    MPC_HORIZON_SLOTS = sc.get('MPC_HORIZON_SLOTS', 60)


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
                                 ['SCENARIO', 'TASK_IMAGES_MIN', 'TASK_IMAGES_MAX',
                                  'AOIS'])}
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

# ── Task model K ──────────────────────────────────────────────────────────────
# TASK_IMAGES_MIN/MAX/MAX_TASKS/AOI_COOLDOWN_S are all set by apply_scenario()
# above (scenario-scoped, see _SCENARIOS)
TASK_DEADLINE_MIN_S = 1_800
TASK_DEADLINE_MAX_S = 3_600
# Only used when TASK_DEADLINE_MODE='scaled' (tasks.py): a task's deadline
# becomes slack * (n_images / ENC_IMGS_PER_S), i.e. slack x its own encode
# time, floored at TASK_DEADLINE_MIN_S. 1.15 leaves real but modest room for
# network transmission/queueing after encoding -- not so generous that the
# deadline pushes past the simulation horizon for the scenario's GB range
# (see 'realistic-bigdata-routing').
TASK_DEADLINE_SLACK_FACTOR = 1.15
TASK_WEIGHTS     = [1, 2, 3]        # importance w_k, drawn uniformly
FRESHNESS_ALPHA  = 1.0 / 300.0      # phi_k decay rate after soft deadline
RNG_SEED         = 42

# ── Timeliness objective (Xuanhao's ask: delay-related terms, not just the
#    freshness multiplier) ─────────────────────────────────────────────────
DEADLINE_HARD     = False   # if True, no y[k,q,tau] created past d_k at all
MPC_LAMBDA_LATE   = 5e-2    # weight on the explicit tardiness penalty
TARDINESS_REF_S   = 300.0   # normalizes lateness to "deadline units"

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

# Reference depth for TASK_SIZE_MODE='data_gb' (tasks.py generate_tasks()):
# inverts a sampled target byte volume into n_images via PAYLOAD_B[q_ref].
# q_ref = max(DEPTHS) because PAYLOAD_B is largest there, which *minimizes*
# n_images (and therefore encode time, since ENC_IMGS_PER_S is depth-
# invariant) for any given target size -- the only one of the four depths
# that doesn't actively worsen the degenerate-encode risk documented on
# 'realistic-bigdata' above. Reads naturally as "the data volume this task
# represents at full RQ-VAE fidelity"; the scheduler is still free to choose
# a cheaper depth later, same as today.
TASK_SIZE_REF_DEPTH = max(DEPTHS)

# ── MPC scheduler ─────────────────────────────────────────────────────────────
# MPC_HORIZON_SLOTS (H) is scenario-scoped -- set inside apply_scenario()
# above, default 60 slots/30min unless a scenario (e.g. 'realistic-bigdata')
# overrides it in _SCENARIOS. Not set here directly: apply_scenario(SCENARIO)
# already ran above (module-load time), so a bare assignment here would
# silently clobber it back to a fixed value for every scenario.
MPC_RESOLVE_EVERY = 5      # re-solve at least every 5 slots (and on arrivals)

# Task-2 feasibility check: distinct from MPC_REPLAN_TIME_BUDGET_S above,
# which only bounds the inner route-refinement loop. These are compared
# against the TOTAL wall time of one _solve() call, after it finishes, to
# decide whether to commit the fresh plan or keep running the previous one.
MPC_SOLVE_BUDGET_SLOT_S  = SLOT_S                      # 30s: beat the very
                                                        # next simulated slot?
MPC_SOLVE_BUDGET_STALE_S = MPC_RESOLVE_EVERY * SLOT_S  # 150s: outer bound
                                                        # before a stale plan
                                                        # is unacceptable
# MPC_LAMBDA_QUEUE (lambda_1, early-delivery backlog weight) removed
# 2026-09-21 -- ablated per Xuanhao's Eq. 22 request, confirmed negligible
# (<=0.1% utility delta, no solve-time change, 5 seeds x 3 schedulers), see
# schedulers.py's MILP objective comment for the ablation numbers.

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


# ── Two-Level-MPC, peer coupling (twolevel.py, scheduler `Two-Level-MPC`) ────
MPC2L_ITERS        = 3      # routing-LP / depth-MILP exchange rounds
MPC2L_TOL          = 0.02   # convergence: relative L1 change in bit demand
MPC2L_ROUTE_HORIZON = 30    # routing horizon, slots (<= MPC_HORIZON_SLOTS;
                            # steps beyond it keep the static single path)
MPC2L_SHORTFALL_PENALTY = 1e3   # big-M on unrouted demand, keeps the LP feasible
MPC2L_DELAY_REF_S  = 0.05   # normalizes propagation delay in the routing cost
