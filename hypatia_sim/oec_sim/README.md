# oec_sim — OEC RQ-NAC Network Simulation + MPC Scheduler

Implements the system model and MPC of the Overleaf formulation
(*OEC RQ-NAC*) on a Hypatia constellation (Kuiper-630 shell 1, 1156
satellites). Extended 2026-08-20 to close Dr. Liu / Xuanhao's Aug 7-8
feedback -- see `FORMULATION.md` for the math and headline numbers.

## Run

```bash
cd hypatia_sim
pip3 install -r requirements.txt      # numpy, scipy (HiGHS MILP), matplotlib
python3 -m oec_sim.run_all            # flat mpc + greedy baselines, ~25 s
python3 -m oec_sim.run_all --hier --oracle   # + hierarchical MPC + HiGHS bound, ~1 min
python3 -m oec_sim.run_all --scenario fabric-limited --congestion  # predicted-cost routing
python3 -m oec_sim.sweep --quick      # smoke-test the committed congestion sweep
```

RL baseline needs a separate venv (Python 3.11/3.12, not this repo's 3.14)
and is meant to run on the server -- see `rl_train.py`'s docstring.

## Modules

| file | role |
|---|---|
| `config.py` | every parameter; `apply_scenario()`/`config_override()` for regime switching |
| `geometry.py` | vectorized Walker-delta propagation, elevation, ISL line-of-sight test |
| `topology.py` | E_t: LOS+range-gated +Grid ISLs, elevation-gated GSLs, static geometric Dijkstra routing |
| `routing.py` | congestion-aware predicted link costs + per-predicted-step Dijkstra fixed point (Dr. Liu's directive) |
| `tasks.py` | K: AOI-triggered task arrivals; drop/reject/completion tracking |
| `schedulers.py` | MPC (MILP) + greedy baselines; O(H) encoder-pipeline constraint; timeliness objective |
| `hier.py` | two-level hierarchical MPC (`mpc-hier`) + admission control / drop / backpressure (rotting fix) |
| `oracle.py` | offline HiGHS upper bound (LP + time-limited MILP) |
| `rl_env.py` | Gymnasium env for the PPO baseline (needs `gymnasium`) |
| `rl_train.py` | PPO training/eval (needs the RL venv; run on the server) |
| `sweep.py` | committed congestion sweep across load regimes x seeds |
| `plots.py` | figures in `../oec_scenario/plots/` |
| `viewer.py` | writes `viewer.html`, a self-contained interactive simulator |
| `FORMULATION.md` | all parameters + new objective/routing/hierarchy/oracle math, paste-ready |

## Outputs (`../oec_scenario/`)

* `summary.txt` — scheduler comparison, delay/depth-mix, solve-time, optimality-gap tables
* `contact_windows.csv`, `topology_state.csv`, `tasks.csv`
* `timeline_<sched>.csv`, `task_outcomes_<sched>.csv`, `solve_log_<sched>.csv` per scheduler
* `upper_bound.txt` — HiGHS LP/MILP bound and per-scheduler optimality gap (`--oracle`)
* `sweep/results.csv`, `plots/sweep_utility_vs_load.png` — committed congestion sweep
* `plots/*.png` — constellation map, snapshots, contact Gantt, ISL dynamics, results comparison, delay CDF
* `viewer.html` — open in a browser: play/scrub the constellation, toggle ISLs/coverage, live per-GBS visibility stats

## Headline result (D = {2,4,8,16}, measured u_q = 1-LPIPS, gbs-limited)

MPC utility 64.95, within **0.4% of the HiGHS offline upper bound** (65.22);
best fixed depth flips between regimes (depth-8 at light load, depth-2/4 at
heavy load, verified across the committed sweep) while MPC tracks the upper
envelope without needing to know which regime it's in. Hierarchical MPC
trades ~6% utility for a 3.7x solve-time cut and honest drop accounting.
Full numbers, and the routing/hierarchy/oracle/RL math, in `FORMULATION.md`.

## Cesium visualization (Hypatia SatViz)

`satviz.py` writes `../oec_scenario/satviz_oec.html` from the flat `mpc`
scheduler's output; its client-side JS Dijkstra mirrors the *static*
router only, so it will diverge from `mpc-congestion` runs -- exporting a
`routes_mpc.csv` replay for satviz to consume is on the pending list. The
offline equivalent, `viewer.html`, is unaffected (doesn't render routes).

## Switching constellations / scenarios

`CONSTELLATION_NAME` in `config.py`: `kuiper-630` (default), `starlink-550`,
`telesat-1015`, `small-walker`. `config.SCENARIO` (or `--scenario` on
`run_all.py`): `gbs-limited` (published numbers) or `fabric-limited` (makes
ISL contention, and therefore `mpc-congestion`, actually bite).

## Pending / explicitly out of scope this pass

* PPO not yet trained (env implemented + smoke-tested; needs the server + RL venv)
* `mpc-congestion` not tuned against `mpc-hier` together (routing + hierarchy composed but not jointly ablated)
* `satviz.py` route replay for predictive/hierarchical routing
* Full 6-rate x 5-seed sweep (a 3x3 subset is committed; the full grid is `python3 -m oec_sim.sweep`, ~15-20 min)
* Downstream-task utility (mIoU/F1) to replace the reconstruction-based u_q — still the biggest lever on how much depth choice can matter
