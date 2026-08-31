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
python3 -m oec_sim.run_all --check-golden    # regression gate on the committed numbers

# unified four-factor utility (downstream mIoU / timeliness / concave
# coverage / cost + fairness) instead of the legacy single-factor score
python3 -m oec_sim.run_all --utility unified --oracle

# both routing/depth couplings, under the scenario where routing can matter
python3 -m oec_sim.run_all --scenario fabric-limited --utility unified \
        --couplings --congestion --oracle

python3 -m oec_sim.sweep --quick                 # congestion sweep smoke test
python3 -m oec_sim.sweep --couplings --quick     # coupling grid smoke test
```

`--utility legacy` is the **default** and reproduces every committed number
exactly; `--check-golden` enforces that against
`../oec_scenario/golden/legacy_summary.json` and exits non-zero on drift.

RL baseline needs a separate venv (Python 3.11/3.12, not this repo's 3.14)
and is meant to run on the server -- see `rl_train.py`'s docstring.

## Modules

| file | role |
|---|---|
| `config.py` | every parameter; `apply_scenario()`/`config_override()` for regime switching |
| `geometry.py` | vectorized Walker-delta propagation, elevation, ISL line-of-sight test |
| `topology.py` | E_t: LOS+range-gated +Grid ISLs, elevation-gated GSLs, static geometric Dijkstra routing |
| `routing.py` | congestion-aware predicted link costs + per-predicted-step Dijkstra fixed point (Dr. Liu's directive); `build_multi()` generates candidate path sets by edge-penalized re-Dijkstra |
| `utility.py` | **the single source of truth for utility** — one term-weighted definition shared by the realized score, the flat MPC, both hierarchical levels and the offline bound; concave-coverage linearization and the maximin fairness surrogate live here |
| `twolevel.py` | `mpc-2level`: routing MPC (multicommodity-flow LP) and depth MPC as iterated peers |
| `tasks.py` | K: AOI-triggered task arrivals; drop/reject/completion tracking |
| `schedulers.py` | MPC (MILP) + greedy baselines; O(H) encoder-pipeline constraint; timeliness objective |
| `hier.py` | hierarchical MPC (`mpc-hier`, admission/drop/backpressure) **and** `mpc-hier-route`, the slow-routing/fast-depth coupling |
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
* `routes_<sched>.csv` — per-slot path splits for the routing-MPC schedulers
* `golden/legacy_summary.json` — the committed legacy fingerprint (`--check-golden`)
* `quality_table.json` — downstream mIoU -> s_q, written by `rq-vae/downstream/harvest_metrics.py`
* `plots/*.png` — constellation map, snapshots, contact Gantt, ISL dynamics, results comparison, delay CDF, utility decomposition, coupling comparison, quality curve
* `viewer.html` — open in a browser: play/scrub the constellation, toggle ISLs/coverage, live per-GBS visibility stats

## Headline result (D = {2,4,8,16}, measured u_q = 1-LPIPS, gbs-limited)

MPC utility 64.95, within **0.4% of the HiGHS offline upper bound** (65.22);
best fixed depth flips between regimes (depth-8 at light load, depth-2/4 at
heavy load, verified across the committed sweep) while MPC tracks the upper
envelope without needing to know which regime it's in. Hierarchical MPC
trades ~6% utility for a 3.7x solve-time cut and honest drop accounting.
Full numbers, and the routing/hierarchy/oracle/RL math, in `FORMULATION.md`.

Under the **unified** utility the same comparison is far less close: MPC 44.56
vs 41.42 for the best fixed depth — a 7.6% margin where the single-factor
score showed only 1.2%. Grounding quality in a downstream task is what makes
the depth decision non-degenerate.

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
