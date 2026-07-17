# oec_sim — OEC RQ-NAC Network Simulation + MPC Scheduler

Implements the system model and rolling-horizon MPC of the Overleaf
formulation (*OEC RQ-NAC*, July 2026) on a Hypatia constellation
(Kuiper-630 shell 1, 1156 satellites), replacing the earlier 6-satellite
`small_scenario.py`.

## Run

```bash
cd hypatia_sim
pip3 install numpy scipy matplotlib   # scipy provides the HiGHS MILP solver
python3 -m oec_sim.run_all            # ~1 min total (MPC ≈ 30 s)
```

## Modules

| file | role |
|---|---|
| `config.py` | every parameter, mapped 1:1 to the formulation symbols |
| `geometry.py` | vectorized Walker-delta propagation, elevation, ISL line-of-sight test |
| `topology.py` | E_t: LOS+range-gated +Grid ISLs, elevation-gated GSLs, per-slot Dijkstra routing |
| `tasks.py` | K: AOI-triggered task arrivals (N_k, w_k, d_k) |
| `schedulers.py` | MPC (MILP, eq. 8) + greedy fixed-depth / adaptive baselines |
| `plots.py` | figures in `../oec_scenario/plots/` |
| `viewer.py` | writes `viewer.html`, a self-contained interactive simulator |
| `FORMULATION.md` | all parameters in the Overleaf notation (paste-ready) |

## Outputs (`../oec_scenario/`)

* `summary.txt` — headline comparison table
* `contact_windows.csv`, `topology_state.csv`, `tasks.csv`
* `timeline_<sched>.csv`, `task_outcomes_<sched>.csv` per scheduler
* `plots/*.png` — constellation map, snapshots, contact Gantt, ISL dynamics, MPC-vs-baseline comparison
* `viewer.html` — open in a browser: play/scrub the constellation, toggle ISLs/coverage, live per-GBS visibility stats

## Headline result (D = {1,2,4,8}, measured u_q = 1−LPIPS)

Which depth is best depends on load: at light load (2 Mbps/GBS) fixed
depth-8 wins outright (utility 64.15, 100% delivered); under congestion
(1 Mbps/GBS, utilization 1.0) depth-8 collapses (87.3% delivered, utility
23.78) and depth-4 becomes the best fixed choice (62.25). The MPC matches
or beats the best fixed depth in both regimes (64.11 / 63.21, always 100%
delivered) by mixing depths per task. Full numbers in `FORMULATION.md`.

## Cesium visualization (Hypatia SatViz)

`../../hypatia/satviz/scripts/visualize_kuiper_630.py` generates the
official Hypatia 3D-globe visualization of this exact constellation
(`viz_output/kuiper_630.html`). Paste a free Cesium ion token
(https://cesium.com/ion) at line 10 of the generated HTML and open it in a
browser. `viewer.html` here is the offline equivalent.

## Switching constellations

Set `CONSTELLATION_NAME` in `config.py` to `kuiper-630` (default),
`starlink-550`, `telesat-1015`, or `small-walker` — parameters follow
Hypatia's satgenpy definitions.
