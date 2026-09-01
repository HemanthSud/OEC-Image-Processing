# RQ-VAE Satellite Image Compression for Orbital Edge Computing

Research project evaluating on-board satellite image compression using Residual Quantized Variational Autoencoders (RQ-VAE) under realistic LEO downlink constraints, with LEO network simulation via the Hypatia simulator.

| | |
|---|---|
| Lab | NICE Lab, North Carolina State University |
| Researcher | Hemanth Sudhaharan |
| Graduate Mentor | Xuanhao Luo |
| Faculty Advisor | Dr. Yuchen Liu |
| Server | `eb3-2402-grd04.csc.ncsu.edu` — 2× NVIDIA RTX A6000 |

---

## Project Goal

Evaluate how RQ-VAE compression depth affects reconstruction quality and end-to-end viability under real LEO orbital link constraints (latency, bandwidth, contact windows).

---

## Datasets

| Dataset | Size | Resolution | Use |
|---|---|---|---|
| EuroSAT RGB | 27,000 images, 10 classes | 64×64 | Initial sweep across spatial sizes and depths |
| FLAIR-1 | 77,412 aerial GeoTIFF patches | 512×512 | Main FLAIR training and evaluation |

---

## Results

### EuroSAT — Full Sweep

> Original: `64×64×3×8 = 98,304 bits` | Codebook 2048 → 11 bits/code

| Spatial | Depth | Codes | PSNR (dB) | SSIM | LPIPS | FID | Compression |
|---------|-------|-------|-----------|------|-------|-----|-------------|
| 8×8 | 1 | 64 | 29.93 | 0.7861 | 0.1665 | 19.10 | 139.6:1 |
| 8×8 | 2 | 128 | 31.30 | 0.8273 | 0.1259 | 14.24 | 69.8:1 |
| 8×8 | 3 | 192 | 32.50 | 0.8546 | 0.1037 | 13.43 | 46.5:1 |
| 8×8 | 4 | 256 | 33.28 | 0.8705 | 0.0946 | 14.43 | 34.9:1 |
| **8×8** | **8** | **512** | **36.27** | **0.9362** | **0.0513** | **9.40** | **17.5:1** |
| 4×4 | 1 | 16 | 28.41 | 0.7128 | 0.2294 | 25.46 | 558.5:1 |
| 4×4 | 4 | 64 | 27.93 | 0.7107 | 0.2573 | 34.90 | 139.6:1 |
| 4×4 | 8 | 128 | 23.34 | 0.5634 | 0.3802 | 102.80 | 69.8:1 |
| 2×2 | 1 | 4 | 24.68 | 0.5864 | 0.3477 | 67.20 | 2234.2:1 |
| 2×2 | 8 | 32 | 25.72 | 0.6497 | 0.2985 | 42.35 | 279.3:1 |

`8×8×8` is the best overall. The `4×4` family degrades with depth, likely a latent-grid mismatch at 64×64.

### FLAIR-1 — Subset Training (val split)

> Original: `512×512×3×8 = 6,291,456 bits` | 50% of FLAIR-1 (23,800 train / 7,050 val)

| Model | Payload | PSNR (dB) | SSIM | LPIPS | FID | Compression | Epochs |
|-------|---------|-----------|------|-------|-----|-------------|--------|
| 8×8×1 | 88 B | 20.63 | 0.4560 | 0.4595 | 71.33 | 8,937:1 | 120+ |
| 8×8×8 | 704 B | 21.02 | 0.4643 | 0.4779 | 73.19 | 1,117:1 | 120+ |
| 8×8×16 | 1,408 B | 21.06 | 0.4899 | 0.5039 | 125.78 | 559:1 | 150 |

PSNR and SSIM improve with depth as expected. LPIPS and FID worsen at depth-16 — depth-8 is the best overall checkpoint.

### FLAIR-1 — Depth Truncation Experiment

Tested whether the depth-16 model can be reused at shallower depths by truncating to the first k codebook stages (`evaluate_truncation.py`, `forward_partial_code`).

| Depth-16 truncated to | PSNR (dB) | SSIM | LPIPS | FID |
|---|---|---|---|---|
| 1 stage | 18.67 | 0.4793 | 0.5948 | 225.25 |
| 2 stages | 19.72 | 0.4838 | 0.5570 | 197.76 |
| 4 stages | 20.35 | 0.4831 | 0.5314 | 163.90 |
| 8 stages | 20.73 | 0.4861 | 0.5171 | 143.19 |
| 16 stages | 21.06 | 0.4899 | 0.5039 | 125.78 |

Reuse is not viable. Truncated depth-1 is 2 dB worse in PSNR and 3× worse in FID than the dedicated depth-1 model. Early stages are undertrained in deeper models because they rely on later stages to correct residuals. Dedicated models at each depth are required.

### On-Board Compute Delay (A6000 GPU, 100 runs, 512×512)

| Model | Encode time | Std |
|---|---|---|
| 8×8×1 | 12.34 ms | 0.054 ms |
| 8×8×8 | 12.34 ms | 0.029 ms |

Compute delay is identical across depths — the encoder CNN dominates, not the quantization step.

---

## Satellite Network Simulation — `hypatia_sim/oec_sim/` (current)

Implements the system model and rolling-horizon MPC of the Overleaf
formulation (*OEC RQ-NAC*). Every symbol maps 1:1 to code (`config.py`):

| Formulation | Implementation |
|---|---|
| Satellites S | Kuiper-630 shell 1 — Walker 1156/34/1 (34 planes × 34 sats, 630 km, 51.9°); switchable to `starlink-550` / `telesat-1015` |
| GBSs G | Tokyo, New York, São Paulo, Sydney (Hypatia top-100 city list); GSL feasible at elevation ≥ 20°, 2 Mbps aggregate per GBS |
| Links E_t, C_ij(t) | +Grid ISLs gated per 30 s slot by a line-of-sight test (segment must clear Earth + 80 km atmosphere) and 5,016 km max range; 100 Mbps per ISL. All 2,312 Kuiper +Grid links pass — unlike the legacy small scenario's 120°-apart links, whose chords pass ~2,900 km below the surface (the issue Xuanhao flagged) |
| Tasks K | 8 AOIs (wildfires, Amazon, Sahel, …), imaging at elevation ≥ 40°; 64 tasks / 12.8M images over the window, weights w_k ∈ {1,2,3}, soft deadlines with freshness decay |
| Depths D = {2,4,8,16} | Quality s_q — **downstream FLAIR segmentation mIoU** on the depth-q reconstruction (`--utility unified`), or the legacy reconstruction proxy u_q = 1 − LPIPS_q = 0.443 / 0.469 / 0.483 / 0.496 (`--utility legacy`); payload b_q = 88·q B (8×8 latent); encoder 12.34 ms/image |
| Utility | Unified and term-weighted (`oec_sim/utility.py`): quality × **concave** coverage × freshness, minus tardiness and resource cost, plus a maximin fairness floor. One definition shared by the realized score, the flat MPC, both hierarchical levels and the offline bound — these five used to disagree |
| Routing | Static per-slot Dijkstra; predictive congestion-weighted Dijkstra (`route_mode='predictive'`); or a genuine **routing MPC** — a multicommodity-flow LP over candidate path sets — coupled to the depth MPC either as a peer (`mpc-2level`) or hierarchically (`mpc-hier-route`) |

Window: 5 h (3.1 orbits) at 30 s slots. Run with `python3 -m oec_sim.run_all`
(~25 s; add `--hier --oracle` for the hierarchical scheduler and HiGHS bound,
~1 min). `oec_sim/FORMULATION.md` states all parameters in the Overleaf
notation.

### Unified Utility — one number that balances all the factors

Utility used to be defined in **five places that disagreed**. The *reported*
score was `Σ_k Σ_t w_k · φ_k · u_q · Δimages/N_k` — no tardiness, no cost, no
fairness. The flat MPC objective added a backlog bonus and a tardiness penalty;
the hierarchical upper level had neither; its lower level swapped the bonus for
a backpressure term; the offline bound had a third combination. Timeliness,
coverage and depth mix lived in side tables, so no single number said whether a
scheduler was actually *good*.

`oec_sim/utility.py` is now the single source of truth for all five:

```
U_k = w̄_k [ ω_Q · s_{q_k} · Ĝ_k  −  ω_T · T_k  −  ω_E · C_k ]
U   = Σ_k U_k  +  ω_F · |K| · min_k Û_k
```

| term | what it is | how it stays MILP-linear |
|---|---|---|
| `s_q` | **downstream segmentation mIoU** at depth q (was 1 − LPIPS) | table lookup |
| `Ĝ_k` | coverage gain, **concave** in delivered fraction, each image weighted by the freshness at which it arrived | piecewise-linear segments with decreasing slopes; maximizing a concave separable function needs **no binaries and no SOS2** — the LP fills segment 1 before segment 2 on its own |
| `T_k` | tardiness, clipped at one deadline-unit | the clip is on a *coefficient* (arrival times are data) |
| `C_k` | resource cost: payload-proportional + per-image encode | constant per (k,q) |
| `min_k Û_k` | fairness floor, weight-relative | one column + \|K\| rows; **Jain is reported but not optimized** — it is a ratio of quadratics, so optimizing it would leave the offline bound with nothing valid to compare against |

**`--utility legacy` is the default and reproduces every committed number
exactly** — it is a parameter setting (ω_T = ω_E = ω_F = 0, one unit coverage
segment, s_q = 1 − LPIPS), not a code branch. `run_all --check-golden` enforces
this against `oec_scenario/golden/legacy_summary.json` and exits non-zero on
drift.

**What the unified score changes.** The point of grounding quality in a
downstream task is that depth choice stops being degenerate:

| | best fixed depth | mpc | mpc's margin | gap to HiGHS bound |
|---|---|---|---|---|
| legacy utility | 64.15 (`fixed-8`) | **64.95** | +1.2% | 0.4% |
| unified utility | 13.71 (`fixed-8`) | **18.13** | **+32.2%** | 3.7% |

Under the old single-factor score the MPC was barely distinguishable from
picking depth 8 and never thinking again. Under the unified score — now
measured with real downstream mIoU, not a placeholder — it wins by **~26×**
the legacy margin, because coverage is now concave, lateness is charged for,
and the quality term actually varies steeply with depth.

Concretely, what "concave coverage" buys: a task's **first 50% of images earns
79.3% of its full value** (it is exactly 50% under the legacy linear score), so
covering many AOIs partially beats saturating one — which is the behaviour you
actually want from an imaging constellation, and which the old score had no way
to express.

Two findings fell out of building it:

**Encode energy dominates downlink by ~230×, and encode is depth-independent.**
`t_enc` = 12.34 ms was measured at *both* 8×8×1 and 8×8×8 (σ ≈ 0.03–0.05 ms).
Per image that is 0.370 J of encode against 2.25 × 10⁻³ J of downlink at q=16;
over a full run, 4,522 kJ vs 19.65 kJ. So a literal Joule-denominated term is
nearly constant in q and **cannot** drive depth choice — depth selection here
is a *bandwidth* decision, not an energy one. The objective therefore uses a
normalized cost whose shape is physical and whose scale is a stated policy
weight (ω_E = 0.05, a tie-breaker); absolute Joules are reported as accounting
only. The 30 W / 20 W power figures are **assumed** (Jetson AGX Orin mid-band;
the A6000 the timing came from is not a flight part) and labelled as such.

**The fairness floor buys +8.5% for the worst-served task at a cost of 1.4% of
aggregate utility**, and saturates by ω_F = 0.1 — which is why that is the
default. It also exposes something the old score hid completely:
`greedy-fixed-16` has `u_min = 0.0000` and Jain 0.4272, i.e. it wins its (poor)
utility partly by *starving tasks outright*. Nothing in the single-factor
number showed that.

The bound carries every new term, relaxed optimistically, and `oracle.report`
now **asserts** `bound ≥ realized` per scheduler. That check earned its keep
immediately during development — it caught the offline bound double-counting
already-realized utility in its fairness constant, which had pushed the LP
bound above the analytic ceiling. Fixed; with the real quality table the
ordering is ceiling 23.34 > LP 18.93 > MILP dual 18.83 ≥ realized 18.13 (the
MILP's *incumbent* search didn't converge to a good integer solution in the
120s budget against these steeper real coefficients — the dual bound is still
valid and is what the gap column uses; a tighter primal incumbent is a
follow-up, not a blocker).

> ✅ Unified-mode numbers on this page are **measured**, not provisional —
> `oec_sim/quality_table.json` from the completed FLAIR segmentation sweep on
> the NCSU server (`flair-unet-r34-rgbie / val7050`, full results in
> "Downstream Task Utility" below). The legacy numbers were always measured
> and remain unaffected.

### MPC Scheduler vs. Fixed Depths vs. Hierarchical MPC

> The table below is **legacy utility mode** (`--utility legacy`, the default)
> and is exactly reproducible; those values are *not* comparable to the
> unified-utility numbers above, which are on a different scale.

The flat scheduler is a deterministic rolling-horizon MPC: HiGHS MILP (via
scipy) over an H = 60-slot (30 min) horizon, executes the first slot,
re-plans on task arrivals and at least every 5 slots — with a timeliness
objective (route-delay-aware freshness + an explicit tardiness penalty, not
just the freshness discount) and an O(H) encoder-pipeline constraint (was
O(H²), a 3-10× MILP speedup). Baselines: greedy with each fixed depth, a
greedy that adapts depth to queue length, and a two-level hierarchical MPC
(`mpc-hier`) that separates slow admission/drop/budget decisions from a fast
per-task depth MILP — see `oec_sim/FORMULATION.md` for the full math.

**Which fixed depth is best depends on load; MPC tracks the upper envelope
without knowing the regime in advance — and is within 0.4% of the offline
HiGHS optimum:**

| scheduler | utility (D={2,4,8,16}, GBS-limited, util≈1.00) | gap to HiGHS bound |
|---|---|---|
| mpc | **64.95**, 95.4% delivered | 0.4% |
| greedy-fixed-8 | 64.15, 100% delivered | 1.6% |
| greedy-adaptive | 64.12, 100% delivered | 1.7% |
| greedy-fixed-4 | 62.25, 100% delivered | 4.5% |
| mpc-hier | 61.4, 95.1% delivered (16 tasks explicitly dropped, not silently starved) | 5.9%, but 4.3× faster to solve |
| greedy-fixed-2 | 58.85, 100% delivered | 9.8% |
| greedy-fixed-16 | 24.43, 87.3% delivered | 62.5% |

Outputs land in `hypatia_sim/oec_scenario/`: per-task outcomes, per-slot
timelines, per-scheduler MILP solve-time logs, the HiGHS upper-bound report
(`--oracle`), `summary.txt`, and `plots/*.png`.

**Delay and violation metrics** (new this pass, in every `task_outcomes_*.csv`
and in `summary.txt`'s "Delay / depth mix" block): per-task completion delay
and lateness, `on-time%` (image-weighted — what fraction of *images* beat
their deadline) and `viol%` (task-weighted — what fraction of *tasks* had
*any* late images, a stricter, complementary reading of the same run), mean
and p95 completion delay, the chosen-depth distribution per scheduler, and
`dropped`/`rejected` counts for schedulers that support giving up on a task.
`delay_cdf.png` plots the completion-delay distribution across schedulers.

| scheduler | on-time% (image-weighted) | viol% (task-weighted) | mean delay | p95 delay | dropped |
|---|---|---|---|---|---|
| mpc | 88.7% | 43.8% | 2,216 s | 3,189 s | 0 |
| greedy-adaptive | 88.7% | 43.8% | 2,463 s | 3,536 s | 0 |
| greedy-fixed-2 | 88.7% | 43.8% | 2,453 s | 3,536 s | 0 |
| greedy-fixed-4 | 88.7% | 43.8% | 2,453 s | 3,536 s | 0 |
| greedy-fixed-8 | 88.7% | 43.8% | 2,460 s | 3,536 s | 0 |
| greedy-fixed-16 | 31.3% | 85.9% | 4,275 s | 7,836 s | 0 |
| mpc-hier | 84.3% | 45.3% | 2,678 s | 3,388 s | 16 |

`greedy-fixed-16` is the clearest illustration of what these metrics are for:
its per-image on-time rate collapses to 31.3% and p95 delay balloons to
7,836 s (~2.2 h) — those are the numbers utility alone doesn't show directly.
`mpc-hier`'s 16 dropped tasks are exactly the "rotting" ones it gave up on
rather than letting them silently miss every deadline forever.

These measure timeliness, not whether the delivered image was still *useful*
for its task. That gap is what the unified utility above closes: timeliness is
now folded into the single number rather than living only in this side table,
and the quality term is grounded in downstream segmentation rather than pixel
fidelity.

### Downstream Task Utility — segmentation on reconstructions

The pipeline that replaces `u_q = 1 − LPIPS_q` with real downstream
performance lives in `rq-vae/downstream/`. It runs **server-side** (needs the
RQ-VAE checkpoint, the FLAIR GeoTIFFs and CUDA) and has now completed a full
run against the frozen 7,050-image val population.

**Hybrid 5-band design.** RQ-VAE compresses RGB only, while FLAIR's baseline
segmenter takes RGB + NIR + Elevation. Each output raster is therefore

```
bands 1-3 = the depth-q RECONSTRUCTION
bands 4-5 = the ORIGINAL NIR and Elevation, copied through untouched
```

which keeps IGNF's published 5-band U-Net/ResNet34 checkpoint usable **with no
segmentation training**, and is physically honest — only the optical bands went
through the codec.

**Anchoring is the whole ballgame.** Two conditions are run beyond
q ∈ {1,2,4,8,16}:

| condition | what it gives |
|---|---|
| `--depth orig` | `mIoU_ref` — the uncompressed reference. Also the **checkpoint gate**: run first, before anything costly, to catch a wrong class-weight vector / model provider / normalization before it silently corrupts every later number |
| `--depth blank` | `mIoU_floor` — RGB bands filled with the dataset means, so the segmenter runs on **NIR + Elevation alone** |

`s_q = (mIoU_q − mIoU_floor) / (mIoU_ref − mIoU_floor)`. The decision-theoretic
zero for a scheduler is not "mIoU = 0" but *what you get by not delivering the
image at all* — which is exactly the blanked condition. Anchoring there instead
of at zero is the difference between a table that barely varies and one where
depth choice matters. Both anchorings are written to `quality_table.json`, so
the choice stays visible and reversible. **If the spread is still narrow after
floor anchoring, that is the finding** — to be reported, not tuned away.

```bash
# on the server, in tmux
tmux new -s flair_sweep
MAX_SAMPLES=500 ./downstream/run_depth_sweep.sh   # subset sanity check first
./downstream/run_depth_sweep.sh                   # full 7,050
```

One condition at a time, deleting reconstructions after harvest, so peak disk
stays low rather than needing all six conditions' reconstructions at once
(`recon_to_geotiff.py` also refuses to start below `--min-free-gb`; this box
has hit 100% before). `metrics.py` is the bottleneck — a serial
`confusion_matrix` over 262k pixels × 7,050 patches per condition.

**Results, full 7,050-image val population:**

| condition | mIoU | s_q (floor-anchored) |
|---|---|---|
| orig (`mIoU_ref`) | 68.87% | — |
| blank (`mIoU_floor`) | 6.08% | — |
| q1 | 12.60% | 0.104 |
| q2 | 17.53% | 0.182 |
| q4 | 22.54% | 0.262 |
| q8 | 25.68% | 0.312 |
| q16 | 28.70% | 0.360 |

Floor-anchored spread from q1 to q16 is **~247%**, against **~12%** for the
1−LPIPS proxy it replaces — pixel fidelity was massively understating how
much reconstruction depth actually matters for a downstream task. That's the
core validation of grounding the utility function this way.

One honest wrinkle: the checkpoint's own self-reported mIoU (58.6% for the
15-class RGB+IR+Elevation ResNet34-UNet — see the model card at
`IGNF/FLAIR-INC_rgbie_15cl_resnet34-unet` on HuggingFace) doesn't match
`mIoU_ref` here. Root cause, best understanding: FLAIR-1-main ships two
different held-out sets — the *val* split used here (a held-out-domain slice
of the *train* data release) and a separate official *test* split (a
different data release entirely). IGNF's published number was almost
certainly benchmarked against the test split, not val; a domain-leakage check
between our val population and the training domains came back clean, ruling
out the more concerning explanation. Per-class IoU on our val population is
structurally sane throughout (easy common classes like building/water score
high, rare/small classes like swimming_pool score lower) — the signature of
a correctly-loaded, working model, just evaluated against an easier
population than IGNF's headline number. Verifying against the true test split
would need unzipping a 14 GB test-image archive plus an unfetched test-label
archive and predicting on 15,700 images — judged not worth it, since what the
OEC utility needs is a self-consistent `mIoU_q` on one fixed population
across depths, not an exact match to an external benchmark number.

### Offline Optimality Bound (HiGHS)

`oec_sim/oracle.py` computes two valid upper bounds — dropping ISL/GSL rows
and keeping only the per-window GBS-aggregate budget, which is always a
*relaxation* (it never binds tighter than the true feasible region, since
those rows never bind in the real schedule either — see below):

| bound | value | method |
|---|---|---|
| analytic ceiling | 69.95 | every image at best depth, delivered instantly (sanity check only) |
| LP bound | 65.66 | depth relaxed to continuous, 5-slot windows |
| MILP bound (dual) | 65.22 | full resolution, integral depth, HiGHS, 120 s time limit, 0.1% MIP gap |

**MPC lands at 64.95 — 0.4% off the true optimum.** That's the single
strongest number from this pass: the flat MPC isn't "pretty good," it's
essentially solving the problem optimally in this regime.

### Two-MPC Split: routing MPC + depth MPC, built both ways

*"Hierarchical MPC — one MPC to select the best path, another one [for the
rest]."* `hier.py` was **named** hierarchical but split admission/budget from
depth and routed statically at *both* levels, so the routing/depth split Dr.
Liu asked for did not exist. It now exists in two forms, and both are reported.

| | `mpc-2level` (peer) | `mpc-hier-route` (hierarchical) |
|---|---|---|
| routing solver | multicommodity-flow **LP** over per-(task, path, τ) flows | same LP, in bits, over macro-windows |
| cadence | every re-plan, iterated with the depth MILP to a damped fixed point | once per 10-min macro-epoch, path set then **frozen** |
| coupling | demand ↔ mix, MSA damping, keep-best-iterate | `Directive.routes` handed down, like the existing budget directive |
| how multipath reaches the depth MILP | capacity coefficients become `b_q·8·θ_e` — fractional edge **shares**, so no path variables are added to the depth problem | same |

Both reduce exactly to their baselines by construction, and that is tested:
`MPC2L_ITERS=1, MPC_ROUTE_NPATHS=1` reproduces `mpc` bit-for-bit, and
`HIER_ROUTE_ON=False` reproduces `mpc-hier` bit-for-bit.

**Results** (`fabric-limited`, unified utility, ISL 1 Mbps — the regime where
29.6% of links are oversubscribed and routing can actually matter):

| scheduler | utility | gap to bound (46.06) | solve time | delivery% | on-time% | paths/(k,t) |
|---|---|---|---|---|---|---|
| `mpc-congestion` | **45.68** | 0.8% | 54.9 s | 90.8% | 87.6% | 1 |
| `mpc-2level` | **45.66** | 0.9% | 52.5 s | **92.0%** | **88.7%** | **1.51** |
| `mpc` | 45.45 | 1.3% | 21.4 s | 91.3% | 88.1% | 1 |
| `mpc-hier` | 40.42 | 12.2% | **6.3 s** | 82.9% | 76.8% | 1 |
| `mpc-hier-route` | 40.41 | 12.3% | **5.5 s** | 84.5% | 77.1% | frozen route usable at execution only **7.6%** of the time |

**The pick: `mpc-2level`** — and the pre-registered rule made that call, not
hindsight. The rule, written before any coupling was run, was: *recommend the
cheaper `mpc-hier-route` unless `mpc-2level` beats it by more than 1.0 utility
point (≈1.5%) or 1.0 point of bound gap, in at least 4 of the load regimes.*
The default favoured the cheap coupling deliberately — `mpc-hier-route` solves
~10× faster, and routes across a 1,156-satellite fabric are not realistically
re-planned every 30 s with a ground solver in the loop.

It loses anyway, and not narrowly — in **5 of 5 regimes**, by 4.4 to 5.6
points. Committed sweep, `oec_scenario/sweep/results_routing.csv`, 5 ISL rates
× 3 seeds, mean over seeds (the rule was drafted for a 6-point grid and is
applied as ≥4 of 5):

| ISL Mbps | `mpc` | `mpc-2level` | `mpc-hier` | `mpc-hier-route` | 2level − hier-route |
|---|---|---|---|---|---|
| 0.50 | 40.88 | **41.23** | 36.03 | 36.18 | **+5.04** |
| 0.75 | 41.87 | **42.13** | 37.10 | 36.93 | **+5.19** |
| 1.00 | 43.71 | **43.92** | 37.96 | 38.37 | **+5.55** |
| 1.50 | 44.00 | 44.01 | 38.56 | 39.64 | **+4.37** |
| 2.00 | 44.03 | 44.03 | 39.59 | 38.90 | **+5.13** |

**But that margin is not about routing at all**, and the sweep is what makes
that clear. Broken out against each coupling's own baseline:

| ISL Mbps | `mpc-2level` − `mpc` | `mpc-hier-route` − `mpc-hier` |
|---|---|---|
| 0.50 | +0.344 | +0.156 |
| 0.75 | +0.258 | −0.171 |
| 1.00 | +0.211 | +0.405 |
| 1.50 | +0.003 | +1.082 |
| 2.00 | +0.000 | −0.694 |

Two things follow, and both are worth more than the headline:

1. **The routing MPC buys at most +0.84%**, only where the fabric is genuinely
   scarce, and **exactly nothing** once ISL ≥ 1.5 Mbps — where it converges to
   the flat MPC because there is no contention left to route around. It costs
   37–61 s of solve time against `mpc`'s ~22 s to do it. A routing *optimizer*
   is simply not worth much here over a good routing *heuristic*:
   `mpc-congestion`, plain congestion-weighted Dijkstra, matches it (45.68 vs
   45.66 at the operating point).
2. **Route freezing contributes nothing**: `mpc-hier-route` − `mpc-hier` swings
   both signs and averages ≈ +0.16, i.e. noise — exactly what a frozen route
   that is usable under 8% of the time predicts.

So the honest reading of the 5-point gap is that it is **admission and drop**,
not routing: the hierarchical family explicitly gives up 36 tasks, and that,
not its path choice, is what costs it. `mpc-2level` is the right pick under the
rule, but the result that actually matters for the paper is that **routing is
not where the utility is in this system** — depth selection and admission are.

**Why the hierarchical coupling loses is the more interesting half.** It isn't
tuning. A frozen path is pinned to *specific satellites*, and GSL contact
windows here average 227–265 s, so the satellite serving a given GBS turns
over faster than any useful planning epoch. Measured survival of a route
frozen at epoch start:

| lookahead | 0 s | 30 s | 150 s | 300 s | 570 s |
|---|---|---|---|---|---|
| still feasible | **14.4%** | 11.7% | 12.2% | 3.3% | 0.0% |

(that breakdown is from a 250-slot diagnostic; over the full 601-slot run the
executed-slot figure is **7.6%**, and `summary.txt` reports it alongside the
across-the-horizon rate, 4.9%, since the two answer different questions.)

Even at the *executed* slot, with zero lookahead, the frozen route is usable
under 15% of the time, and shortening the epoch to 2 minutes does not help
(90.6% fallback vs 92.8%). **Freezing concrete paths does not survive a LEO
fabric.** The honest
caveat: this is a negative result about freezing *concrete paths*. Freezing a
more abstract decision — a serving-GBS assignment, or a route class — might
survive, and that is now on the Pending list rather than claimed here.

**Two things worth stating plainly about the winner, too.** First, the routing
MPC beats the flat MPC by only **+0.46%** (45.66 vs 45.45) even in a
deliberately fabric-limited regime — and `mpc-congestion`, the far simpler
predicted-cost Dijkstra, matches it (45.68) at the same cost. The routing
*optimizer* is not buying much over a well-chosen routing *heuristic*. Second,
the ~5-point gap between the flat/peer family (~45.5) and the hierarchical
family (~40.4) is about **admission and drop**, not routing at all —
`mpc-hier` gives up 36 tasks explicitly.

Two implementation traps produced convincing-looking null results before being
caught, both recorded in `FORMULATION.md` because they generalize:

1. **Circular demand.** Deriving the routing LP's demand from the depth MILP's
   plan is circular — that plan was already made feasible against the static
   paths, so the LP sees no contention and re-derives Dijkstra. `mpc-2level`
   came out bit-for-bit equal to `mpc` on a fabric with 30% of links
   oversubscribed. The demand must be what a task *wants* to send.
2. **A big-M that wasn't.** The hierarchical LP's shortfall penalty was divided
   by 1e9 while the flow cost was not, making *not routing* ~5×10⁵ times
   cheaper than routing; the LP shorted everything, returned zero routes, and
   `mpc-hier-route` silently degenerated into `mpc-hier`.

Both look identical from the outside — "no difference between couplings" —
which is why `summary.txt` now reports paths-per-decision and frozen-route
survival, not just utility.

### Predicted-Cost Routing (Dr. Liu's directive)

*"MPC and Dijkstra are not necessarily mutually exclusive — have MPC
predict link costs, and Dijkstra compute the path at each predicted step."*
Implemented in `oec_sim/routing.py`: an M/M/1-style congestion penalty on
top of propagation distance, built from the MPC's own predicted load, with
Dijkstra re-solved per horizon step and iterated to a damped fixed point
(`route_mode='predictive'`, scheduler name `mpc-congestion`).

**Verified no-op under the committed `gbs-limited` rates** — an audit found
the per-GBS aggregate budget is ≥12× tighter than any ISL segment a route
could cross, so ISL contention is mathematically unreachable (max measured
ISL utilization ≈0.08) and predicted-cost routing has nothing to act on
(`route_mode='predictive'` reproduces `route_mode='static'` bit-for-bit at
iteration 0). A second scenario, `SCENARIO='fabric-limited'` (`--scenario
fabric-limited --congestion`), rebalances rates so ISL segments can
genuinely saturate; there, predictive routing measurably changes the plan
and utility. Only single-path routing is implemented — multipath candidate
sets are documented future work.

### Hierarchical MPC + the Rotting-Backlog Fix

*"Hierarchical MPC — one MPC to select the best path, another one [for the
rest]."* `oec_sim/hier.py` splits the problem into a slow upper level
(admission control, explicit task abandonment, per-task GBS-budget
allocation, every 10 min) and a fast lower level (today's per-task depth
MILP, but short-horizon and budget-capped, every 5 slots). This also gives
the rotting-backlog problem — a task whose freshness has decayed to ~0
still holds backlog and still generates MILP variables every re-plan, but
is never worth serving, so under the flat scheduler it silently starves
forever — an explicit, reported fix: tasks are either admitted, dropped
(counted), or rejected at arrival (counted), never silently abandoned.

Measured trade-off (see table above): `mpc-hier` gives up ~3.6 utility
points relative to flat MPC (5.9% vs 0.4% gap to the HiGHS bound) in
exchange for **4.3× faster solving** (3.7 s vs 15.8 s total solve time
across the run) and honest accounting of the 16 tasks it explicitly drops.
A small grid search (5 combinations) tuned `PHI_DROP` (0.05→0.01) and
`HIER_THETA_ADMIT` (0.9→0.5), closing about a point of the gap (60.71→61.4
utility) — a first pass, not exhaustive.

### Committed Congestion Sweep

![Utility vs. GBS downlink rate: MPC and the hierarchical MPC track the upper envelope across load regimes, while the best fixed depth crosses over between them](hypatia_sim/oec_scenario/plots/sweep_utility_vs_load.png)

> Re-run this pass with the full column set (`utility_quality`, `u_min`,
> `jain`, `n_route_fallbacks`, `solve_wall_s`) and with `mpc-hier` included.
> **All 180 previously-committed rows came back bit-identical** on utility,
> images delivered, delivery fraction and violation fraction — a regression
> check across the whole grid, not just the single `--check-golden` scenario.

`oec_sim/sweep.py` replaces what used to be a single manual congested run
that was never saved to the repo. Full grid — 6 GBS rates × 5 seeds × 6
schedulers (+ `mpc-hier`), **210 rows**, committed at
`oec_scenario/sweep/results.csv`:

| GBS rate | best fixed depth | best-fixed utility | mpc utility |
|---|---|---|---|
| 0.75 Mbps | depth-2 | 57.08 | **60.34** |
| 1.0 Mbps | depth-4 | 60.38 | **61.18** |
| 1.5 Mbps | depth-4 | 60.38 | **62.34** |
| 2.0 Mbps | depth-8 | 62.22 | **62.96** |
| 3.0 Mbps | depth-8 | 62.22 | **63.61** |
| 4.0 Mbps | depth-16 | 63.92 | 63.76 |

The best *fixed* depth climbs 2→4→8→16 as load falls; `mpc` beats it in 5
of 6 regimes and is statistically tied (within 0.2 utility) at the one
point where there's no scarcity to exploit. No cherry-picking — this is the
mean over all 5 seeds per rate.

### RL Baseline (PPO) — Trained and Evaluated

> These numbers are **legacy utility mode**. The trained `ppo_oec.zip` optimized
> the old single-factor reward, so it is stale under `--utility unified`;
> retraining is on the Pending list.
>
> Two comparison-fairness bugs in the RL path were fixed this pass, both of
> which quietly favoured the RL baseline over the schedulers it is measured
> against. (1) `rl_env._execute_one` recorded deliveries with **no propagation
> delay**, while the MPC and greedy paths both charge the real route delay — so
> RL deliveries were credited with better freshness than they earned. The
> effect turns out to be tiny (16.16335 → 16.16308, 0.0017%), because ~50-75 ms
> of propagation is negligible against a 30 s slot and a 300 s freshness
> constant — a correctness fix, not a result. (2) `RLScheduler.run()` built its
> own history dict and summed `delivered_utility` directly, bypassing
> `utility.run_utility`, so under `--utility unified` it would have omitted the
> fairness term that every other scheduler's total includes. Both now match the
> rest of the fleet.

Dr. Liu asked for an MPC-vs-RL comparison. `oec_sim/rl_env.py` implements a
Gymnasium environment (`Discrete(6)`: commit one of 4 depths, defer, or
drop, decided per active task rather than per slot so the action space is
independent of how many tasks are active or how large the constellation
is). Reward matches the MILP objective plus potential-based backlog
shaping.

**A real bug turned up during this pass and is worth stating plainly**: the
environment had a genuine infinite loop — a task the policy deferred would
get re-offered at the *same simulated time slot* forever instead of moving
on to the next slot, because the "rebuild this slot's queue" check used
empty-list truthiness, which can't tell "just drained, waiting to advance"
apart from "not built yet." A diagnostic run caught it directly: 3,000
consecutive decisions with the simulation clock frozen at slot 420 of 601.
Fixed with an explicit `None` sentinel (`rl_env.py::_advance_to_next_decision`);
a full episode now runs in ~0.2 s instead of hanging indefinitely.

With that fixed, PPO was **trained** (300k timesteps, load regimes 1.5–2.5
Mbps × 10 seeds, ~4 min on a CPU Mac) and **evaluated** in-distribution and
out-of-distribution:

| regime | RL utility | RL delivered% | RL µs/decision | MPC utility | MPC µs/decision |
|---|---|---|---|---|---|
| in-dist (2.0 Mbps) | 58.13 / 52.31 | 100% | ~500–1,400 | 61.58 / 55.55 | ~145,000–243,000 |
| OOD-light (4.0 Mbps) | 58.13 / 52.31 | 100% | ~500–1,400 | 62.44 / 56.14 | ~145,000–243,000 |
| OOD-heavy (0.75 Mbps) | 54.76 / 50.83 | 100% | ~500–700 | 58.92 / 53.27 | ~163,000–173,000 |
| *(two seeds shown per regime: 100 / 101)* | | | | | |

Three findings — two matched what I predicted in advance, one didn't:

1. **PPO loses to MPC on utility, as expected** — 5–8% below MPC everywhere tested.
2. **PPO is ~250–400× faster per decision** (µs vs. hundreds of ms) — the inference-cost story holds up strongly.
3. **Unexpected**: the observation vector has no capacity/congestion feature (task properties + active-count + mean freshness only), so the policy's *actions* are provably rate-independent — utility is bit-for-bit identical between 2.0 and 4.0 Mbps at a fixed seed. It still delivers 100% of images at 0.75 Mbps (where `greedy-fixed-8` collapses to 13–20 utility) purely because it happened to learn conservative depth choices, not because it senses congestion. That's accidental robustness, not adaptive — the clear next step is adding residual-bandwidth features to the observation, which the original design called for but the shipped version dropped for a smaller vector.

### Visualizations

![OEC scenario on the Cesium globe: Kuiper-630 constellation with LOS-gated ISLs and active MPC task routes (dark red, q=8) at t = 2:38](hypatia_sim/oec_scenario/plots/satviz_oec_screenshot.jpg)

*`satviz_oec.html` at t = 2:38 — 13 active tasks routed to their GBSs (dark
red = MPC chose depth 8), 2,312 feasible ISLs, with the legend, scenario/model
panel, and live MPC stats in the HUD. Screenshot predates this pass's switch
to D={2,4,8,16} (taken under the older {1,2,4,8} depth set); regenerate with
`python3 -m oec_sim.satviz` after a fresh `run_all` to match current depths
(yellow q=2, orange q=4, red q=8, purple q=16).*

Three viewers, all showing the same verified topology (the drawn ISLs, GSLs,
and routes were checked node-for-node against `topology.py`'s routing):

> **Cesium ion tokens are no longer embedded.** `satviz_oec.html` is a tracked
> file, so baking a token into it commits a credential. `satviz.py` now emits an
> empty token by default; supply one at generation time with
> `CESIUM_ION_TOKEN=<token> python3 -m oec_sim.satviz`, or
> `--embed-local-token` for a copy you will not commit. The command prints
> which of the two happened.

1. **`oec_scenario/satviz_oec.html`** *(main)* — the OEC scenario on the
   Cesium 3D globe in Hypatia's SatViz style. Constellation animated over the
   full 5 h window with a timeline scrubber; per-frame LOS-gated ISLs,
   elevation-gated GSLs, GBS/AOI markers; active MPC tasks drawn as
   shortest-delay routes coloured by the chosen depth (green q=1, yellow q=2,
   orange q=4, dark red q=8); legend, scenario/model panel, and live stats
   HUD (delivered images, utility, backlog) driven by `timeline_mpc.csv`.
   Landmass is embedded (Natural Earth 110m, `oec_sim/land_110m.json`) rather
   than streamed — tile hosts block `file://` pages — so only the Cesium
   library loads from the network. Regenerate: `python3 -m oec_sim.satviz`.
2. **`oec_scenario/viewer.html`** — self-contained 2D canvas simulator
   (play/scrub, ISL/coverage toggles, live stats); fully offline.
3. **Stock Hypatia SatViz** (`hypatia/satviz`, local clone, not tracked) —
   `scripts/visualize_kuiper_630.py` writes `viz_output/kuiper_630.html`,
   the same static snapshot style as Hypatia's README figures (one epoch,
   intra-plane orbit rings only, no ISL model). Needs a free Cesium ion
   token pasted at line 10.

### Small Scenario (legacy, superseded)

`hypatia_sim/small_scenario.py` — an earlier 6-satellite debug scenario,
fully replaced by `oec_sim/`. Its 120°-separation intra-plane ISLs are not
physically feasible (no line of sight — the direct path passes through the
Earth), which is exactly the issue `oec_sim`'s LOS+range gating was built
to fix. Kept in the repo for reference only; not part of the active
results.

---

## Repository Structure

```
.
├── hypatia_sim/
│   ├── oec_sim/                    # OEC scenario + MPC scheduler (current)
│   │   ├── config.py               # All parameters (constellation, links, tasks, depths)
│   │   ├── geometry.py             # Walker propagation, ISL line-of-sight test
│   │   ├── topology.py             # E_t: feasibility-gated links, static geometric routing
│   │   ├── routing.py              # MPC-predicted link costs + per-tau Dijkstra fixed point
│   │   ├── tasks.py                # AOI-triggered task arrivals, drop/reject tracking
│   │   ├── utility.py              # THE unified utility: one definition shared by the score,
│   │   │                           #   both MPCs, the hierarchy and the offline bound
│   │   ├── schedulers.py           # MPC (MILP) + greedy baselines; O(H) constraint; timeliness objective
│   │   ├── twolevel.py             # mpc-2level: routing MPC + depth MPC as iterated peers
│   │   ├── hier.py                 # mpc-hier (admission/drop) + mpc-hier-route (slow route/fast depth)
│   │   ├── oracle.py               # Offline HiGHS upper bound (LP + time-limited MILP)
│   │   ├── sweep.py                # Committed congestion sweep across load regimes x seeds
│   │   ├── rl_env.py / rl_train.py # Gymnasium env + PPO baseline (needs a separate RL venv)
│   │   ├── plots.py / viewer.py    # Figures + interactive HTML simulator
│   │   ├── satviz.py               # OEC scenario on the Cesium globe (Hypatia SatViz style)
│   │   ├── land_110m.json          # Embedded Natural Earth landmass for satviz
│   │   └── FORMULATION.md          # Parameters in the Overleaf notation
│   ├── oec_scenario/               # Simulation outputs (CSVs, plots, viewer.html, satviz_oec.html)
│   │   └── golden/                 # Committed legacy fingerprint (run_all --check-golden)
│   ├── ppo_oec.zip                 # Trained PPO baseline (MaskablePPO, 300k timesteps)
│   ├── requirements.txt            # Core sim deps (numpy, scipy, matplotlib)
│   ├── requirements-rl.txt         # + RL training deps (gymnasium, sb3-contrib, torch)
│   ├── small_scenario.py           # Legacy 6-sat scenario (infeasible ISLs)
│   ├── topology_config.py          # Full Starlink-550 topology config
│   ├── generate_sim_inputs.py      # UDP burst schedule + ns-3 config
│   ├── extract_topology.py         # ISL degree, serving sats, handoffs, windows
│   ├── analyse_results.py          # Latency, drop rate, ISL utilization
│   ├── profile_compute_delay.py    # On-board encode time profiling
│   ├── TOPOLOGY.md                 # Topology design doc
│   └── small_scenario/             # Simulation outputs
├── rq-vae/
│   ├── downstream/                 # Segmentation-on-reconstruction pipeline (server-side)
│   │   ├── recon_to_geotiff.py     #   depth-q recon RGB + original NIR/Elev -> 5-band GeoTIFF
│   │   ├── make_flair_csv.py       #   recon tree -> the 2-column CSV FLAIR expects
│   │   ├── harvest_metrics.py      #   metrics.json -> oec_sim/quality_table.json (s_q)
│   │   ├── run_depth_sweep.sh      #   tmux driver, one condition at a time (disk-safe)
│   │   └── configs/                #   FLAIR predict+metrics template (no training)
│   ├── evaluate_metrics.py
│   ├── evaluate_truncation.py      # Depth-16 codebook reuse experiment
│   ├── run_flair_8x8_sweep.sh
│   ├── train_eurosat.py
│   └── rqvae/
├── nac/
│   ├── arithmetic_coding.py
│   ├── nac_eurosat.py
│   └── ngram.py
├── results/
└── flair_val_metrics_all.txt
```

---

## Reproducing

### Dependencies

```bash
pip install torch torchvision torchaudio
pip install omegaconf einops lpips tensorboard scikit-image tqdm matplotlib pillow numpy scipy pyyaml
pip install "rasterio<1.5"
```

### FLAIR-1 Training

```bash
cd rq-vae
CUDA_VISIBLE_DEVICES=0 DEPTHS="1 2 4" MAX_TRAIN_SAMPLES=23800 ./run_flair_8x8_sweep.sh
CUDA_VISIBLE_DEVICES=1 DEPTHS="8 16" MAX_TRAIN_SAMPLES=23800 ./run_flair_8x8_sweep.sh
```

### Evaluate Metrics

```bash
python3 evaluate_metrics.py --split val --output-dirs output/flair-rqvae-8x8x1 output/flair-rqvae-8x8x8 output/flair-rqvae-8x8x16
python3 evaluate_truncation.py
```

### Satellite Simulation

```bash
cd hypatia_sim
pip3 install -r requirements.txt      # numpy, scipy (HiGHS MILP), matplotlib
python3 -m oec_sim.run_all                    # ~25 s; outputs in oec_scenario/
python3 -m oec_sim.run_all --hier --oracle    # + hierarchical MPC + HiGHS bound, ~1 min
python3 -m oec_sim.sweep --quick              # committed congestion sweep, smoke test
python3 small_scenario.py             # legacy 6-sat scenario
```

### Unified utility and the two-MPC couplings

```bash
cd hypatia_sim
python3 -m oec_sim.run_all --check-golden      # legacy regression gate
python3 -m oec_sim.run_all --utility unified --oracle
# both routing/depth couplings, into their own output dir so the canonical
# gbs-limited run stays committed alongside
python3 -m oec_sim.run_all --scenario fabric-limited --utility unified \
        --couplings --congestion --oracle --out-suffix _couplings
python3 -m oec_sim.sweep --couplings --utility unified   # grid over ISL rates
```

### Downstream segmentation (server, in tmux)

```bash
tmux new -s flair_sweep
cd ~/Research-clean/rq-vae
MAX_SAMPLES=500 ./downstream/run_depth_sweep.sh   # subset first, ~1 h
./downstream/run_depth_sweep.sh                   # full 7,050, ~8-10 h
```

### SatViz (Cesium 3D visualization)

```bash
# OEC scenario on the Cesium globe (animated, ISLs/GSLs/MPC overlay):
cd hypatia_sim
python3 -m oec_sim.satviz             # writes oec_scenario/satviz_oec.html

# stock Hypatia static snapshot:
cd hypatia/satviz/scripts
pip3 install ephem
python3 visualize_kuiper_630.py       # writes ../viz_output/kuiper_630.html
# paste your Cesium ion token (free, cesium.com/ion) at line 10 of the
# generated HTML, then open it in a browser
```

---

## Pending

**OEC network simulation:**

1. Add residual-bandwidth/congestion features to the PPO observation (`rl_env.py::_obs()`) — the trained policy's actions are currently rate-independent (see RL section above), which is the main lever left to close its 5–8% utility gap to MPC
2. `mpc-hier`'s admission/drop/backpressure thresholds were tuned once (`HIER_THETA_ADMIT` 0.9→0.5, `PHI_DROP` 0.05→0.01, closing ~1 utility point) but not exhaustively — still trades ~5.9% utility for 3.7× faster solves
3. `satviz.py` route replay so the Cesium viewer's client-side Dijkstra matches predictive/hierarchical routing instead of only the static router — `routes_<sched>.csv` is now written by the routing-MPC schedulers, so the data side of this is done
4. Robust/stochastic MPC variants (forecast uncertainty), per-GBS antenna constraints
5. Optimize Jain's index directly rather than through the maximin surrogate — needs a formulation the offline bound can also carry, which a ratio of quadratics is not
6. Freeze a more *abstract* routing decision in `mpc-hier-route` (a serving-GBS assignment, or a route class) instead of a concrete path — concrete paths survive only 14.4% of the time on a moving LEO fabric (see the two-MPC section)
7. Retrain PPO under the unified reward — `ppo_oec.zip` was trained against the legacy single-factor score and is stale for `--utility unified`
8. Train PPO on the full protocol (100 seeds, `starlink-550` OOD-topology transfer) on the NCSU server — this pass used a reduced local run (30 training seeds, 2 eval seeds/regime) to fit a CPU laptop

**RQ-VAE compression:**

9. Complete FLAIR-1 dedicated-model sweep for depths 2 and 4 (truncation eval already covers 1/2/4/8/16)
10. ~~Run the downstream segmentation sweep~~ — **done**: full 7,050-image val population, real `oec_sim/quality_table.json` wired in (see "Downstream Task Utility"). Follow-up if it's ever worth the time: verify `mIoU_ref` against FLAIR-1-main's official *test* split rather than val, to get a number directly comparable to IGNF's published one
11. Run NAC entropy coding on exported FLAIR codes
12. Run ns-3 end-to-end simulation (full Starlink scenario, ns-3 build unblocked)

---

## References

- RQ-VAE: "Residual Quantized Variational Autoencoders", CVPR 2023
- EuroSAT: https://github.com/phelber/eurosat
- FLAIR-1: https://github.com/IGNF/FLAIR-1
- Hypatia: https://github.com/snkas/hypatia

## Contact

Hemanth Sudhaharan — NICE Lab, NC State University — hsudhah@ncsu.edu
