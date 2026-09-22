# Session Notes — OrbitFlow revision (2026-09-21/22)

Working notes for this session's response to Xuanhao's two emails + his
OrbitFlow paper draft. Casual, not polished — see `RESULTS_INDEX.md` for
the formal catalog and the Meeting Prep artifact for the presentation-ready
version. Nothing here has been committed or pushed.

## Results: before vs. now

**Downstream mIoU (what "quality" means to the utility function)**
- Before: measured on a 7,050-image *val*-split subset (mislabeled "full"
  at the time — it was always half the true 14,125-image val population).
  `mIoU_ref` = 68.87%, didn't match IGNF's own published baseline (0.5443)
  or the HF model card (58.6%) for this checkpoint — an open question.
- Now: measured on the complete, official 15,700-image **test** split
  (the split FLAIR-1's own numbers are actually benchmarked against).
  `mIoU_ref` = 52.12%, closely matches IGNF's 0.5443 baseline — mismatch
  resolved, was a wrong-split bug, not a broken pipeline.
- The whole quality table was remeasured this way, not just the reference
  point, so nothing mixes the two splits. Full table below.

**Network capacities**
- Before: two scenarios, both artificial (`gbs-limited`: 2 Mbps GBS;
  `fabric-limited`: 1 Mbps ISL), deliberately tuned to force congestion for
  earlier ablations, not drawn from real hardware.
- Now: a third scenario, `realistic` (10 Gbps ISL / 1 Gbps GSL / 10 Gbps
  GBS) — Xuanhao's own nominal point, checked against real published specs
  (the GBS number matches an actual Starlink gateway almost exactly).

**Workload (task count)**
- Before: 64 tasks, 8 source regions, one new task per region every 30 min.
- Now: had to grow substantially — at realistic capacities, 64 tasks can
  never create real network contention no matter what (the on-satellite
  encoder physically caps demand well below even the loosest capacity
  setting). Found real scheduler divergence starting at **n=3,000 tasks**
  under a forced-down 2 Gbps ground-station capacity. Needed 56 more
  synthetic source regions (64 total) to even generate that many tasks —
  the original 8 regions hard-capped generation at 4,808 no matter the
  settings, a fixed orbital-mechanics ceiling.

**Scheduler family**
- Before: 5 MPC variants (`mpc`, `mpc-congestion`, `mpc-2level`,
  `mpc-hier`, `mpc-hier-route`) plus greedy baselines; PPO reported in its
  own separate section.
- Now: 3 MPC variants, renamed to match Xuanhao's paper exactly —
  `Flat-MPC`, `OrbitFlow` (the paper's own namesake method), `Two-Level-MPC`.
  The two hierarchical variants are archived (code kept, not deleted, just
  unplugged). PPO now folds into the main comparison table.

**The two planning-regularizer terms (Xuanhao's Eq. 22)**
- Before: present, untested whether they actually mattered.
- Now: ablated (5 seeds × 3 schedulers, baseline vs. zeroed) — negligible
  impact (≤0.1% utility, no solve-time change) under the scenario that
  actually matters for the reported comparison. Removed from the code.
  Caveat: NOT negligible under the old legacy-scale baseline (real 1.3%
  utility shift, real depth-mix change there) — treated as intentional,
  accepted drift since Xuanhao's ask was to remove them outright.

## What this means for the project

The mIoU correction changes every scheduler's absolute utility number
going forward (the quality term's denominator moved), but the *relative*
story — pixel-fidelity massively understates real downstream task
quality — holds up on the corrected data (+181% mIoU spread vs. +12% for
the old pixel-fidelity proxy).

The capacity/workload investigation surfaced a real, previously-invisible
structural fact: at genuinely realistic satellite hardware speeds, a small
workload can *never* create meaningful network scarcity — the on-board
encoder is the actual bottleneck until task counts reach the thousands,
not dozens. That's a finding worth keeping regardless of what else changes.

The regularizer removal and scheduler cleanup both point the same
direction: a leaner, more defensible 3-method comparison (a baseline, a
named hero method, a stronger reference bound) running on capacity
assumptions a reviewer won't immediately question — closer to
submission-shape than before this session started.

## Tables

**mIoU, old vs. new (%, higher = better)**

| Condition | Old (val-split, wrong) | New (test-split, verified) |
|---|---|---|
| floor (blank) | 6.08 | 5.07 |
| q1 | 12.60 | 11.76 |
| q2 | 17.53 | 15.48 |
| q4 | 22.54 | 19.32 |
| q8 | 25.68 | 21.70 |
| q16 | 28.70 | 23.85 |
| uncompressed (ref) | 68.87 | 52.12 |

**Network capacities, old vs. new**

| Scenario | ISL | GSL | GBS (aggregate) |
|---|---|---|---|
| Old `gbs-limited` | 100 Mbps | 100 Mbps | 2 Mbps |
| Old `fabric-limited` | 1 Mbps | 50 Mbps | 500 Mbps |
| New `realistic` | 10 Gbps | 1 Gbps | 10 Gbps |

**Scheduler renames**

| Old name | New name | Status |
|---|---|---|
| `mpc` | `Flat-MPC` | kept |
| `mpc-congestion` | `OrbitFlow` | kept, hero method |
| `mpc-2level` | `Two-Level-MPC` | kept, reference bound |
| `mpc-hier` | — | archived |
| `mpc-hier-route` | — | archived |

**Graphs:** `hypatia_sim/oec_scenario/plots/results_comparison.png` and
`quality_curve.png` are the most relevant existing visuals; none have been
regenerated yet against the new `realistic` scenario.

## What's left

- Lock the final workload size for `realistic` (pending the current server confirmation)
- Run the ISL/GSL/GBS sensitivity sweeps (Xuanhao's fix-two-vary-one ask)
- Fill Xuanhao's 5 empty result tables in his own draft
- Update `RESULTS_INDEX.md` with this session's new results
- Update `paper/sections/*.tex` with corrected numbers + new scheduler names
- Get explicit sign-off before anything touches Overleaf

## What's running right now

- 3 parallel server jobs confirming Flat-MPC/OrbitFlow/Two-Level-MPC at n=3,000, GBS=2Gbps, seed 42 — no result yet, ~20+ min of compute so far

## What it means for the conclusion

Once the workload size and sensitivity sweeps land, this session will have
addressed all four of Xuanhao's requests plus his follow-up: realistic
network assumptions, a verified quality metric, a simpler MPC formulation,
and a cleaner three-method comparison — the remaining work is filling in
numbers against an already-settled design, not more open questions.
