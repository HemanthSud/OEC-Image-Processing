# Results Index

Single place to answer "where did that result come from, and where has it
been shared" — built because results have gone out via README updates,
Claude Artifacts, and (untracked here) email, with no one place tying them
together. If your instructor says "I saw that somewhere," check here first.

**Maintenance rule:** every time a new results Artifact is published, or a
result is emailed/presented to Dr. Liu or Xuanhao, add one row here in the
same sitting — don't rely on remembering it later.

## Compression results (RQ-VAE)

| Result | Source of truth | Shared via | Status |
|---|---|---|---|
| EuroSAT full sweep (spatial × depth, PSNR/SSIM/LPIPS/FID/compression) | `README.md` "EuroSAT — Full Sweep"; raw numbers in `results/summary.json` | README (committed, public on GitHub) | Current. 8×8×8 is the best config on every metric. |
| FLAIR-1 subset training (val split, 3 depths) | `README.md` "FLAIR-1 — Subset Training" | README | Current |
| FLAIR-1 depth-truncation experiment (reuse depth-16 model at shallower depths) | `README.md` "FLAIR-1 — Depth Truncation Experiment" | README | Current. Negative result: truncation underperforms a dedicated model at every depth. |
| On-board compute delay (A6000, 100 runs) | `README.md` "On-Board Compute Delay" | README | Current |
| EuroSAT classifier smoke-test | `results/classifier_test_results.json` | Not written up anywhere in prose | **Orphaned raw file** — 100% accuracy but only 1 of 10 classes has any test examples (see per-class breakdown); looks like a degenerate/incomplete run, not a citable result. Don't reference this number without re-checking the test set. |

## Downstream task / unified utility

| Result | Source of truth | Shared via | Status |
|---|---|---|---|
| Downstream segmentation mIoU quality table (`s_q` per depth) | `README.md` "Downstream Task Utility"; `oec_sim/quality_table.json`; per-class metrics added in commit `ac6548e` | README, Overleaf formulation notation | Current / measured, source label `flair-unet-r34-rgbie / val7050 / 2026-08-31`. IGNF baseline mismatch still open (measured mIoU_ref 68.87% vs. checkpoint-gate target ~54.43% vs. HF card's 58.6% — see README "Pending" #10 and `[[project_state]]`). |
| Unified utility function definition (single formula replacing 5 disagreeing definitions) | `README.md` "Unified Utility"; `oec_sim/utility.py` | README, Overleaf formulation (`main.tex` §1) | Current |
| Legacy-vs-unified utility margin (headline: mpc +1.2% legacy → +32.2% unified) | `README.md` "MPC Scheduler vs. Fixed Depths..."; `oec_scenario/golden/legacy_summary.json` | **[Downstream-Grounded Utility](https://claude.ai/code/artifact/57268d58-3744-47c8-949b-6d011c022d86)** artifact (linked `README.md:108`); commit `0ecb36a` | Current. This is the one artifact that's actually linked anywhere durable. |
| 10-scheduler fabric-limited comparison (mpc-2level wins, 0.5% off the offline bound) + two-MPC routing/depth split | `README.md` "Two-MPC Split"; `hypatia_sim/oec_scenario_couplings/summary.txt` + `task_outcomes_*.csv` | **[Signal & Depth](https://claude.ai/code/artifact/c044ebf4-4a26-4c6b-ba50-7daa08da6c32)** artifact (2026-09-08 full synthesis) — **now linked in README** (added alongside the Downstream-Grounded Utility link, this session) | Current. Supersedes the placeholder-quality-table version of this comparison — don't cite the old numbers next to these, they're on different utility scales. |
| Depth-selection decision distributions per scheduler | `paper/sections/decisions.tex` (local, untracked draft); `task_outcomes_*.csv` | Not yet shared anywhere outside this repo | Reported for one scenario/seed only — flagged in the draft as needing repeats across scenarios. |
| Route-freezing negative result (usable only 7.7% of the time) | `README.md` "Two-MPC Split" section | README, Signal & Depth artifact | Current — structural finding, not a tuning failure. |
| RL (PPO) baseline vs. MPC | `README.md` "RL Baseline (PPO)"; `hypatia_sim/ppo_oec.zip` | README | Stale for `--utility unified` — trained against the legacy score (README "Pending" #1, #7). |
| GBS-rate and ISL-rate sensitivity sweeps | `paper/sections/sweeps.tex` (local draft); `hypatia_sim/oec_scenario/sweep/results.csv`, `results_routing.csv` | Not shared anywhere yet | **Provisional** — both still on the placeholder quality table, explicitly flagged as needing a rerun before the exact numbers are final. |

## Concept / explainer artifacts

| Result | Source of truth | Shared via | Status |
|---|---|---|---|
| Orbit School — from-zero primer covering every concept in the project | n/a (a teaching aid, not a new result — see its own footer: "nothing here was re-run") | **[Orbit School](https://claude.ai/code/artifact/3c074b01-1f64-4eae-bb60-e83be00b75dc)** — **now linked at the top of README**, this session | Current as of 2026-09-09 |

## Paper drafts (not yet public)

| Result | Source of truth | Shared via | Status |
|---|---|---|---|
| Local "experimental report" (setup/compression/utility/decisions/sweeps) | `paper/report.tex` + `paper/sections/*.tex` | **Not pushed anywhere** — untracked by git, not on Overleaf | Draft. See the paper-restructuring plan for next steps. |
| Overleaf `OEC-RQ-NAC` (Xuanhao's formulation skeleton) | https://www.overleaf.com/project/6a4917ebb65d67631b69b266 | Shared with Hemanth by Xuanhao (March 2025) | System model + MPC objective only, 2 pages. Not yet merged with the local experimental report. |

## Not research (excluded from this index on purpose)

"Morning Brief — Aug 24" and "Morning brief" artifacts are personal daily
briefs, unrelated to the OEC project — listed here once so they don't get
mistaken for missing research artifacts later.

## Known gap

Email correspondence with Dr. Liu / Xuanhao (`hsudhah@ncsu.edu`) isn't
searchable from this tool's Gmail connector, which is bound to a personal
account instead. If your instructor references something sent by email,
check that mailbox directly — this index can't do it for you yet.

### Manual sharing log

*(Add a line whenever something in this file gets emailed, presented, or
otherwise shared outside this repo — this is the part no tool can infer.)*

- _(none logged yet)_
