# Reward-normalization bug: findings and fix (2026-10-20)

## Background

The `'realistic-bigdata'` data-volume sweep (config.py, sweep.py Part 12/13)
found that Flat-MPC, OrbitFlow, and Two-Level-MPC collapse to identical,
worse-than-greedy delivery at every congested grid point. This was flagged
as an open hypothesis: "the MILP objective's per-task reward normalization
structurally disadvantages very large tasks once overall capacity is
scarce." This doc confirms that hypothesis, locates the exact mechanism,
implements a scoped fix, and documents a second, independent finding about
why the scaled-bandwidth sweep couldn't have shown routing differentiation
in the first place.

## Finding 1: the 1/N_k normalization bug (CONFIRMED, not just a hypothesis)

**Location:** `utility.py`, `y_coeff()` (was line 346), plus four sibling
sites: `accumulate()` (244), `objective_tardiness_coeff()` (228), and
`SegmentBlock.add_rows()` (455, 463).

**Mechanism:** every one of those functions divided a task's accumulated
reward by `task.n_images` (N_k) to produce a normalized, per-task score in
roughly `[0,1]` — a deliberate design choice so a huge task can't dominate
total utility just by being big (see `utility.py`'s module docstring: `U_k`
is explicitly a `[0,1]`-bounded per-task score).

The side effect: the **marginal reward per image transmitted is `c/N_k`**,
so it is **inversely proportional to task size**. The coverage term is also
normalized by *fraction* delivered, not absolute images, so the normalized
value earned for reaching e.g. 25% of a task is the same fixed amount
*regardless of task size* — but the bits needed to get there scale linearly
with N_k. Value-per-bit-of-capacity is therefore inversely proportional to
task size, full stop. Under a capacity constraint, any LP/MILP spends
capacity where reward-per-bit is highest, so it will always prefer fully
completing many small tasks over partially feeding one large task — not
because the large task doesn't matter, but because the objective literally
says progress on it is worth less per bit. Confirmed this isn't escapable by
choosing a cheaper compression depth: `task.n_images` is fixed at
task-generation time (`tasks.py`, inverted from the sampled GB target via
the depth-16 reference payload) and never depends on which depth the
scheduler later picks.

## The fix: cap N_k at what's achievable, not what's requested

Added `utility.reward_norm(task)`:

```python
def reward_norm(task):
    if C.TASK_SIZE_MODE != 'data_gb':
        return float(task.n_images)
    return max(1.0, min(float(task.n_images),
                        task.encoded_by(task.deadline_s)))
```

`task.encoded_by(task.deadline_s)` (already existed in `tasks.py`) is "how
many images this task could possibly have encoded by its own deadline,"
using the fixed `ENC_IMGS_PER_S` encoder rate. Capping the normalizer there
means a task sized to fit within its deadline is unaffected; a task that
was never going to fully complete gets judged against what it could
*actually* achieve, restoring a fair `[0,1]` scale regardless of size.

All five sites now call `reward_norm(task)` instead of raw `task.n_images`.

### Scoping — important

**The cap only applies when `TASK_SIZE_MODE == 'data_gb'`** (i.e. only the
`'realistic-bigdata'` scenario). This was not optional:

- Checked empirically: **27/64 tasks (42%) in the default `'gbs-limited'`
  scenario ALSO exceed their own deadline-encode ceiling** — `TASK_IMAGES_MAX`
  (300,000) was never designed against `ENC_IMGS_PER_S × deadline`
  (~146,000–292,000 images achievable in a 30–60 min deadline).
- Applying the cap unconditionally broke `--check-golden`: every scheduler's
  utility moved (Flat-MPC 64.1 → 68.2, greedy-fixed-16 24.4 → 27.2, etc.),
  silently invalidating every already-published number in the paper (Tables
  2–4 of the results memo), not just the data-volume experiment.
- Gating on `TASK_SIZE_MODE=='data_gb'` keeps every `'images'`-mode scenario
  (every existing table) **bit-for-bit identical** — confirmed,
  `--check-golden` passes again after scoping. Only `'realistic-bigdata'`
  is affected.

**Consistency note for the paper:** Tables 2–4 (scenario `'realistic'`) and
Tables 5–6 (scenario `'realistic-bigdata'`) now use different normalizers.
This should be documented as a methods footnote, not left silent — it's
scoped on principle (only the scenario where the raw-N_k formula actually
produces infeasible-by-construction normalizers), not cherry-picked for
a better number.

## Validation: the fix works

At Table 5's exact congested point (n=150, GB 2.0–5.0GB, seed 42):

| Scheduler | Utility (before) | Utility (after fix) | Delivery |
|---|---|---|---|
| greedy-fixed-16 | 15.94 | **149.793869** | 60.4% |
| Flat-MPC | 15.56 | **149.793869** | 10.8% |

**Utility ties bit-for-bit** despite wildly different raw delivery (60.4% vs
10.8%). Mechanism: both schedulers already deliver enough per-task to
saturate the capped ceiling for every task they choose to serve — extra
images beyond that are worth zero, so Flat-MPC rationally pushes a *subset*
of tasks past their ceiling rather than spreading capacity thin across all
of them, landing at the same aggregate normalized score as greedy's more
evenly-spread policy. **The utility gap that motivated this whole
investigation is closed at n=150**, not just narrowed. Raw delivery-fraction
is unaffected by the fix (it's a different, non-normalized metric) — this
is a correction to the utility *score*, not a claim that MPC now moves more
bits.

At `MAX_TASKS=2000` (the scale used in the Section 7 scaled sweep), nearly
the *entire* grid is affected, not just the congested end — e.g. even the
"control point" (0.1–0.3GB, scale=1) has 223/2000 tasks (11%) over ceiling,
and scale=2's control point has 1464/2000 (73%). The full 48-combo scaled
sweep is being rerun under the fix on the server (tmux session
`orbitflow-scaled-fixed`, resumable, pre-fix results archived as
`results_realistic_datasize_scaled_PREFIX.csv` for before/after comparison).

## Finding 2: the scaled-bandwidth sweep couldn't have shown routing differentiation anyway

Independent of the normalization bug: at Gbps-scale link rates, transmitting
1GB takes **~0.8 seconds**; encoding that same GB takes **~2.44 hours**
(`ENC_IMGS_PER_S ≈ 81.07 img/s`, `PAYLOAD_B[16]=1408 B`). Bandwidth is
~4 orders of magnitude faster than encoding for any single task, so **no
individual task can ever be fabric-bound** at these rates — it is always
throttled by its own encoder first. Scaling ISL/GSL/GBS up or down (the
whole premise of the 1x/2x/5x scaled sweep) therefore can't matter much for
an individual task's feasibility, which is consistent with Flat-MPC/
OrbitFlow/Two-Level-MPC continuing to track each other closely across scale
in Section 7's data: the routing decision was never the thing standing
between a task and completion.

Concretely: a task is only encoder-feasible within its deadline if
`V_GB ≤ deadline_s / 8762`, i.e. **≤0.2–0.4GB** for the 1800–3600s deadline
window used everywhere. That's almost exactly Table 5/6's "control point" —
anything above it is encoder-dead *before the network ever sees the data*,
regardless of ISL/GSL/GBS capacity.

### What would actually let routing matter

Two things need to hold at once, which no current scenario combines:

1. **Deadlines generous enough that encoding finishes with time to spare**
   (currently: deadlines are fixed constants independent of task size, so
   anything over ~0.3–0.4GB is already encoder-dead).
2. **Many tasks' encoded backlogs landing on the network concurrently, with
   aggregate bandwidth too scarce to carry all of it** — what
   `'fabric-limited'` (`ISL_RATE_BPS=1e6`) was built for, just never
   combined with data-volume-sized tasks.

**Proposed next scenario:** data-volume task sizing (`TASK_SIZE_MODE='data_gb'`)
+ deadlines scaled to each task's own encode time plus real slack (e.g.
`deadline_s = max(TASK_DEADLINE_MIN_S, slack_factor * V_GB * 8762)`,
`slack_factor` 1.5–2x) + `fabric-limited`-style scarce ISL/GSL/GBS. This
needs its own calibration probe (same process `'realistic-bigdata'` itself
went through — checking it produces real, non-degenerate contention rather
than "always fits" or "always encoder-dead") before a full sweep is worth
running. **Not yet implemented** — this is a scoped recommendation, pending
confirmation this is the direction to take (Hemanth asked to hold off and
let the fixed scaled-sweep finish first before deciding whether to run this
in parallel).

## Status as of 2026-10-20

- Fix implemented and scoped in `utility.py`, deployed to both the local
  worktree and the server worktree (`~/Research-clean-wt-results-index`).
- `--check-golden` passes on both — zero effect on any `'images'`-mode
  scenario.
- Full 48-combo scaled sweep (Section 7 of the results memo) relaunched
  under the fix on the server, tmux session `orbitflow-scaled-fixed`,
  resumable via the existing `sweep.py` resume logic. Pre-fix results
  preserved at `results_realistic_datasize_scaled_PREFIX.csv` for
  before/after comparison once the rerun completes.
- Finding 2's proposed scenario redesign is a recommendation only, not yet
