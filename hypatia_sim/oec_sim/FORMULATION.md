# Network Parameters in the Notation of the Overleaf Formulation

States every simulation parameter in the mathematical form of *OEC RQ-NAC*
(Sec. 1.1-1.2), ready to paste into the Overleaf document. Source of truth:
`oec_sim/config.py`. Updated 2026-08-20 to close the four items from
Dr. Liu / Xuanhao's Aug 7-8 feedback (routing, timeliness, RL baseline,
optimality bound) plus the three smaller open items (depths, metrics,
committed sweep) -- see `hypatia_sim/oec_sim/README.md` for the full list
and what's implemented vs. left as future work.

## Sets and time

| Symbol | Value in the simulation |
|---|---|
| $\mathcal{S}$ | Kuiper shell-1 (Hypatia `kuiper_630`): Walker-delta 1156/34/1, $\lvert\mathcal{S}\rvert = 1156$ satellites in 34 planes x 34, altitude 630 km, inclination 51.9deg, orbital period 5830 s |
| $\mathcal{G}$ | $\lvert\mathcal{G}\rvert = 4$ GBSs: Tokyo, New York, Sao Paulo, Sydney (Hypatia top-100 city list) |
| $\Delta t$ | 30 s |
| $\mathcal{T}$ | $\lvert\mathcal{T}\rvert = 601$ slots, total 18 000 s (5 h ~ 3.1 orbital periods) |

## Time-varying links $\mathcal{E}_t$ and capacities $C_{ij}(t)$

An ISL $(i,j)$ exists in $\mathcal{E}_t$ iff line-of-sight and range hold
(unchanged from the July version; +Grid pattern, 2312 candidate links, all
feasible at Kuiper spacing). A GSL $(i,g)$ exists iff elevation $\ge 20$deg.

**Two named capacity regimes** (`config.SCENARIO`, `config.apply_scenario`),
added after an Explore-agent audit found that at the original rates the
per-GBS aggregate budget is $\ge$12x tighter than any ISL segment a route
could cross, so ISL contention is mathematically unreachable (max measured
ISL utilization ~0.08) -- congestion-aware routing has nothing to act on
under those rates:

| regime | $C_{ij}$ (ISL) | $C_{ig}$ (per sat-GBS GSL, new) | $C_g$ (GBS aggregate) | task load scale |
|---|---|---|---|---|
| `gbs-limited` (default; published headline numbers) | 100 Mbps | 100 Mbps | 2 Mbps | 1x |
| `fabric-limited` (routing.py ablations) | 100 Mbps | 20 Mbps | 200 Mbps | 20x |

Every route is still Dijkstra shortest-*something* -- either the static
geometric distance (`route_mode='static'`, default) or a congestion-aware
predicted cost recomputed per horizon step (`route_mode='predictive'`, see
below). Routes never transit another GBS.

## Tasks $\mathcal{K}$

Unchanged generation (8 AOIs, 1800 s cooldown, elevation $\ge 40$deg,
64 tasks/run, seed 42), with three additions per the rotting-backlog fix:

| Symbol | Value |
|---|---|
| $N_k$ | $\sim \mathcal{U}\{100{,}000,\,300{,}000\} \times$ LOAD_SCALE |
| $w_k$ | $\sim \mathcal{U}\{1,2,3\}$, aged by the hierarchical scheduler's backpressure term (see below) |
| $d_k$ | soft deadline, freshness $\phi_k(t) = 1$ before $d_k$, $\exp(-\alpha(t-d_k))$ after, $\alpha=1/300\,\text{s}^{-1}$ |
| drop / reject | a task can be explicitly `dropped` (abandoned mid-flight) or `rejected` (refused at admission) -- both are now first-class, reported outcomes instead of silent starved backlog |

## Compression depths $\mathcal{D} = \{2, 4, 8, 16\}$

Depth-16 finished training (2026-08); the $\{1,2,4,8\}$ stand-in used while
it trained is retired.

| $q$ | $b_q$ (B/img) | PSNR (dB) | LPIPS | $u_q = 1-\text{LPIPS}_q$ |
|---|---|---|---|---|
| 2 | 176 | 19.72 | 0.5570 | 0.4430 |
| 4 | 352 | 20.35 | 0.5314 | 0.4686 |
| 8 | 704 | 20.73 | 0.5171 | 0.4829 |
| 16 | 1408 | 21.06 | 0.5039 | 0.4961 |

Still measured on 7,050 FLAIR-1 val images, single depth-16 model truncated
to its first $q$ stages (`truncation_eval.txt`). **Research risk, unchanged
by this pass:** $u_q$ spans only 0.443 -> 0.496 (+12%) while $b_q$ spans 8x,
so depth choice is still driven mainly by deliverability, not reconstructed
quality; the real fix is swapping in downstream-task metrics (mIoU / F1),
still on the README pending list.

## MPC objective (eq. 8, extended)

$$\max \sum_{k,q,\tau}\Big[ w_k\,\phi_k(t^{\text{arrive}}_{k,\tau})\,u_q
      + \lambda_1(H-\tau)\,b_q\!\cdot\!8/10^9
      - \lambda_{\text{late}}\,w_k\,\frac{\max(0,\,t^{\text{arrive}}_{k,\tau}-d_k)}{\Delta_{\text{ref}}}\Big]
  \frac{y_{k,q,\tau}}{N_k}$$

Two additions close Xuanhao's "include timeliness / delay-related terms"
ask, which previously entered only through $\phi_k$ evaluated at slot end:

* $t^{\text{arrive}}_{k,\tau} = (t+\tau+1)\Delta t + \text{route delay}$ --
  `Topology.delay_ms()` was defined but never called before this pass; now
  every utility/tardiness term uses the real propagation delay of the path
  actually assigned to $(k,\tau)$.
* an explicit tardiness penalty $\lambda_{\text{late}}=5\times10^{-2}$,
  $\Delta_{\text{ref}}=300\,\text{s}$, separate from the multiplicative
  $\phi_k$ discount. Consequence: MPC can now rationally choose **not** to
  transmit a task's remaining backlog once $t^{\text{arrive}}-d_k$ is large
  enough that the tardiness cost exceeds the (already $\phi_k$-decayed)
  value -- this is what the rotting-backlog fix (hierarchical MPC) makes
  explicit via admission control / drop instead of silent non-delivery.

Constraints (1) one depth per task, (2) link/GBS capacity, (3) encoder
pipeline, (4) volume are unchanged in spirit but the encoder-pipeline
constraint was reformulated from $O(H^2)$ to $O(H)$ nonzeros per task
(a cumulative-delivery variable $Y_{k,\tau}$ replaces rebuilding the running
sum at every $\tau$) -- an Explore-agent finding that the old form generated
~468k nonzeros at $H=60$, 64 active tasks; the new form generates ~25k and
gives a 3-10x MILP speedup, which is also what makes the offline oracle
(below) and the routing fixed point (below) tractable.

## MPC-predicted link costs + per-predicted-step Dijkstra

Dr. Liu: *"MPC predicts link costs, Dijkstra computes the path at each
predicted step; MPC and Dijkstra are not necessarily mutually exclusive."*
Implemented in `oec_sim/routing.py`, `route_mode='predictive'`
(scheduler name `mpc-congestion`):

$$w_e(\tau) = d_e(\tau) + \beta\,d_{\text{ref}}\,
             \frac{\rho_e(\tau)}{1-\min(\rho_e(\tau),\,\rho_{\max})},\qquad
  \rho_e(\tau) = \frac{f_e(\tau)}{C_e\Delta t}$$

an M/M/1-style congestion penalty on top of propagation distance, where
$f_e(\tau)$ is the bits the MPC's *own current plan* puts on edge $e$ at
predicted step $\tau$. Because the plan depends on the routes and the
routes depend on the plan, `MPCScheduler._solve` iterates this to a damped
fixed point (method-of-successive-averages, $\eta_{it}=1/(it+2)$,
`MPC_ROUTE_ITERS=3` rounds, keep-best-objective-iterate as the safety net --
iteration 0 is exactly today's static geometric routing, so the predictive
variant can never score worse than the static one on its own predicted
objective). $\rho$ is clipped at 0.99, never used to delete an edge, so a
saturated link never disconnects a task's whole horizon of variables.

Only single-path routing is implemented (`MPC_ROUTE_NPATHS=1`, the literal
directive); multipath / candidate-path-set routing is documented future
work, not built.

**Only measurable under `SCENARIO='fabric-limited'`** -- see the capacity
table above; under `gbs-limited` the GBS aggregate budget dominates every
route regardless of ISL cost, so `mpc-congestion` is a provable no-op there
(verified: iteration-0 output is bit-identical to `route_mode='static'`).

## Hierarchical MPC + the rotting-backlog fix

Dr. Liu: *"hierarchical MPC ... one MPC to select the best path, another
one [for the rest]"*. Implemented in `oec_sim/hier.py`, scheduler
`mpc-hier`:

| | upper (`CoordinatorMPC`) | lower (per-task MILP) |
|---|---|---|
| cadence | every `HIER_MACRO_SLOTS`=20 (10 min) | every `MPC_RESOLVE_EVERY`=5 + on arrival |
| horizon | 6 macro-steps (~2 h) | `HIER_LOW_HORIZON`=20 slots (10 min) |
| form | LP, continuous depth $x_{k,q}\in[0,1]$, GBS-aggregate budget only | MILP, integral depth, same $O(H)$ machinery as flat MPC |
| decides | admission, explicit drop, per-task bit budget, aged priority | depth $x_{k,q}$, schedule $y_{k,q,\tau}$ within the budget |

**Rotting = stale-backlog starvation**: a task whose $\phi_k$ has decayed
toward 0 still holds backlog and still generates MILP variables every
re-plan, but earns ~0 objective value, so it is never served and never
removed -- it silently inflates problem size and the backlog metric
forever. Three mechanisms, placed per the rule "admission/drop live
upstairs (irreversible), aging/backpressure lives downstairs (fast
tie-breaking)":

1. **Admission control** (upper, at arrival): reject if the cheapest depth
   can't plausibly finish before $d_k$ given the forecasted GBS budget
   (`ADMISSION_ON`, `HIER_THETA_ADMIT=0.5`, tuned down from 0.9 -- see below).
2. **Explicit drop** (upper): abandon if $\phi_k(t) < \text{PHI\_DROP}=0.05$
   with $>2\%$ backlog remaining, or $t-d_k > \text{T\_ABANDON\_S}=3600$.
   Reported as `dropped`/`n_dropped`, not hidden.
3. **Backpressure / anti-starvation** (lower, Lyapunov drift-plus-penalty
   style): objective gains $\text{AGING\_ETA}\cdot\text{backlog\_bits}(k,q)/10^{11}$,
   so large remaining backlogs get preferential clearing before they rot.

Measured trade-off (gbs-limited, seed 42), after a small grid search over
`PHI_DROP` and `HIER_THETA_ADMIT` (2026-08-20/21) that moved the defaults
from `PHI_DROP=0.05, HIER_THETA_ADMIT=0.9` to `PHI_DROP=0.01,
HIER_THETA_ADMIT=0.5` and closed about a point of gap (60.71 -> 61.4
utility, 93.4% -> 95.1% delivered): `mpc-hier` utility **61.4** vs flat
`mpc` **64.95** (5.9% vs 0.4% gap to the oracle bound below) but **4.3x
faster** per-run solve time (3.7 s vs 15.8 s total solve time across the
run) and 16 tasks explicitly reported dropped instead of silently starved.
Still a first-pass tuning (5 combinations tried, not an exhaustive search)
-- closing the rest of the gap is a natural next step.

## Offline HiGHS upper bound

"Look into the optimization software -- the upper bound should be in
HiGHS." `oec_sim/oracle.py`, three bounds of increasing tightness:

1. **Analytic ceiling**: every image at the best depth, delivered
   instantly. Sanity check only.
2. **LP bound**: depth relaxed to continuous $x_{k,q}\in[0,1]$, slots
   aggregated into 5-slot windows, ISL/GSL rows dropped (valid because they
   never bind under `gbs-limited` -- see capacity table). Utility must be
   evaluated at the *start* of each aggregated window, not its end, to stay
   a genuine upper bound (using the end under-credits early delivery and
   can make the "bound" tighter than what's achievable -- caught and fixed
   during this pass).
3. **MILP bound**: same relaxed constraint set but full time resolution and
   integral depth, solved via `scipy.optimize.milp` (HiGHS) with a time
   limit; reports the HiGHS dual bound even when the incumbent hasn't
   converged, which is itself a valid upper bound.

Measured (gbs-limited, seed 42, 120 s MILP time limit): LP bound 65.66,
MILP dual bound 65.22, and every scheduler is within that:

| scheduler | utility | gap to bound |
|---|---|---|
| mpc | 64.95 | 0.4% |
| greedy-adaptive | 64.12 | 1.7% |
| greedy-fixed-8 | 64.15 | 1.6% |
| greedy-fixed-4 | 62.25 | 4.5% |
| mpc-hier | 61.4 | 5.9% |
| greedy-fixed-2 | 58.85 | 9.8% |
| greedy-fixed-16 | 24.43 | 62.5% |

MPC is essentially optimal (0.4% gap) in this regime -- the flat MPC's
apparent near-tie with `greedy-fixed-8` in the old $\{1,2,4,8\}$ table was
real, not a sign MPC has more headroom to find.

## MPC vs. RL (PPO)

`oec_sim/rl_env.py` (`OECDepthEnv`) steps per (slot, task) decision point --
`Discrete(6)`: commit one of the 4 depths, defer, or drop -- so the action
space is independent of the active-task count and of the constellation
size, which is what lets a trained policy transfer to an unseen topology
without retraining. Reward is the task's realized $\Delta$(MILP objective)
plus potential-based backlog shaping (Ng, Harada & Russell 1999 -- provably
preserves the optimal policy).

**A real infinite-loop bug was found and fixed during this pass.**
`_advance_to_next_decision()` used `not self._active_queue` (empty-list
truthiness) as its "need to rebuild this slot's task queue" signal. A
deferred task pops the queue to `[]` *without* advancing the slot clock;
since `[]` is also falsy, the very next loop iteration immediately rebuilt
the queue from the *same* slot's still-active tasks -- including the task
that was just deferred -- so a policy that deferred one task consistently
would be asked about it forever at a frozen `t`. Caught empirically: a
diagnostic run logged 3,000 consecutive decisions with `env.t` stuck at
420/601. Fixed by using an explicit `None` sentinel for "not yet built,"
distinct from "drained and needs to wait for the next slot"
(`rl_env.py::_advance_to_next_decision`). Post-fix, a full 601-slot episode
completes in ~0.2 s.

**Trained**: `MaskablePPO`, 300k timesteps, 3 load regimes (1.5/2.0/2.5
Mbps) x 10 seeds each, `net_arch=[128,128]`, ~1000-1150 steps/s on a CPU
Mac (`ppo_oec.zip`). **Evaluated** in-distribution (2.0 Mbps) and two OOD
load regimes (4.0 Mbps light, 0.75 Mbps heavy), 2 held-out seeds each:

| regime | seed | RL utility | RL delivered% | RL µs/decision | MPC utility | MPC delivered% | MPC µs/decision |
|---|---|---|---|---|---|---|---|
| in-dist (2.0 Mbps) | 100 | 58.13 | 100.0% | 521 | 61.58 | 95.5% | 218,409 |
| in-dist (2.0 Mbps) | 101 | 52.31 | 100.0% | 507 | 55.55 | 97.2% | 243,410 |
| OOD-light (4.0 Mbps) | 100 | 58.13 | 100.0% | 675 | 62.44 | 95.6% | 167,938 |
| OOD-light (4.0 Mbps) | 101 | 52.31 | 100.0% | 1,376 | 56.14 | 97.3% | 145,424 |
| OOD-heavy (0.75 Mbps) | 100 | 54.76 | 100.0% | 669 | 58.92 | 95.5% | 173,462 |
| OOD-heavy (0.75 Mbps) | 101 | 50.83 | 100.0% | 512 | 53.27 | 97.1% | 162,685 |

Three real findings, not the ones hypothesized in advance:

1. **PPO loses to MPC on utility, as predicted** -- 5-8% below MPC in every
   regime tested. Confirms the pre-registered expectation rather than
   fitting one after the fact.
2. **PPO is ~250-400x faster per decision** (500-1,400 µs vs. MPC's
   145-243 ms) -- the inference-cost story holds, strongly.
3. **The observation has no capacity/congestion feature** (`rl_env.py`'s
   `_obs()`: task intrinsics + active-count + mean freshness only, no
   residual-bandwidth term) -- so the trained policy's *actions* are
   provably rate-independent: utility is bit-for-bit identical between the
   2.0 Mbps and 4.0 Mbps regimes at a fixed seed (58.13/58.13, 52.31/52.31)
   because it picks the same depths in the same order regardless of load;
   only the *shared execution engine's* real capacity constraint makes the
   0.75 Mbps numbers differ. The policy still delivers 100% of images in
   every regime tested (vs. `greedy-fixed-8` collapsing to 13-20 utility at
   0.75 Mbps) because it happens to have learned conservative depth
   choices, not because it senses congestion -- an accidental robustness,
   not an adaptive one. Adding residual-bandwidth/demand features to the
   observation (the design originally called for this; the shipped `_obs()`
   dropped them for a smaller vector) is the clear next step and could
   close real distance to MPC's 5-8% margin.

## Committed congestion sweep

`oec_sim/sweep.py` replaces the one manual congested run that "isn't saved
in the repo yet" (meeting_script_oec.txt, section 4/6). Full grid: 6 rates
x 5 seeds x 6 schedulers = **180 rows**, `oec_scenario/sweep/results.csv`,
`plots/sweep_utility_vs_load.png`. Mean utility by rate (5-seed average):

| GBS rate (Mbps) | best fixed depth | best-fixed utility | mpc utility |
|---|---|---|---|
| 0.75 | depth-2 | 57.08 | **60.34** |
| 1.0 | depth-4 | 60.38 | **61.18** |
| 1.5 | depth-4 | 60.38 | **62.34** |
| 2.0 | depth-8 | 62.22 | **62.96** |
| 3.0 | depth-8 | 62.22 | **63.61** |
| 4.0 | depth-16 | 63.92 | 63.76 |

The best *fixed* depth climbs 2 -> 4 -> 8 -> 16 as load falls; `mpc` beats
the best fixed depth in 5 of 6 regimes and is statistically tied (within
0.2 utility) at the one point (4.0 Mbps, no scarcity) where "always max
depth" is already close to optimal. This is the crossover the one-off
manual run gestured at, now with 5 seeds and committed to the repo.
