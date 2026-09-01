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

Measured on 7,050 FLAIR-1 val images, single depth-16 model truncated to its
first $q$ stages (`truncation_eval.txt`).

**The research risk this pass set out to remove:** $u_q$ spans only
0.443 -> 0.496 (+12%) while $b_q$ spans 8x, so depth choice was driven almost
entirely by deliverability rather than by quality. That is why `greedy-fixed-8`
(64.15) came within 1.2% of the MPC (64.95) -- there was barely a decision to
get right.

The fix is to replace the reconstruction proxy with **downstream segmentation
performance**: $s_q$ = mIoU of the FLAIR-1 baseline segmenter run on the
depth-$q$ reconstruction. `UTIL_QUALITY_SOURCE = 'miou'` selects it and
`utility.load_quality_table()` reads `oec_sim/quality_table.json`, written by
`rq-vae/downstream/harvest_metrics.py`.

Two anchorings, `UTIL_QUALITY_ANCHOR`:

* `ratio`: $s_q = \text{mIoU}_q/\text{mIoU}_{\text{ref}}$.
* `floor` (default): $s_q = (\text{mIoU}_q-\text{mIoU}_{\text{floor}})/(\text{mIoU}_{\text{ref}}-\text{mIoU}_{\text{floor}})$,
  where $\text{mIoU}_{\text{floor}}$ is the **blanked-RGB** condition --
  the segmenter run on NIR + Elevation with the optical bands carrying no
  information. The decision-theoretic zero for a scheduler is not
  "mIoU = 0" but "what you get by *not* delivering the image", which is
  exactly that condition. `recon_to_geotiff.py --depth blank` produces it.

Both are written to `quality_table.json` so the choice stays visible and
reversible. **If the spread is still narrow after floor anchoring, that is the
finding** -- report it rather than tuning it away.

The server sweep has landed: `oec_sim/quality_table.json` holds real
mIoU on the full 7,050-image val population (`flair-unet-r34-rgbie`,
`FLAIR-INC_rgbie_15cl_resnet34-unet`, floor-anchored). Measured
$s_q$ = 0.104 / 0.182 / 0.262 / 0.312 / 0.360 at q = 1/2/4/8/16 --
a ~247% spread from q1 to q16, against ~12% for the 1-LPIPS proxy it
replaces. $\text{mIoU}_{\text{ref}}$ = 68.87%, $\text{mIoU}_{\text{floor}}$
= 6.08%. `config.QUALITY_TABLE_FALLBACK` remains as a PROVISIONAL fallback
only for environments without the real file (e.g. a laptop with no server
access); every run that falls back to it still prints
`PROVISIONAL - NOT MEASURED` in `summary.txt` so the two are never confused.

One caveat worth recording: $\text{mIoU}_{\text{ref}}$ (68.87%) does not match
the checkpoint's own self-reported mIoU (58.6%, its HuggingFace model card).
Best-understood cause: FLAIR-1-main ships a held-out-domain *val* split (used
here) and a separately-released official *test* split, and IGNF's published
number was almost certainly benchmarked on the latter. Domain leakage between
our val population and the training domains was checked and ruled out;
per-class IoU is structurally sane (common classes score high, rare ones
lower) -- the signature of correctly-loaded weights evaluated against an
easier population, not a broken pipeline. Verifying against the true test
split needs a 14 GB test-image archive plus an unfetched test-label archive
and predict+metrics on 15,700 images -- judged not worth it, since the OEC
utility only needs a self-consistent $\text{mIoU}_q$ on one fixed population
across depths, not a match to an external benchmark.

## Unified utility (the reported score AND the objective)

Until this pass, utility was defined in **five places that disagreed**: the
realized score (`schedulers._record_delivery`) had only
$w_k\phi_k u_q\rho_k$; the flat MPC objective added a backlog bonus and a
tardiness penalty; the hierarchical upper level had neither; its lower level
swapped the backlog bonus for a Lyapunov backpressure term; and the offline
bound had a third combination. Timeliness, coverage and depth mix lived in
side tables, so no single number said whether a scheduler was good.

`oec_sim/utility.py` is now the single source of truth for all five. With
$\bar w_k = w_k/\max_j w_j$ and arrival time
$t^{\text{arr}} = (t{+}1)\Delta t + \delta_{k,t}$ along the path actually used:

$$U_k \;=\; \bar w_k\Big[\;\omega_Q\, s_{q_k}\,\hat G_k \;-\; \omega_T\,T_k \;-\; \omega_E\, C_k\Big],
\qquad
U \;=\; \sum_k U_k \;+\; \omega_F\,|\mathcal K|\,\min_k \hat U_k$$

| term | meaning | range |
|---|---|---|
| $s_q$ | downstream **segmentation** quality at depth $q$ (mIoU-derived), or the legacy reconstruction proxy $1-\text{LPIPS}_q$ | $[0,1]$ |
| $\hat G_k$ | coverage gain: a **concave** function of delivered fraction, each image weighted by the freshness at which it arrived | $[0,1]$ |
| $T_k$ | tardiness, $\sum_t \frac{\Delta r_{k,t}}{N_k}\min\!\big(1,\frac{\max(0,\,t^{\text{arr}}-d_k)}{\Delta_{\text{ref}}}\big)$ | $[0,1]$ |
| $C_k$ | resource cost, $\sum_t \frac{\Delta r_{k,t}}{N_k}\big[\hat\omega_{\text{tx}}\frac{b_q}{b_{\max}}+\hat\omega_{\text{enc}}\big]$ | $[0,1]$ |
| $\hat U_k$ | $U_k/(\omega_Q s_{\max}\bar w_k)$ — scale-free **and weight-relative** | $\le 1$ |

The $\min(1,\cdot)$ in $T_k$ applies to a *coefficient* (arrival times are
data, not variables), so it costs nothing in linearity.

**Legacy identity.** With $\omega_Q{=}1$, $\omega_T{=}\omega_E{=}\omega_F{=}0$,
one coverage segment of unit width and unit slope, no weight normalization and
$s_q = 1-\text{LPIPS}_q$, this collapses *literally* to the old
$\sum_k\sum_t w_k\phi_k(t^{\text{arr}})u_{q_k}\Delta r_{k,t}/N_k$. Legacy is a
parameter setting, not a code branch, which is why `--utility legacy` still
reproduces every committed number exactly (`--check-golden` enforces it).

### Concave coverage without breaking linearity

Introduce $\lambda_{k,q,\tau,j}\ge 0$ over $J$ segments of width $\Delta_j$ and
**strictly decreasing** slopes $m_1>\dots>m_J$, with $\sum_j\Delta_j m_j = 1$:

$$\hat G_k=\sum_{\tau,j} m_j\,\phi_k(t^{\text{arr}}_\tau)\,\lambda_{k,q,\tau,j},
\quad
\sum_j \lambda_{k,q,\tau,j}=\frac{y_{k,q,\tau}}{N_k},
\quad
\sum_{q,\tau}\lambda_{k,q,\tau,j}\le\Delta_j^{\text{res}}$$

**No binaries and no SOS2 are required.** A concave separable function being
*maximized* linearizes exactly: because the slopes decrease, the LP relaxation
saturates segment 1 before touching segment 2, so the concave envelope is tight
at every vertex. Indexing $\lambda$ by $q$ as well as $j$ is what lets $s_q$
multiply the coverage gain while both stay linear.

*Why no chronological-ordering constraints are needed*: nothing forces the
solver to fill segments in time order, but it does so anyway. The sub-problem
is a transportation problem with cost $m_j\phi_\tau$, where $m_j$ is decreasing
in $j$ and $\phi_\tau$ is non-increasing in $\tau$; by the rearrangement
inequality the north-west-corner (chronological) assignment is optimal. The LP
value therefore equals the chronological value that
`utility.CoverageAccumulator` computes on the realized side.

$\Delta_j^{\text{res}}$ is the width still **unclaimed** at the coverage
already realized before this horizon. Omitting the residual would let the MPC
re-earn the steep segment-1 credit at every re-plan.

Default $J=4$, a discretization of $g(\rho)=(1-e^{-3\rho})/(1-e^{-3})$:
widths $(.25,.25,.25,.25)$, slopes $(2.00,1.12,0.60,0.28)$.

`UTIL_SEG_STRIDE` indexes the $\lambda$ columns every $N$ horizon steps rather
than every step (freshness taken at the group's first step). Measured on the
default scenario: stride 1 → 33.1 s / utility 24.248; stride 3 → 14.0 s /
24.795; stride 5 → 15.1 s / 24.780; stride 10 → 12.8 s / 24.780, against a
10.1 s legacy baseline. Coarser indexing is both **faster and slightly
better** here — the smaller MILP solves more reliably inside HiGHS's default
effort — so exact per-step columns buy nothing. Default 5.

### Fairness: maximin, not Jain

Jain's index $(\sum U_k)^2/(n\sum U_k^2)$ is a ratio of quadratics and is not
MILP-representable. It is therefore **reported as a diagnostic only**. What is
optimized is a maximin floor: one continuous column $u_{\min}$ plus
$|\mathcal K|$ rows $u_{\min}\le\hat U_k$, with $+\omega_F|\mathcal K|u_{\min}$
in the objective.

* $\hat U_k$ must be **cumulative** (realized utility as a constant plus the
  in-horizon linear expression); the within-window gain alone would make every
  late-admitted task look starved and turn the term into noise.
* Normalizing by $\omega_Q s_{\max}\bar w_k$ makes the floor weight-relative:
  each task is measured against a fraction of *its own* achievable value, so
  the floor cannot be gamed by starving low-weight AOIs, nor does it perversely
  starve high-weight ones.
* $u_{\min}$ needs a **free lower bound** — $\hat U_k$ goes negative once the
  tardiness/cost penalties exceed the quality gain, and a floor pinned at 0
  would make the row infeasible.
* **The deciding argument is the oracle.** The maximin form is exactly
  LP-representable, so `oracle.py` can carry it and the gap table stays
  meaningful; Jain cannot be, so putting Jain in the score would permanently
  break the bound.

Measured effect (congested, `GS_RATE_BPS` = 1 Mbps): $\omega_F=0$ gives
$u_{\min}=0.6475$, Jain 0.9905, $\sum_k U_k=37.211$; $\omega_F=0.10$ gives
$u_{\min}=0.7024$, Jain 0.9948, $\sum_k U_k=36.687$. So the floor buys **+8.5%
for the worst-served task at a cost of 1.4% of aggregate utility**, and
saturates by $\omega_F=0.1$ — which is why that is the default.

### Energy: a measured negative result

$t_{\text{enc}}=12.34$ ms was measured at **both** $8\!\times\!8\!\times\!1$ and
$8\!\times\!8\!\times\!8$ ($\sigma\approx0.03$–$0.05$ ms), so encode time does
not vary with depth. Per image: encode $=30\,\text{W}\times12.34\,\text{ms}=0.370$ J;
downlink at $q{=}16 = 11{,}264\,\text{bits}\times2\times10^{-7}=2.25\times10^{-3}$ J.
**Encode dominates downlink by ~164× per image (~230× over a whole run's depth
mix), and encode is depth-independent** — so a literal Joule-denominated term
is nearly constant in $q$ and *cannot* drive depth choice. Depth selection in
this system is a **bandwidth** decision, not an energy one.

The objective therefore uses a normalized, unit-free `cost_coeff(q)` whose
*shape* is physical and whose *scale* is a stated policy weight
($\omega_E=0.05$, a tie-breaker); absolute Joules are reported in
`summary.txt` and `task_outcomes_*.csv` as accounting only. $P_{\text{enc}}=30$ W
(Jetson AGX Orin mid-band — the A6000 the timing came from is not a flight
part) and $P_{\text{tx}}=20$ W are **assumed** and labelled as such in
`config.py`.

### Keeping the offline bound valid

Rule: every term in the reported score must appear in the bound's objective,
relaxed in the optimistic direction.

| term | in the bound | valid because |
|---|---|---|
| $s_q$ | same table | exact |
| $\phi_k$, $T_k$ | evaluated at **window start** | most optimistic instant in the window |
| $\lambda$ coverage | same segments, grouped `ORACLE_SEG_AGG`× coarser than $y$ | coarser grouping only *loosens* which window credit lands in |
| $C_k$ | exact per-$(k,q)$ constant | a penalty must not be over-charged, and it isn't |
| $\omega_F|\mathcal K|u_{\min}$ | 1 column + $|\mathcal K|$ rows | adding a non-negative term to a max can only raise the optimum |

`oracle.report` now **asserts** `bound >= realized` for every scheduler and
prints a loud `!! BOUND VIOLATED` otherwise — one check that catches
essentially any sign or normalization mistake in the terms above. It found a
real one during this pass: the bound is handed an already-simulated task list
but re-plans every task from scratch, so `fair_row_coeffs` was double-counting
`delivered_utility` as the maximin constant, letting $u_{\min}$ reach ~2 and
pushing the LP bound (51.16) *above* the analytic ceiling (50.58, provisional
quality table). Fixed by passing `realized=0.0`; with the provisional table
the ordering was ceiling 50.58 > LP 46.22 > MILP dual 45.72 >= realized 44.56,
and with the real quality table it is ceiling 23.34 > LP 18.93 > MILP dual
18.83 >= realized 18.13 (the MILP incumbent search itself didn't converge
within the 120s budget against the real, steeper coefficients -- the dual
bound is still valid and is what the gap table uses).

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

## Two-MPC split: routing MPC + depth MPC

Dr. Liu: *"Hierarchical MPC -- one MPC to select the best path, another one
[for the rest]."* `hier.py`'s original split was admission/budget vs depth,
with **static routing at both levels**, so the routing/depth split was never
actually built. It now exists in two forms, and both are reported.

### Candidate path sets

Both need more than one path per $(k,\tau)$. Full Yen's $K$-shortest over a
1,160-node graph per $(k,\tau)$ is unaffordable, so
`routing.PredictiveRouter.build_multi` uses **edge-penalized re-Dijkstra**:
round $i$ re-solves the per-$(\tau, g)$ trees with every edge used by rounds
$1..i{-}1$ scaled by $(1+\texttt{ROUTE\_DIVERSITY\_PENALTY})$. One Dijkstra
call yields the tree for *all* sources, so $K$ rounds give $K$ candidates for
every task at once. These are **diversified** paths, not provably the $K$
shortest -- stated plainly; Yen's remains the right tool if exact
$K$-shortest for a single $(k,\tau)$ is ever needed.

Verified: $K{=}1$ is bit-identical to `build()`; at $K{=}3$, 88% of
$(k,\tau)$ pairs get $\ge 2$ distinct candidates, every hop is feasible in
`isl_ok`/`gsl_ok`, and delays are non-decreasing across rounds.

### `mpc-2level` — iterated peers (`twolevel.py`)

Routing MPC = min-cost multicommodity flow, continuous, so a pure LP:

$$\min \sum_{k,p,\tau} f_{k,p,\tau}\frac{\delta_{p,\tau}}{\Delta^{\text{delay}}_{\text{ref}}}
      + M\sum_{k,\tau}\text{sh}_{k,\tau}
\quad\text{s.t.}\quad
\sum_p f_{k,p,\tau}+\text{sh}_{k,\tau}=D_{k,\tau},\quad
\sum_{k}\sum_{p\ni e} f_{k,p,\tau}\le C_e\Delta t$$

The capacity rows are what make this a routing *optimizer*: a linear delay
objective without them would simply re-pick the shortest path, i.e. Dijkstra
in an expensive wrapper. Congestion enters endogenously through the capacity
duals.

The mix returns per-edge **shares** $\theta$, and the depth MILP's capacity
rows become $\sum_k\sum_q b_q\!\cdot\!8\big(\sum_{p\ni e}\theta_{k,p,\tau}\big)y_{k,q,\tau}\le C_e\Delta t$
— structurally the same rows, one coefficient change. That is what lets
multipath enter the depth problem **without adding any path variables to it**.

They exchange demand $\leftrightarrow$ mix and iterate with the same MSA
damping ($\eta = 1/(it{+}2)$) and keep-best-iterate the predictive router
already uses. Iteration 0 *is* the flat MPC, so `mpc-2level` can never plan
worse than `mpc`. Verified: `MPC2L_ITERS=1, MPC_ROUTE_NPATHS=1` reproduces
`mpc` bit-for-bit.

> **A correctness trap worth recording.** $D_{k,\tau}$ must be what a task
> *wants* to move, not what the depth MILP already conceded. Deriving it from
> the previous plan is circular: that plan was already feasible against the
> static single paths, so the routing LP faces no contention, finds the
> shortest path optimal for everything, and reproduces Dijkstra. The first
> implementation did exactly this, and `mpc-2level` came out **bit-for-bit
> equal to `mpc`** even on a fabric with 30% of links oversubscribed. The
> honest demand is what the encoder can supply and the task still owes
> (`_desired_bits`); with that fixed, 35% of $(k,\tau)$ decisions become
> genuinely multipath (mean 1.40 paths).

### `mpc-hier-route` — slow routing, fast depth (`hier.py`)

A `RouteCoordinatorMPC` runs once per macro-epoch (10 min) over a 60-slot
horizon in 10-slot macro-windows, on the topology at each window's **middle**
slot (a forecast, documented as one). It takes the existing
`CoordinatorMPC._lp_allocate` budgets $B_k$, solves the same flow LP in bits,
and freezes the top-`HIER_ROUTE_KEEP` paths per task for the epoch. The
existing fast depth MILP then solves inside those paths. `Directive` gains
`routes` and `route_delay`; empty dicts reproduce `mpc-hier` exactly
(verified bit-for-bit with `HIER_ROUTE_ON=False`).

> **A second scaling trap.** The flow coefficient is $\delta/\Delta_{\text{ref}}\sim O(1)$
> *per bit*, so the shortfall big-M must also be per-bit. An earlier version
> divided it by $10^9$ (bits $\to$ Gbit) without normalizing the flow term the
> same way, making shorting $\sim5\times10^5$ times **cheaper** than routing:
> the LP shorted everything, returned zero routes, and `mpc-hier-route`
> silently degenerated into `mpc-hier`. Both traps produce the same symptom --
> a routing coupling that looks like a clean "no difference" result -- which is
> why `summary.txt` now reports paths-per-decision and frozen-route survival
> rather than utility alone.

### Measured: freezing routes does not survive a LEO fabric

The fallback counter (frozen route infeasible $\to$ revert to the geometric
path) is the quantitative price of the hierarchical coupling:

| macro-epoch | window | utility | fallback rate |
|---|---|---|---|
| 20 slots (10 min) | 10 | 19.419 | 92.8% |
| 10 slots (5 min) | 5 | 19.459 | 94.4% |
| 5 slots (2.5 min) | 5 | 19.459 | 94.4% |
| 4 slots (2 min) | 2 | 18.895 | 90.6% |

Split by lookahead, a route frozen at epoch start survives:

| $\tau$ | lookahead | survives |
|---|---|---|
| 0 | 0 s | **14.4%** |
| 1 | 30 s | 11.7% |
| 5 | 150 s | 12.2% |
| 10 | 300 s | 3.3% |
| 19 | 570 s | 0.0% |

Even at the **executed** slot with zero lookahead the frozen route is usable
only 14.4% of the time, and shortening the epoch does not help. This is
structural, not a tuning failure: a concrete path is pinned to specific
satellites, and GSL contact windows here average 227-265 s, so the satellite
serving a given GBS turns over faster than any useful planning epoch. The
caveat: this is a negative result about freezing **concrete paths**. Freezing
a more abstract decision (a serving-GBS assignment, or a route *class*) might
survive; that is future work, not something measured here.

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
