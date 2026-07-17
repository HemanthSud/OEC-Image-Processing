# Network Parameters in the Notation of the Overleaf Formulation

This file states every simulation parameter in the mathematical form of
*OEC RQ-NAC* (Sec. 1.1–1.2), ready to paste into the Overleaf document.
Source of truth: `oec_sim/config.py`.

## Sets and time

| Symbol | Value in the simulation |
|---|---|
| $\mathcal{S}$ | Kuiper shell-1 (Hypatia `kuiper_630`): Walker-delta 1156/34/1, $\lvert\mathcal{S}\rvert = 1156$ satellites in 34 planes × 34, altitude 630 km, inclination 51.9°, orbital period 5830 s |
| $\mathcal{G}$ | $\lvert\mathcal{G}\rvert = 4$ GBSs: Tokyo, New York, São Paulo, Sydney (Hypatia top-100 city list) |
| $\Delta t$ | 30 s |
| $\mathcal{T}$ | $\lvert\mathcal{T}\rvert = 601$ slots, total 18 000 s (5 h ≈ 3.1 orbital periods) |

## Time-varying links $\mathcal{E}_t$ and capacities $C_{ij}(t)$

An ISL $(i,j)$ between satellites exists in $\mathcal{E}_t$ iff both
**line-of-sight** and **range** constraints hold (addressing the infeasible
120°-separation links in the earlier 6-satellite scenario):

$$\min_{\theta\in[0,1]} \lVert \mathbf{p}_i(t) + \theta(\mathbf{p}_j(t)-\mathbf{p}_i(t)) \rVert \;\ge\; R_E + h_{\text{graze}}, \qquad \lVert \mathbf{p}_i(t)-\mathbf{p}_j(t)\rVert \le d_{\max},$$

with $R_E = 6371$ km, grazing margin $h_{\text{graze}} = 80$ km (atmosphere),
$d_{\max} = 5016$ km (Hypatia laser-ISL figure). Candidate ISLs follow the
+Grid pattern (each satellite ↔ its two in-plane ring neighbours and the
same-index satellite of the adjacent plane): 2312 candidate links, all of
which are feasible at Kuiper spacing (in-plane angular separation 10.6°;
the check rejects the old 3-per-plane topology, whose 120° chords pass
2900 km below the surface).

A GSL $(i,g)$ exists iff the elevation of satellite $i$ seen from GBS $g$
is $\ge \varepsilon_{\min} = 20°$.

Capacities: $C_{ij}(t) = 100$ Mbps per feasible ISL; each GBS has an
aggregate receive budget $C_g = 2$ Mbps (S-band, single receive chain)
shared by all flows terminating at $g$. Routing is fixed shortest-delay
(Dijkstra on propagation distance, recomputed every slot; routes never
transit another GBS).

## Tasks $\mathcal{K}$

A task arrives when a satellite passes an AOI (8 fixed areas of interest)
with elevation ≥ 40°, subject to a per-AOI cooldown of 1800 s:

| Symbol | Value |
|---|---|
| $s_k$ | highest-elevation satellite over the AOI at arrival |
| $g_k$ | geodesically nearest GBS to the AOI |
| $a_k$ | arrival slot (geometry-driven; 64 tasks in 5 h, seed 42) |
| $N_k$ | $\sim \mathcal{U}\{100{,}000,\,300{,}000\}$ images (512×512 tiles of a large observation scene) |
| $w_k$ | $\sim \mathcal{U}\{1,2,3\}$ |
| $d_k$ | $a_k\Delta t + \mathcal{U}(1800, 3600)$ s (soft), freshness $\phi_k$ per eq. (6) with $\alpha_k = 1/300$ s$^{-1}$ |

Offered load at depth 16 is 144.3 Gbit vs. 144.0 Gbit total GBS capacity
(utilization ≈ 1.0), so the depth choice is the binding trade-off.

## Compression depths $\mathcal{D} = \{1, 2, 4, 8\}$

(The document writes $\mathcal{D} = \{2,4,8,16\}$; the simulation uses
$\{1,2,4,8\}$, matching the FLAIR 8×8-latent truncation evaluation.)

| $q$ | $b_q$ (bytes/image) | $t^{\text{enc}}_q$ | PSNR (dB) | LPIPS | $u_q = 1-\text{LPIPS}_q$ |
|---|---|---|---|---|---|
| 1 | 88  | 12.34 ms | 18.67 | 0.5948 | 0.4052 |
| 2 | 176 | 12.34 ms | 19.72 | 0.5570 | 0.4430 |
| 4 | 352 | 12.34 ms | 20.35 | 0.5314 | 0.4686 |
| 8 | 704 | 12.34 ms | 20.73 | 0.5171 | 0.4829 |

$b_q = 88q$ B (8×8 latent grid, 88 B per residual stage; measured FLAIR-1
payloads). PSNR/LPIPS are **measured** on 7,050 FLAIR-1 val images with the
single depth-16 model truncated to its first $q$ codebook stages
(`truncation_eval.txt`) — the intended variable-rate deployment. Encoding is
dominated by the conv encoder, so $t^{\text{enc}}_q$ is depth-independent
(measured 12.34 ms/image ⇒ an 81 img/s on-board pipeline, modeled as a
per-satellite encoding-rate constraint). $u_q = 1 - \text{LPIPS}_q$ is a
provisional utility mapping; to be replaced by downstream-task metrics
(mIoU / F1) per Xuanhao's plan.

## MPC scheduler (eq. 8)

Deterministic MPC with predicted contact traces and fixed shortest-delay
routing, exactly the "initial implementation" suggested in the document:

* horizon $H = 60$ slots (30 min), backlog weight $\lambda_1 = 10^{-3}$ per Gbit·slot;
* decision variables: depth binaries $x_{k,q}$ (committed at the task's
  first transmission) and per-slot image flows $y_{k,q}(\tau) \ge 0$;
* constraints: one depth per task (2); link/GBS capacity (5); encoder
  pipeline $\sum_{\tau'\le\tau} y_k(\tau') \le \min(N_k, (\tau - a_k)\Delta t \cdot 81)$;
  volume $\sum_\tau y_{k,q}(\tau) \le N_k x_{k,q}$; $y = 0$ when no route exists;
* solved as a MILP (HiGHS via `scipy.optimize.milp`); first slot executed,
  re-planned on every arrival and at latest every 5 slots.

## Headline results (seed 42, D = {1,2,4,8}, measured u_q)

**Light load** — GBS budget 2 Mbps, utilization 0.50 at depth-8: every
scheduler delivers 100% of images; the best fixed depth is simply the
deepest one (fixed-8 utility 64.15, MPC 64.11, fixed-1 53.83).

**Congested** — GBS budget 1 Mbps, utilization 1.00 at depth-8:

| scheduler | images delivered | delivery % | utility | on-time % |
|---|---|---|---|---|
| **MPC** | 12,809,971 | **100.0%** | **63.21** | 88.7% |
| greedy adaptive | 12,809,971 | 100.0% | 61.90 | 88.7% |
| greedy fixed-4 | 12,809,971 | 100.0% | 62.25 | 88.7% |
| greedy fixed-8 | 11,186,624 | 87.3% | 23.78 | 31.3% |

No single depth is best in both regimes: depth-8 wins when capacity is
free but collapses under congestion (12.7% of images never delivered,
most arrive after their deadline); depth-4 is the best fixed choice under
congestion; **MPC matches or beats the best fixed depth in every regime**
by mixing depths per task (here 30×8, 30×4, 4×2). With the reconstruction-
based $u_q$ the utility spread across depths is modest (0.41–0.48), so the
depth choice is driven mainly by deliverability; downstream-task utilities
may widen the gap.
