# OEC Downlink Simulation — Topology Design

This document defines the network topology for evaluating RQ-VAE on-board image
compression under realistic LEO satellite downlink conditions.

Single source of truth for the **small scenario** (current): `small_scenario.py`.
Single source of truth for the **full Starlink scenario** (future): `topology_config.py`.

---

## In Plain Words

We simulate a small but complete LEO satellite network to evaluate how different
RQ-VAE compression depths affect end-to-end downlink performance.

The constellation is a **Walker 6/2/1** — **6 satellites** arranged in
**2 orbital planes of 3**, at **550 km altitude** and **53° inclination**.
Each satellite carries **inter-satellite laser links (ISLs)** connecting it to
its two ring neighbours in the same plane and to the nearest satellite in the
opposite plane. This forms a small mesh that can route data from any satellite
to any other, even when there is no direct ground link.

On the ground we place **2 receiving stations** — Tokyo and Sao Paulo — chosen
from the existing Hypatia top-100 city list for maximum geographic spread.
As the constellation orbits, satellites pass over each station for a **contact
window** of roughly 5–8 minutes. Outside those windows the satellite must relay
data through ISL hops to another satellite that is currently visible to the
station.

The design principle is unchanged: **all satellites are equal** — every one can
image a region, run the RQ-VAE encoder on-board (~12.34 ms), and relay traffic.

---

## 1. Design Principle: All Satellites Are Equal

There is **no special class of compute satellites**. Every satellite can:

1. **Image** — capture a 512×512 aerial scene
2. **Compute** — run the RQ-VAE encoder on-board (compression)
3. **Relay** — forward traffic over inter-satellite links

Which satellites are "doing computation" at any instant is simply whichever ones
are currently over an imaged region and downlinking.

---

## 2. Constellation

Walker 6/2/1 shell, propagated with pure circular-orbit mechanics in `small_scenario.py`.

| Parameter | Value |
|---|---|
| Orbital planes | 2 |
| Satellites per plane | 3 |
| **Total satellites** | **6** |
| Altitude | 550 km |
| Inclination | 53° |
| Orbital period | ~5,730 s (95.5 min) |
| ISL pattern | Intra-plane ring + nearest inter-plane |
| **Total ISLs** | **9** |

**Node numbering:** satellites 0–5; ground stations 6 (Tokyo) and 7 (Sao Paulo).

### ISL topology

9 links total:

| Type | Links |
|---|---|
| Intra-plane ring (plane 0) | 0↔1, 1↔2, 2↔0 |
| Intra-plane ring (plane 1) | 3↔4, 4↔5, 5↔3 |
| Inter-plane nearest | 0↔3, 1↔4, 2↔5 |

**Intra-plane ISL distances** are constant at **11,988 km** (3 satellites equally
spaced 120° apart on a 6,921 km radius orbit — the chord length does not change).

**Inter-plane ISL distances** are time-varying: **7,214 – 13,200 km**
(±2,061 km standard deviation over the simulation). This is the primary source
of time-varying link availability in the small scenario.

---

## 3. Ground Stations — 2, from the existing Hypatia top-100 list

| Station | Node ID | Lat | Lon | Top-100 index |
|---|---|---|---|---|
| Tokyo | 6 | 35.69°N | 139.69°E | 0 |
| Sao Paulo | 7 | 23.55°S | 46.64°W | 3 |

No regeneration of satellite state is needed — both cities are already part of
the Hypatia top-100 ground station file the full Starlink state was built with.

---

## 4. Simulation Results (small_scenario.py, 6,000 s, 10 s steps)

### Contact Windows

| Station | Windows | Total coverage | Coverage % | Avg window | Serving sats (in order) | Handoffs |
|---|---|---|---|---|---|---|
| Tokyo | 4 | 1,390 s | 23.2% | 348 s | 1 → 0 → 2 → 1 | 3 |
| Sao Paulo | 3 | 1,010 s | 16.8% | 337 s | 2 → 1 → 0 | 2 |

**Window detail:**

Tokyo:
- Sat 1:  t = 0 – 380 s  (380 s)
- Sat 0:  t = 1,890 – 2,370 s  (480 s)
- Sat 2:  t = 3,890 – 4,340 s  (450 s)
- Sat 1:  t = 5,920 – 6,000 s  (80 s, orbit repeat)

Sao Paulo:
- Sat 2:  t = 1,130 – 1,280 s  (150 s)
- Sat 1:  t = 3,010 – 3,400 s  (390 s)
- Sat 0:  t = 4,960 – 5,430 s  (470 s)

### End-to-End Paths from Sat 0 (imaging satellite → ground station)

Sat 0 is the designated imaging/compute satellite. The routing graph uses ISLs
to reach whichever satellite currently has a GSL to the ground station.

| Station | Reachable steps | Prop delay avg | Prop delay min | Prop delay max | Avg hops | Paths observed |
|---|---|---|---|---|---|---|
| Tokyo | 143 / 601 (23.8%) | 30.11 ms | 1.86 ms | 45.98 ms | 1.66 | 0→6, 0→1→6, 0→2→6 |
| Sao Paulo | 104 / 601 (17.3%) | 26.04 ms | 2.38 ms | 46.03 ms | 1.54 | 0→7, 0→1→7, 0→2→7 |

Three distinct paths exist to each station (direct 1-hop GSL, or 2-hop ISL relay
via sat 1 or sat 2). Which path is active depends on which satellite is currently
visible to the ground station — this is the time-varying routing behaviour that a
future MPC scheduler will exploit.

### ISL Distance Variation (time-varying link availability)

| Link | Type | Min (km) | Max (km) | Avg (km) | Std dev |
|---|---|---|---|---|---|
| 0↔1 | Intra-plane | 11,988 | 11,988 | 11,988 | 0 |
| 1↔2 | Intra-plane | 11,988 | 11,988 | 11,988 | 0 |
| 2↔0 | Intra-plane | 11,988 | 11,988 | 11,988 | 0 |
| 3↔4 | Intra-plane | 11,988 | 11,988 | 11,988 | 0 |
| 4↔5 | Intra-plane | 11,988 | 11,988 | 11,988 | 0 |
| 5↔3 | Intra-plane | 11,988 | 11,988 | 11,988 | 0 |
| **0↔3** | **Inter-plane** | **7,214** | **13,200** | **10,466** | **2,061** |
| **1↔4** | **Inter-plane** | **7,214** | **13,200** | **10,526** | **2,100** |
| **2↔5** | **Inter-plane** | **7,214** | **13,200** | **10,290** | **2,146** |

---

## 5. Link Parameters

| Link | Rate | Queue |
|---|---|---|
| Ground-to-Satellite Link (GSL) | 100 Mbps | 1,000 pkts |
| Inter-Satellite Link (ISL) | 10 Gbps | 1,000 pkts |

---

## 6. Scaling Plan

| Phase | Constellation | Ground stations | Goal |
|---|---|---|---|
| **A (current)** | Walker 6/2/1 (6 sats) | 2 (Tokyo, Sao Paulo) | Complete small scenario: contact windows, time-varying ISLs, end-to-end paths |
| B | Walker 24/3/1 or Starlink-550 subset | 5 (from top-100) | Wider coverage, more handoffs, congestion |
| C | Full Starlink-550 (1,584 sats) | 5–10 | Large-scale ISL contention, scheduling |

All phases use existing Hypatia top-100 ground station data — no regeneration needed.

---

## 7. Connection to Future Scheduling (MPC)

The simulation produces three outputs that are direct inputs to a future
rolling-horizon MPC scheduler:

- **`contact_windows.csv`** — when each satellite is visible to each station:
  the scheduler's prediction horizon for GSL availability
- **`link_availability.csv`** — ISL distance and state at every time step:
  the scheduler's link capacity and latency model
- **`path_info.csv`** — which routing path is available and at what delay:
  the scheduler's state for compression depth and transmission timing decisions

The MPC problem: given predicted future contact windows, queue states, and link
capacity, select (a) compression depth, (b) transmission start time, and
(c) routing path to minimise end-to-end latency and drop rate for each image.

---

## 8. How to Run

```bash
cd hypatia_sim
python3 small_scenario.py
# outputs: small_scenario/contact_windows.csv
#          small_scenario/link_availability.csv
#          small_scenario/path_info.csv
#          small_scenario/summary.txt
```

No Hypatia state generation or ns-3 build required.

---

## 9. Open Items

- Integrate compressed payload sizes into path scheduling (payload 88 B depth-1
  → 1,408 B depth-16; transmission time on 100 Mbps GSL = 7 µs → 113 µs)
- Extend to Walker 24/3/1 or Starlink-550 subset (Phase B)
- Begin MPC scheduler design using contact_windows.csv as prediction horizon
