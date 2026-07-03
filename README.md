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

## Satellite Network Simulation

### Small Scenario (current)

A small but complete LEO scenario (`hypatia_sim/small_scenario.py`) — no ns-3 or Hypatia state generation required, runs in under 1 second.

**Constellation:** Walker 6/2/1 — 6 satellites, 2 orbital planes of 3, 550 km, 53°, ~95.5 min orbital period.

**ISLs (9 total):** Intra-plane rings 0↔1↔2↔0 and 3↔4↔5↔3, inter-plane nearest-neighbour 0↔3, 1↔4, 2↔5.

**Ground stations:** Tokyo (node 6) and Sao Paulo (node 7), from the Hypatia top-100 city list.

**Simulation window:** 6,000 s at 10 s steps, minimum elevation 10°.

#### Contact Windows

| Station | Windows | Coverage | Avg window | Serving sats | Handoffs |
|---|---|---|---|---|---|
| Tokyo | 4 | 1,390 s (23.2%) | 348 s | 1→0→2→1 | 3 |
| Sao Paulo | 3 | 1,010 s (16.8%) | 337 s | 2→1→0 | 2 |

#### End-to-End Paths (from imaging sat 0)

| Station | Reachable | Avg delay | Min | Max | Avg hops | Paths |
|---|---|---|---|---|---|---|
| Tokyo | 23.8% of steps | 30.11 ms | 1.86 ms | 45.98 ms | 1.66 | 0→6, 0→1→6, 0→2→6 |
| Sao Paulo | 17.3% of steps | 26.04 ms | 2.38 ms | 46.03 ms | 1.54 | 0→7, 0→1→7, 0→2→7 |

#### ISL Distance Variation

Intra-plane ISLs are constant at 11,988 km. Inter-plane ISLs (0↔3, 1↔4, 2↔5) vary from 7,214 to 13,200 km (±2,100 km std dev) as the planes drift — the primary source of time-varying link availability.

Outputs: `contact_windows.csv`, `link_availability.csv`, `path_info.csv` — direct inputs to a future MPC scheduler.

---

## Repository Structure

```
.
├── hypatia_sim/
│   ├── small_scenario.py           # Small complete satellite scenario (current)
│   ├── topology_config.py          # Full Starlink-550 topology config
│   ├── generate_sim_inputs.py      # UDP burst schedule + ns-3 config
│   ├── extract_topology.py         # ISL degree, serving sats, handoffs, windows
│   ├── analyse_results.py          # Latency, drop rate, ISL utilization
│   ├── profile_compute_delay.py    # On-board encode time profiling
│   ├── TOPOLOGY.md                 # Topology design doc
│   └── small_scenario/             # Simulation outputs
├── rq-vae/
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
python3 small_scenario.py
```

---

## Pending

1. Complete FLAIR-1 sweep for depths 2 and 4
2. Run ns-3 end-to-end simulation (full Starlink scenario, ns-3 build unblocked)
3. Evaluate classification accuracy on reconstructed images
4. Run NAC entropy coding on exported FLAIR codes
5. MPC / rolling-horizon scheduler design for compression depth and transmission timing

---

## References

- RQ-VAE: "Residual Quantized Variational Autoencoders", CVPR 2023
- EuroSAT: https://github.com/phelber/eurosat
- FLAIR-1: https://github.com/IGNF/FLAIR-1
- Hypatia: https://github.com/snkas/hypatia

## Contact

Hemanth Sudhaharan — NICE Lab, NC State University — hsudhah@ncsu.edu
