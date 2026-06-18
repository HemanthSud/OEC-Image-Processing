# RQ-VAE Satellite Image Compression for Orbital Edge Computing

Orbital Edge Computing (OEC) research project on satellite image compression using Residual Quantized Variational Autoencoders (RQ-VAE) and N-gram Arithmetic Coding (NAC). The project studies how different compression levels affect satellite and aerial image reconstruction quality, and evaluates compressed payloads under realistic LEO satellite network constraints using the Hypatia simulator.

## Research Context

| | |
|---|---|
| Lab | NICE Lab, North Carolina State University |
| Researcher | Hemanth Sudhaharan |
| Graduate Mentor | Xuanhao Luo |
| Faculty Advisor | Dr. Yuchen Liu |
| Timeline | February 2026 – present |

## Project Goal

Evaluate how different RQ-VAE compression settings affect:

1. Reconstruction quality of satellite and aerial images
2. Compression ratio and latent-code efficiency
3. End-to-end viability under real LEO orbital link constraints (latency, drop rate, bandwidth)

---

## Hardware and Datasets

| | |
|---|---|
| Server | NCSU cluster `eb3-2402-grd04.csc.ncsu.edu` |
| GPUs | 2 × NVIDIA RTX A6000 (48 GB each) |
| Primary dataset | EuroSAT RGB — 27,000 images, 10 land-use classes, `64×64` |
| Secondary dataset | FLAIR-1 — 77,412 aerial GeoTIFF patches, `512×512` RGB |

---

## What Was Completed

### Phase 1–3: Setup

- RQ-VAE training pipeline for EuroSAT with fixed `80/10/10` splits
- Baseline ResNet-18 classifier: **97.89% test accuracy** on original images
- Moved project to NCSU cluster with CUDA environment

### Phase 4: EuroSAT Full Sweep

Trained 9 RQ-VAE configurations across spatial sizes `8×8`, `4×4`, `2×2` and depths `1`, `4`, `8` with reconstruction + LPIPS + GAN loss.

### Phase 5: EuroSAT Quantitative Evaluation

Computed PSNR, SSIM, LPIPS, and FID on test split for all 9 models.

### Phase 6: Follow-Up Experiments

Added depths `2` and `3` for all spatial sizes, plus `1024`-entry codebook variants for `4×4`. Results show `8×8` follows the expected trend (deeper = better), while `4×4` degrades with depth.

### Phase 7: Cleanup and GitHub Publishing

Cleaned code, organized configs and metrics, published to GitHub.

### Phase 8: FLAIR-1 Extension

- FLAIR-1 dataset loader via official CSV files and `rasterio` for GeoTIFF
- FLAIR `8×8×D` configs for depths `1`, `2`, `4`, `8`, `16`
- PNG pre-conversion pipeline for 47,587 training images (faster loading)
- Extended `evaluate_metrics.py` and NAC export for FLAIR

### Phase 9: FLAIR-1 Full Subset Training

Trained depth-1 and depth-8 models on 50% of FLAIR-1 (23,800 train / 7,050 val) for 120+ epochs using A6000 GPUs. Val metrics computed and saved.

### Phase 10: Hypatia OEC Simulator Integration

Integrated the Hypatia LEO satellite network simulator to evaluate compressed payloads under real orbital downlink conditions:

- Profiled RQ-VAE encoder compute delay at the satellite (RQ) node: **~12.34 ms** for both depth-1 and depth-8
- Generated Starlink 550km constellation state (1,584 satellites, ISL grid, 200s simulation)
- UDP burst schedule: 500 bursts (100 images × 5 models) with compute delay offsets
- Simulation routes each burst from RQ node through ISLs to AC node (New York ground station, closest top-100 city to NCSU Raleigh)
- Analysis covers: network latency, compute delay, end-to-end latency, drop rate, ILS utilization, communication topology

---

## Results

### EuroSAT RQ-VAE — Full Sweep

> Compression ratio formula: `original bits / (codes × bits_per_code)`
> Original image: `64 × 64 × 3 × 8 = 98,304 bits` | Codebook 2048 → 11 bits/code

| Spatial | Depth | Codebook | Codes | PSNR (dB) | SSIM | LPIPS | FID | Compression |
|---------|-------|----------|-------|-----------|------|-------|-----|-------------|
| 8×8 | 1 | 2048 | 64 | 29.93 | 0.7861 | 0.1665 | 19.10 | 139.6:1 |
| 8×8 | 2 | 2048 | 128 | 31.30 | 0.8273 | 0.1259 | 14.24 | 69.8:1 |
| 8×8 | 3 | 2048 | 192 | 32.50 | 0.8546 | 0.1037 | 13.43 | 46.5:1 |
| 8×8 | 4 | 2048 | 256 | 33.28 | 0.8705 | 0.0946 | 14.43 | 34.9:1 |
| **8×8** | **8** | **2048** | **512** | **36.27** | **0.9362** | **0.0513** | **9.40** | **17.5:1** |
| 4×4 | 1 | 2048 | 16 | 28.41 | 0.7128 | 0.2294 | 25.46 | 558.5:1 |
| 4×4 | 2 | 2048 | 32 | 27.05 | 0.6670 | 0.2668 | 26.84 | 279.3:1 |
| 4×4 | 3 | 2048 | 48 | 27.34 | 0.6842 | 0.2545 | 24.81 | 186.2:1 |
| 4×4 | 4 | 2048 | 64 | 27.93 | 0.7107 | 0.2573 | 34.90 | 139.6:1 |
| 4×4 | 8 | 2048 | 128 | 23.34 | 0.5634 | 0.3802 | 102.80 | 69.8:1 |
| 4×4 | 2 | 1024 | 32 | 26.84 | 0.6591 | 0.2738 | 29.32 | 307.2:1 |
| 4×4 | 3 | 1024 | 48 | 27.19 | 0.6755 | 0.2683 | 28.56 | 204.8:1 |
| 2×2 | 1 | 2048 | 4 | 24.68 | 0.5864 | 0.3477 | 67.20 | 2234.2:1 |
| 2×2 | 2 | 2048 | 8 | 23.43 | 0.5570 | 0.3778 | 95.16 | 1117.1:1 |
| 2×2 | 3 | 2048 | 12 | 23.92 | 0.5671 | 0.3627 | 76.07 | 744.7:1 |
| 2×2 | 4 | 2048 | 16 | 24.68 | 0.5799 | 0.3474 | 70.20 | 558.5:1 |
| 2×2 | 8 | 2048 | 32 | 25.72 | 0.6497 | 0.2985 | 42.35 | 279.3:1 |

### FLAIR-1 RQ-VAE — Subset Training Results (val split)

> Original image: `512 × 512 × 3 × 8 = 6,291,456 bits` | Trained on 50% of FLAIR-1 (23,800 train images, 120+ epochs)

| Model | Code Shape | Codes | Bytes | PSNR (dB) | SSIM | LPIPS | FID | Compression |
|-------|------------|-------|-------|-----------|------|-------|-----|-------------|
| flair-rqvae-8x8x1 | 8×8×1 | 64 | 88 | 20.63 | 0.4560 | 0.4595 | 71.33 | 8936.7:1 |
| flair-rqvae-8x8x8 | 8×8×8 | 512 | 704 | 21.02 | 0.4643 | 0.4779 | 73.19 | 1117.1:1 |

Depth-8 improves PSNR (+0.39 dB) and SSIM (+0.0083) over depth-1. Depth-1 has slightly better LPIPS and FID, likely because the lower-capacity model learns smoother (less detailed) reconstructions that score better on perceptual distance metrics. The expected trend — deeper quantization yields better reconstruction — holds for PSNR and SSIM.

Earlier preliminary subset runs (5,000 train / 1,000 val, 50 epochs) are recorded below for reference:

| Model | Code Shape | PSNR (dB) | SSIM | LPIPS | FID | Compression |
|-------|------------|-----------|------|-------|-----|-------------|
| flair-subset50-rqvae-8x8x1 | 8×8×1 | 18.63 | 0.3253 | 0.5700 | 267.05 | 8936.7:1 |
| flair-subset50-rqvae-8x8x4 | 8×8×4 | 19.70 | 0.3637 | 0.5379 | 117.59 | 2234.2:1 |
| flair-subset50-rqvae-8x8x8 | 8×8×8 | 19.07 | 0.3694 | 0.5771 | 197.02 | 1117.1:1 |

### RQ Node Compute Delay Profile

Measured on A6000 GPU (100 runs, 512×512 input):

| Model | Encode Time | Std |
|-------|-------------|-----|
| flair-rqvae-8x8x1 | 12.34 ms | 0.054 ms |
| flair-rqvae-8x8x8 | 12.34 ms | 0.029 ms |

Compute delay is nearly identical across depths — the encoder CNN dominates, not the quantization step.

---

## Key Findings

**Best EuroSAT quality:** `8×8×8` — PSNR 36.27 dB, SSIM 0.9362, FID 9.40

**Best EuroSAT trade-off:** `8×8×3` (32.50 dB at 46.5:1) and `8×8×4` (33.28 dB at 34.9:1)

**Most compressed:** `2×2×1` at 2234:1 — viable for extreme bandwidth constraints

**Unexpected result:** The `4×4` family degraded with depth (`4×4×1`: 28.41 dB → `4×4×8`: 23.34 dB), suggesting the hyperparameters are not well-matched to smaller latent grids on 64×64 images.

**FLAIR-1 key finding:** Depth-8 outperforms depth-1 on PSNR and SSIM as expected, but both models achieve much lower compression ratios at 512×512 resolution. The 88-byte payload (depth-1) versus 704-byte payload (depth-8) represents a meaningful bandwidth difference under satellite link constraints.

---

## Repository Structure

```text
.
├── hypatia_sim/
│   ├── profile_compute_delay.py    # RQ node encode time profiling
│   ├── generate_sim_inputs.py      # Hypatia UDP burst schedule + config
│   ├── analyse_results.py          # Latency, drop rate, ILS, topology analysis
│   └── run/                        # Simulation inputs/outputs
├── nac/
│   ├── arithmetic_coding.py
│   ├── nac_eurosat.py
│   └── ngram.py
├── results/
│   ├── metrics.log
│   ├── summary.json
│   ├── followup_metrics.log
│   └── followup_2x2_metrics.log
├── rq-vae/
│   ├── configs/eurosat/stage1/
│   ├── configs/flair/stage1/
│   ├── evaluate_metrics.py
│   ├── run_flair_8x8_sweep.sh
│   ├── train_eurosat.py
│   └── rqvae/
│       └── img_datasets/
│           ├── eurosat.py
│           └── flair.py
├── flair_val_metrics.txt
├── eurosat_split_indices.pt
└── split_indices.py
```

---

## Reproducing the Project

### 1. Clone

```bash
git clone https://github.com/HemanthSud/OEC-Image-Processing.git
cd OEC-Image-Processing
```

### 2. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install omegaconf einops lpips tensorboard scikit-image tqdm matplotlib pillow numpy scipy pyyaml
pip install "rasterio<1.5"  # for FLAIR-1 GeoTIFF loading
```

### 3. EuroSAT Training

Download EuroSAT RGB into `EuroSAT_RGB/` with one folder per class, then:

```bash
python split_indices.py
cd rq-vae
python train_eurosat.py -m configs/eurosat/stage1/eurosat-rqvae-8x8x4.yaml -o output/eurosat-rqvae-8x8x4 --epochs 150
```

### 4. FLAIR-1 Training

Place FLAIR-1 data with CSV files at the repo root, then:

```bash
cd rq-vae
MAX_TRAIN_SAMPLES=23800 MAX_VAL_SAMPLES=7050 MAX_TEST_SAMPLES=7850 BATCH_SIZE=16 ./run_flair_8x8_sweep.sh
```

Run two GPU windows in parallel:

```bash
# Window 0 — GPU 0
CUDA_VISIBLE_DEVICES=0 DEPTHS="1 2 4" ./run_flair_8x8_sweep.sh

# Window 1 — GPU 1
CUDA_VISIBLE_DEVICES=1 DEPTHS="8 16" ./run_flair_8x8_sweep.sh
```

### 5. Evaluate Metrics

```bash
python evaluate_metrics.py --split val --output-dirs output/flair-rqvae-8x8x1 output/flair-rqvae-8x8x8
```

### 6. Hypatia Simulation

```bash
# Profile compute delay
cd hypatia_sim
python3 profile_compute_delay.py

# Generate satellite state (Hypatia must be cloned)
cd hypatia/paper/satellite_networks_state
python3 main_starlink_550.py 200 100 isls_plus_grid ground_stations_top_100 algorithm_free_one_only_over_isls 4

# Generate sim inputs
cd hypatia_sim
python3 generate_sim_inputs.py

# Build and run ns-3
cd hypatia/ns3-sat-sim/simulator
./waf configure --build-profile=optimized --enable-mpi && ./waf
./waf --run="main_satnet --run_dir='/path/to/hypatia_sim/run'"

# Analyse results
cd hypatia_sim
python3 analyse_results.py
```

---

## Pending

1. Complete ns-3 build and run end-to-end Hypatia simulation
2. Complete FLAIR-1 sweep for depths 2, 4, 16
3. Evaluate classification accuracy on reconstructed FLAIR images
4. Run NAC entropy coding on FLAIR exported codes

---

## References

- RQ-VAE: "Residual Quantized Variational Autoencoders", CVPR 2023
- EuroSAT: https://github.com/phelber/eurosat
- FLAIR-1: https://github.com/IGNF/FLAIR-1
- Hypatia simulator: https://github.com/snkas/hypatia

## Contact

Hemanth Sudhaharan — NICE Lab, North Carolina State University — hsudhah@ncsu.edu
