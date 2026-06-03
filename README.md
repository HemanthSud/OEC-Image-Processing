# RQ-VAE Satellite Image Compression for Orbital Edge Computing

This repository contains an Orbital Edge Computing (OEC) research project on satellite image compression using Residual Quantized Variational Autoencoders (RQ-VAE) and N-gram Arithmetic Coding (NAC).

The project studies how different compression levels affect satellite and aerial image reconstruction quality. It started with EuroSAT RGB and now includes an added FLAIR-1 path for native `512x512` aerial imagery.

## Project Motivation

Satellites must transmit imagery to Earth under strict bandwidth limits. For orbital edge computing, compression is not only about reducing size, but also about preserving the information needed for downstream tasks such as land-use classification.

This project explores that trade-off by training multiple RQ-VAE compression models on EuroSAT and FLAIR-1 and comparing reconstruction quality across latent sizes and quantization depths.

## Research Context

This work was carried out in NICE Lab at North Carolina State University.

- Researcher: Hemanth Sudhaharan
- Graduate mentor: Xuanhao Luo
- Faculty advisor: Dr. Yuchen Liu
- Timeline: February 2026 to present

## Project Goal

Evaluate how different compression settings affect:

1. Reconstruction quality of satellite images.
2. Compression ratio and latent-code efficiency.
3. Practical usefulness for downstream image classification.

## Hardware and Dataset

- Server: NCSU cluster (`eb3-2402-grd04.csc.ncsu.edu`)
- GPUs: 2 x NVIDIA RTX A6000 (48 GB each)
- Training time: about 34 hours for the initial 9-model sweep using dual-GPU parallel runs
- Primary dataset: EuroSAT RGB
- EuroSAT size: 27,000 images across 10 land-use classes
- EuroSAT image size: `64x64` RGB
- Added dataset: FLAIR-1 RGB aerial patches
- FLAIR-1 image size: native `512x512` RGB patches loaded from official CSVs

## What Was Completed

### Phase 1: Initial Setup

1. Set up the RQ-VAE training pipeline for EuroSAT.
2. Created fixed `80/10/10` train, validation, and test splits.
3. Trained a baseline ResNet-18 classifier on original images.

Session notes recorded a baseline classifier test accuracy of `97.89%` on original images.

### Phase 2: Initial Experiments

1. Trained initial RQ-VAE configurations.
2. Uploaded early outputs to Google Drive.
3. Incorporated feedback to expand the study with smaller latents and quantitative metrics.

### Phase 3: Server Setup

1. Moved the project to the NCSU cluster.
2. Uploaded code and dataset.
3. Set up the PyTorch and CUDA environment.

### Phase 4: Full RQ-VAE Model Sweep

Trained 9 EuroSAT RQ-VAE configurations:

- Spatial sizes: `8x8`, `4x4`, `2x2`
- Quantization depths: `1`, `4`, `8`
- Total models: `9`

The architecture was adapted for small `64x64` inputs and trained with:

- reconstruction loss
- latent quantization loss
- LPIPS perceptual loss
- GAN loss

### Phase 5: Quantitative Evaluation

Computed the following reconstruction metrics for the initial 9 trained models:

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index Measure)
- LPIPS (Learned Perceptual Image Patch Similarity)
- FID (Frechet Inception Distance)

The results were saved in the [`results/`](results/) directory.

### Phase 6: Follow-Up Experiments

After feedback, additional models were trained to check the missing intermediate depths and codebook-size effects:

- Depth `2` and `3` for `8x8`, `4x4`, and `2x2` latent grids
- `4x4x2` and `4x4x3` with a smaller `1024`-entry codebook

These runs were added to check whether reconstruction quality improves smoothly with quantization depth and whether a smaller codebook helps the smaller latent grids.

Metrics were computed for all follow-up runs and included in the full results table below.

Follow-up metrics were saved in:

- `results/followup_metrics.log`
- `results/followup_2x2_metrics.log`

### Phase 7: Cleanup and GitHub Publishing

1. Removed credentials and local-only clutter.
2. Cleaned code and project files.
3. Organized configs, metrics, and outputs.
4. Prepared the project for GitHub publishing.

### Phase 8: FLAIR-1 Extension

1. Added a FLAIR-1 dataset loader for official CSV files and geospatial TIFF images through `rasterio`.
2. Added native `512x512` FLAIR transforms and dataset split handling alongside the existing EuroSAT path.
3. Added FLAIR `8x8xD` RQ-VAE configs for depths `1`, `2`, `3`, `4`, and `8`.
4. Added `run_flair_8x8_sweep.sh` to run the FLAIR depth sweep with environment-variable overrides for epochs, batch size, learning rate, codebook size, and sample limits.
5. Extended metric evaluation and NAC code export so non-EuroSAT datasets can be evaluated with the same pipeline.

## Model Configurations and Results

The compression ratios below use the corrected formula from Xuanhao's feedback:

```text
compression ratio = original image bits / (number of codes x bits per code)
original image bits = 64 x 64 x 3 x 8 = 98,304 bits
bits per code = log2(codebook size)
```

For the default `2048`-entry codebook, each code requires `11` bits. For the `1024`-entry follow-up runs, each code requires `10` bits. These are RQ-code compression ratios before NAC lossless entropy coding.

| Spatial | Depth | Codebook | Codes/Image | Bits/Code | PSNR (dB) | SSIM | LPIPS | FID | RQ Code Compression |
|---------|-------|----------|-------------|-----------|-----------|------|-------|-----|---------------------|
| 8x8 | 1 | 2048 | 64 | 11 | 29.93 | 0.7861 | 0.1665 | 19.10 | 139.6:1 |
| 8x8 | 2 | 2048 | 128 | 11 | 31.30 | 0.8273 | 0.1259 | 14.24 | 69.8:1 |
| 8x8 | 3 | 2048 | 192 | 11 | 32.50 | 0.8546 | 0.1037 | 13.43 | 46.5:1 |
| 8x8 | 4 | 2048 | 256 | 11 | 33.28 | 0.8705 | 0.0946 | 14.43 | 34.9:1 |
| 8x8 | 8 | 2048 | 512 | 11 | 36.27 | 0.9362 | 0.0513 | 9.40 | 17.5:1 |
| 4x4 | 1 | 2048 | 16 | 11 | 28.41 | 0.7128 | 0.2294 | 25.46 | 558.5:1 |
| 4x4 | 2 | 2048 | 32 | 11 | 27.05 | 0.6670 | 0.2668 | 26.84 | 279.3:1 |
| 4x4 | 3 | 2048 | 48 | 11 | 27.34 | 0.6842 | 0.2545 | 24.81 | 186.2:1 |
| 4x4 | 4 | 2048 | 64 | 11 | 27.93 | 0.7107 | 0.2573 | 34.90 | 139.6:1 |
| 4x4 | 8 | 2048 | 128 | 11 | 23.34 | 0.5634 | 0.3802 | 102.80 | 69.8:1 |
| 4x4 | 2 | 1024 | 32 | 10 | 26.84 | 0.6591 | 0.2738 | 29.32 | 307.2:1 |
| 4x4 | 3 | 1024 | 48 | 10 | 27.19 | 0.6755 | 0.2683 | 28.56 | 204.8:1 |
| 2x2 | 1 | 2048 | 4 | 11 | 24.68 | 0.5864 | 0.3477 | 67.20 | 2234.2:1 |
| 2x2 | 2 | 2048 | 8 | 11 | 23.43 | 0.5570 | 0.3778 | 95.16 | 1117.1:1 |
| 2x2 | 3 | 2048 | 12 | 11 | 23.92 | 0.5671 | 0.3627 | 76.07 | 744.7:1 |
| 2x2 | 4 | 2048 | 16 | 11 | 24.68 | 0.5799 | 0.3474 | 70.20 | 558.5:1 |
| 2x2 | 8 | 2048 | 32 | 11 | 25.72 | 0.6497 | 0.2985 | 42.35 | 279.3:1 |

### FLAIR-1 Preliminary Subset Results

The FLAIR-1 path is newer than the EuroSAT study. The table below records the first `flair-subset50` validation run for native `512x512` RGB FLAIR patches using depths `1`, `4`, and `8`. The run used a `5,000` image training subset and a `1,000` image validation subset, so these are preliminary subset results rather than final full-dataset conclusions.

For FLAIR-1, the raw image size is:

```text
original image bits = 512 x 512 x 3 x 8 = 6,291,456 bits
```

| Model | Code Shape | Codebook | Codes/Image | PSNR (dB) | SSIM | LPIPS | FID | RQ Code Compression |
|-------|------------|----------|-------------|-----------|------|-------|-----|---------------------|
| flair-subset50-rqvae-8x8x1 | 8x8x1 | 2048 | 64 | 18.63 | 0.3253 | 0.5700 | 267.05 | 8936.7:1 |
| flair-subset50-rqvae-8x8x4 | 8x8x4 | 2048 | 256 | 19.70 | 0.3637 | 0.5379 | 117.59 | 2234.2:1 |
| flair-subset50-rqvae-8x8x8 | 8x8x8 | 2048 | 512 | 19.07 | 0.3694 | 0.5771 | 197.02 | 1117.1:1 |

## Main Findings

### Best Quality

The `8x8x8` model achieved the best reconstruction quality:

- PSNR: `36.27 dB`
- SSIM: `0.9362`
- LPIPS: `0.0513`
- FID: `9.40`

This configuration is the strongest option when reconstruction fidelity matters most.

### Best Trade-Off

The `8x8x3` and `8x8x4` models appear to offer the best quality-to-compression balance:

- `8x8x3`: `32.50 dB`, `46.5:1`
- `8x8x4`: `33.28 dB`, `34.9:1`

### Most Compressed

The `2x2x1` model has the smallest code representation:

- Compression ratio: `2234.2:1`
- PSNR: `24.68 dB`

Among the extreme `2x2` models, `2x2x8` gives better reconstruction quality, but it uses more codes and therefore has a lower compression ratio than `2x2x1`.

### Unexpected Result

The `4x4` family became worse as depth increased:

- `4x4x1`: `28.41 dB`
- `4x4x8`: `23.34 dB`

The follow-up `4x4x2` and `4x4x3` runs improved slightly from depth `2` to depth `3`, but the original `4x4x1` result still remained stronger than expected. The `k1024` runs did not improve the `4x4` results.

### Expected vs Observed Trend

What was expected:

- Increasing depth at a fixed spatial size would generally improve reconstruction quality.
- Under the same overall code budget, RQ-VAE settings would typically outperform the corresponding VQ-VAE-style baseline.

What was observed in the current runs:

- The `8x8` family followed the expected trend as depth increased from `1` to `8`.
- The `4x4` family did not improve with depth, and instead became worse at higher depths.
- The expected RQ-VAE advantage under similar code budgets was not consistently seen in the current results.

This suggests that the current hyperparameter setting may not yet be well matched to `64x64` EuroSAT images, especially for smaller latent grids.

### FLAIR-1 Initial Observation

The first FLAIR subset run shows the same pipeline can train and evaluate native `512x512` aerial patches, but quality is still early-stage. The best preliminary FLAIR subset result was `8x8x4` by PSNR and FID, while `8x8x8` slightly improved SSIM. More tuning is needed before comparing FLAIR directly with the completed EuroSAT sweep.

## Scientific Takeaways

1. Spatial latent size mattered more than depth for these small images.
2. An `8x8` latent grid was much more effective than `4x4` at preserving reconstruction quality.
3. RQ-VAE can support multiple operating points depending on mission needs.
4. Extreme compression appears feasible for constrained scenarios, though with lower image quality.
5. The same RQ-VAE/NAC workflow can now be applied to larger aerial imagery, but FLAIR-1 needs separate tuning because its native resolution and visual structure differ from EuroSAT.

## Repository Structure

```text
.
|-- nac/
|   |-- arithmetic_coding.py
|   |-- nac_eurosat.py
|   `-- ngram.py
|-- results/
|   |-- classifier_test_results.json
|   |-- metrics.log
|   |-- summary.json
|   `-- eurosat-rqvae-*/
|-- rq-vae/
|   |-- configs/eurosat/stage1/
|   |-- configs/flair/stage1/
|   |-- evaluate_metrics.py
|   |-- run_flair_8x8_sweep.sh
|   |-- train_eval_classifier.py
|   |-- train_eurosat.py
|   `-- rqvae/
|       `-- img_datasets/
|           |-- eurosat.py
|           `-- flair.py
|-- eurosat_split_indices.pt
|-- rq_nac_eurosat_colab.ipynb
`-- split_indices.py
```

## Key Files

- `rq-vae/train_eurosat.py`: trains EuroSAT or FLAIR RQ-VAE models and exports latent codes
- `rq-vae/evaluate_metrics.py`: computes PSNR, SSIM, LPIPS, and FID for train, validation, or test splits
- `rq-vae/rqvae/img_datasets/flair.py`: loads FLAIR-1 images from official CSV files
- `rq-vae/run_flair_8x8_sweep.sh`: runs the FLAIR `8x8xD` depth sweep
- `rq-vae/train_eval_classifier.py`: baseline classifier training and planned reconstruction evaluation
- `nac/nac_eurosat.py`: applies N-gram arithmetic coding to exported EuroSAT or FLAIR RQ-VAE codes
- `split_indices.py`: generates reproducible dataset splits
- `results/summary.json`: combined quantitative results for the initial 9 configurations
- `results/followup_metrics.log`: depth `2` and `3` follow-up metrics for `8x8` and `4x4`
- `results/followup_2x2_metrics.log`: depth `2` and `3` follow-up metrics for `2x2`

## Reproducing the Project

### 1. Clone the Repository

```bash
git clone https://github.com/HemanthSud/OEC-Image-Processing.git
cd OEC-Image-Processing
```

### 2. Prepare the EuroSAT Dataset

Download EuroSAT RGB and place it in:

```text
EuroSAT_RGB/
```

with one folder per class.

### 3. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install omegaconf einops lpips tensorboard scikit-image tqdm matplotlib pillow numpy scipy pyyaml
```

### 4. Generate Splits

```bash
python split_indices.py
```

### 5. Train a Model

```bash
cd rq-vae
python train_eurosat.py \
  -m configs/eurosat/stage1/eurosat-rqvae-8x8x4.yaml \
  -o output/eurosat-rqvae-8x8x4 \
  --epochs 150
```

### 5a. Follow-Up Depth 2 and 3 Experiments

Intermediate-depth configs are available for all three spatial sizes:

- `configs/eurosat/stage1/eurosat-rqvae-8x8x2.yaml`
- `configs/eurosat/stage1/eurosat-rqvae-8x8x3.yaml`
- `configs/eurosat/stage1/eurosat-rqvae-4x4x2.yaml`
- `configs/eurosat/stage1/eurosat-rqvae-4x4x3.yaml`
- `configs/eurosat/stage1/eurosat-rqvae-2x2x2.yaml`
- `configs/eurosat/stage1/eurosat-rqvae-2x2x3.yaml`

Example:

```bash
python train_eurosat.py \
  -m configs/eurosat/stage1/eurosat-rqvae-4x4x2.yaml \
  -o output/eurosat-rqvae-4x4x2 \
  --epochs 150
```

The training script also supports direct hyperparameter overrides for follow-up tuning. For example, to try a smaller codebook:

```bash
python train_eurosat.py \
  -m configs/eurosat/stage1/eurosat-rqvae-4x4x3.yaml \
  -o output/eurosat-rqvae-4x4x3-k1024 \
  --epochs 150 \
  --n-embed 1024
```

### 6. Evaluate Reconstruction Metrics

```bash
python evaluate_metrics.py --output-dirs \
  output/eurosat-rqvae-8x8x1 \
  output/eurosat-rqvae-8x8x4 \
  output/eurosat-rqvae-8x8x8
```

### 7. Run NAC on Exported Codes

```bash
cd ../nac
python nac_eurosat.py
```

## FLAIR-1 8x8 Experiments

FLAIR-1 support has been added for native `512x512` RGB aerial patches. The current FLAIR experiment path is intentionally focused on the `8x8xD` family because that is the set planned for the SSH/server sweep.

### Dataset Size and Split

The full FLAIR-1 official CSV splits contain:

| Split | Full size | Used (50%) |
|-------|-----------|------------|
| Train | 47,587 | 23,800 |
| Val | 14,125 | 7,050 |
| Test | 15,700 | 7,850 |
| **Total** | **77,412** | **38,700** |

The recommended approach uses half of each official split (~38,700 total). Drawing from each official split preserves FLAIR's geographic and temporal domain separation across France.

Download FLAIR-1 and place the actual data folders so the official CSV paths resolve from inside `rq-vae/`. The compression loader reads image paths from the CSVs and ignores masks, but the label folders can stay in the normal FLAIR layout. The expected layout is:

```text
Research-clean/
|-- flair-1-paths-train.csv
|-- flair-1-paths-val.csv
|-- flair-1-paths-test.csv
|-- flair_aerial_train/
`-- data/
```

Install the extra TIFF/geospatial dependency:

```bash
pip install "rasterio<1.5"
```

Run the `8x8xD` sweep on the SSH server with the EuroSAT-matched 27,000-image subset:

```bash
cd rq-vae
chmod +x run_flair_8x8_sweep.sh
MAX_TRAIN_SAMPLES=23800 MAX_VAL_SAMPLES=7050 MAX_TEST_SAMPLES=7850 \
  ./run_flair_8x8_sweep.sh
```

To run on the full dataset instead, omit the sample limits:

```bash
./run_flair_8x8_sweep.sh
```

Available FLAIR configs:

- `configs/flair/stage1/flair-rqvae-8x8x1.yaml`
- `configs/flair/stage1/flair-rqvae-8x8x2.yaml`
- `configs/flair/stage1/flair-rqvae-8x8x4.yaml`
- `configs/flair/stage1/flair-rqvae-8x8x8.yaml`
- `configs/flair/stage1/flair-rqvae-8x8x16.yaml`

If the server runs out of memory, lower the batch size without editing configs:

```bash
BATCH_SIZE=1 ./run_flair_8x8_sweep.sh
```

Evaluate the trained FLAIR models. Use `--split val` when the FLAIR test archive is not installed locally:

```bash
python evaluate_metrics.py --split val --output-dirs \
  output/flair-rqvae-8x8x1 \
  output/flair-rqvae-8x8x2 \
  output/flair-rqvae-8x8x4 \
  output/flair-rqvae-8x8x8 \
  output/flair-rqvae-8x8x16
```

Run NAC on one exported FLAIR code file. Use the EuroSAT-matched split counts (21,600 train + 2,700 val = 24,300 total exported sequences):

```bash
cd ../nac
python nac_eurosat.py \
  --dataset flair \
  --height 8 \
  --width 8 \
  --depth 4 \
  --image-size 512 \
  --n-train 23800 \
  --n-total 30850
```

## Possible Next Steps

The EuroSAT reconstruction study and metric evaluation are complete. The main remaining items are:

1. Evaluate classification accuracy on reconstructed images for each compression setting.
2. Run NAC consistently across the updated code files and report entropy-coded bitrates.
3. Complete the full FLAIR-1 `8x8xD` sweep beyond the initial subset run.
4. Tune hyperparameters such as codebook size and loss weights to see whether RQ-VAE gives the expected advantage under similar code budgets.
5. Integrate the compression pipeline into a more realistic downlink or deployment setting.

## References

- RQ-VAE: "Residual Quantized Variational Autoencoders", CVPR 2023
- EuroSAT dataset: https://github.com/phelber/eurosat
- FLAIR-1 dataset and baseline code: https://github.com/IGNF/FLAIR-1

## Contact

Hemanth Sudhaharan  
North Carolina State University  
NICE Lab  
Email: hsudhah@ncsu.edu
