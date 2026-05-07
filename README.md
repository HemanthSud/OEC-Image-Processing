# RQ-VAE EuroSAT Image Compression for Orbital Edge Computing

This repository contains an Orbital Edge Computing (OEC) research project on satellite image compression using Residual Quantized Variational Autoencoders (RQ-VAE) and N-gram Arithmetic Coding (NAC).

The project studies how different compression levels affect EuroSAT image reconstruction quality and downstream classification performance in bandwidth-constrained satellite settings.

## Project Motivation

Satellites must transmit imagery to Earth under strict bandwidth limits. For orbital edge computing, compression is not only about reducing size, but also about preserving the information needed for downstream tasks such as land-use classification.

This project explores that trade-off by training multiple RQ-VAE compression models on EuroSAT and comparing their reconstruction quality across several latent sizes and quantization depths.

## Research Context

This work was carried out in NICE Lab at North Carolina State University.

- Researcher: Hemanth Sudhaharan
- Advisor: Xuanhao Luo
- PI: Dr. Yuchen Liu
- Timeline: February 2026 to April 1, 2026



## Project Goal

Evaluate how different compression settings affect:

1. Reconstruction quality of satellite images.
2. Compression ratio and latent-code efficiency.
3. Practical usefulness for downstream image classification.

## Hardware and Dataset

- Server: NCSU cluster (`eb3-2402-grd04.csc.ncsu.edu`)
- GPUs: 2 x NVIDIA RTX A6000 (48 GB each)
- Training time: about 34 hours for the initial 9-model sweep using dual-GPU parallel runs
- Dataset: EuroSAT RGB
- Dataset size: 27,000 images
- Classes: 10 land-use classes
- Image size: `64x64` RGB

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

Follow-up metrics were saved in:

- `results/followup_metrics.log`
- `results/followup_2x2_metrics.log`

### Phase 7: Cleanup and GitHub Publishing

1. Removed credentials and local-only clutter.
2. Cleaned code and project files.
3. Organized configs, metrics, and outputs.
4. Prepared the project for GitHub publishing.

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

## Scientific Takeaways

1. Spatial latent size mattered more than depth for these small images.
2. An `8x8` latent grid was much more effective than `4x4` at preserving reconstruction quality.
3. RQ-VAE can support multiple operating points depending on mission needs.
4. Extreme compression appears feasible for constrained scenarios, though with lower image quality.

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
|   |-- evaluate_metrics.py
|   |-- train_eval_classifier.py
|   |-- train_eurosat.py
|   `-- rqvae/
|-- eurosat_split_indices.pt
|-- project_session_history.txt
|-- rq_nac_eurosat_colab.ipynb
`-- split_indices.py
```

## Key Files

- `rq-vae/train_eurosat.py`: trains EuroSAT RQ-VAE models and exports latent codes
- `rq-vae/evaluate_metrics.py`: computes PSNR, SSIM, LPIPS, and FID
- `rq-vae/train_eval_classifier.py`: baseline classifier training and planned reconstruction evaluation
- `nac/nac_eurosat.py`: applies N-gram arithmetic coding to exported RQ-VAE codes
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

## Possible Next Steps

The reconstruction study and metric evaluation are complete. The main remaining items are:

1. Evaluate classification accuracy on reconstructed images for each compression setting.
2. Run NAC consistently across the updated code files and report entropy-coded bitrates.
3. Tune hyperparameters such as codebook size and loss weights to see whether RQ-VAE gives the expected advantage under similar code budgets.
4. Integrate the compression pipeline into a more realistic downlink or deployment setting.

## References

- RQ-VAE: "Residual Quantized Variational Autoencoders", CVPR 2023
- EuroSAT dataset: https://github.com/phelber/eurosat

## Contact

Hemanth Sudhaharan  
North Carolina State University  
NICE Lab  
Email: hsudhah@ncsu.edu
