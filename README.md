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
- Training time: about 34 hours for all 9 models using dual-GPU parallel runs
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

Computed the following reconstruction metrics for all 9 trained models:

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index Measure)
- LPIPS (Learned Perceptual Image Patch Similarity)
- FID (Frechet Inception Distance)

The results were saved in the [`results/`](results/) directory.

### Phase 6: Cleanup and GitHub Publishing

1. Removed credentials and local-only clutter.
2. Cleaned code and project files.
3. Organized configs, metrics, and outputs.
4. Pushed the project to GitHub.

## Model Configurations and Results

| Spatial | Depth | Codes/Image | PSNR (dB) | SSIM | LPIPS | FID | Approx. Compression |
|---------|-------|-------------|-----------|------|-------|-----|---------------------|
| 8x8 | 1 | 64 | 29.93 | 0.7861 | 0.1665 | 19.10 | ~12:1 |
| 8x8 | 4 | 256 | 33.28 | 0.8705 | 0.0946 | 14.43 | ~24:1 |
| 8x8 | 8 | 512 | 36.27 | 0.9362 | 0.0513 | 9.40 | ~48:1 |
| 4x4 | 1 | 16 | 28.41 | 0.7128 | 0.2294 | 25.46 | ~48:1 |
| 4x4 | 4 | 64 | 27.93 | 0.7107 | 0.2573 | 34.90 | ~96:1 |
| 4x4 | 8 | 128 | 23.34 | 0.5634 | 0.3802 | 102.80 | ~192:1 |
| 2x2 | 1 | 4 | 24.68 | 0.5864 | 0.3477 | 67.20 | ~192:1 |
| 2x2 | 4 | 16 | 24.68 | 0.5799 | 0.3474 | 70.20 | ~384:1 |
| 2x2 | 8 | 32 | 25.72 | 0.6497 | 0.2985 | 42.35 | ~768:1 |

## Main Findings

### Best Quality

The `8x8x8` model achieved the best reconstruction quality:

- PSNR: `36.27 dB`
- SSIM: `0.9362`
- LPIPS: `0.0513`
- FID: `9.40`

This configuration is the strongest option when reconstruction fidelity matters most.

### Best Trade-Off

The `8x8x4` model appears to offer the best quality-to-compression balance:

- PSNR: `33.28 dB`
- Approximate compression: `24:1`

### Most Compressed

The `2x2x8` model achieved the highest compression level:

- Approximate compression: `768:1`
- PSNR: `25.72 dB`

Even under extreme compression, it still retained usable visual information.

### Unexpected Result

The `4x4` family became worse as depth increased:

- `4x4x1`: `28.41 dB`
- `4x4x8`: `23.34 dB`

This suggests that for `64x64` EuroSAT images, overly small spatial latents may not benefit from deeper residual quantization.

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
- `results/summary.json`: combined quantitative results for all 9 configurations

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

## Status and Next Steps

The reconstruction study and metric evaluation are complete. The main remaining items are:

1. Evaluate classification accuracy on reconstructed images for each compression setting.
2. Investigate why `4x4` models degrade as depth increases.
3. Integrate the compression pipeline into a more realistic downlink or deployment setting.

## References

- RQ-VAE: "Residual Quantized Variational Autoencoders", CVPR 2023
- EuroSAT dataset: https://github.com/phelber/eurosat

## Contact

Hemanth Sudhaharan  
North Carolina State University  
NICE Lab  
Email: hsudhah@ncsu.edu
