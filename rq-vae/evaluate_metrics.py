"""
Evaluate reconstruction quality metrics for trained RQ-VAE models.
Computes PSNR, SSIM, LPIPS, and FID on test set reconstructions.

Usage:
    python evaluate_metrics.py --output-dirs output/8x8x4 output/4x4x4
"""
import yaml
from omegaconf import OmegaConf
from rqvae.losses.vqgan.lpips import LPIPS
from rqvae.img_datasets.transforms import create_transforms
from rqvae.img_datasets.eurosat import EuroSAT
from rqvae.models.rqvae.rqvae import RQVAE
import os
import sys
import math
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compute_psnr(recon, orig):
    """Compute PSNR between batches. Both in [0,1], shape (B,C,H,W)."""
    mse = F.mse_loss(recon, orig, reduction='none').mean(dim=[1, 2, 3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr


def _gaussian_kernel_1d(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def compute_ssim_batch(img1, img2, window_size=11, sigma=1.5):
    """Compute SSIM per image. Both in [0,1], shape (B,C,H,W). Returns (B,) tensor."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C = img1.size(1)
    device = img1.device

    g = _gaussian_kernel_1d(window_size, sigma).to(device)
    window_2d = g.unsqueeze(1) * g.unsqueeze(0)  # (ws, ws)
    window = window_2d.unsqueeze(0).unsqueeze(
        0).expand(C, 1, window_size, window_size)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(dim=[1, 2, 3])


class InceptionV3Features(torch.nn.Module):
    """Extract 2048-dim pool3 features from InceptionV3 for FID computation."""

    def __init__(self):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model
        self.register_buffer('mean', torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        # x in [0, 1], any spatial size
        x = F.interpolate(x, size=(299, 299),
                          mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        return self.model(x)


def compute_fid(real_feats, fake_feats):
    """Compute Frechet Inception Distance from numpy feature arrays."""
    from scipy import linalg

    mu1 = np.mean(real_feats, axis=0)
    mu2 = np.mean(fake_feats, axis=0)
    sigma1 = np.cov(real_feats, rowvar=False)
    sigma2 = np.cov(fake_feats, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
    return fid


@torch.no_grad()
def compute_all_metrics(model, dataloader, device, compute_fid_flag=True):
    """
    Compute PSNR, SSIM, LPIPS, and optionally FID for a trained RQ-VAE model.

    Args:
        model: Trained RQVAE model (already on device, in eval mode)
        dataloader: DataLoader yielding (images, _) with images in [-1, 1]
        device: torch device
        compute_fid_flag: Whether to compute FID (slower, needs InceptionV3)

    Returns:
        dict with keys: psnr, ssim, lpips, fid (or None)
    """
    model.eval()

    lpips_fn = LPIPS().to(device).eval()

    all_psnr = []
    all_ssim = []
    all_lpips = []

    if compute_fid_flag:
        inception = InceptionV3Features().to(device).eval()
        real_features = []
        fake_features = []

    for imgs, _ in tqdm(dataloader, desc='Computing metrics'):
        imgs = imgs.to(device)
        recon = model(imgs)[0]

        # Convert to [0, 1] for metrics
        imgs_01 = imgs * 0.5 + 0.5
        recon_01 = torch.clamp(recon * 0.5 + 0.5, 0, 1)

        psnr_vals = compute_psnr(recon_01, imgs_01)
        ssim_vals = compute_ssim_batch(recon_01, imgs_01)
        lpips_vals = lpips_fn(imgs, recon).view(-1)
        
        all_psnr.append(psnr_vals)
        all_ssim.append(ssim_vals)
        all_lpips.append(lpips_vals)

        if compute_fid_flag:
            real_features.append(inception(imgs_01).cpu().numpy())
            fake_features.append(inception(recon_01).cpu().numpy())

    results = {
        'psnr': torch.cat(all_psnr).mean().item(),
        'ssim': torch.cat(all_ssim).mean().item(),
        'lpips': torch.cat(all_lpips).mean().item(),
    }

    if compute_fid_flag:
        real_feats = np.concatenate(real_features, axis=0)
        fake_feats = np.concatenate(fake_features, axis=0)
        results['fid'] = compute_fid(real_feats, fake_feats)
    else:
        results['fid'] = None

    del lpips_fn
    if compute_fid_flag:
        del inception
    torch.cuda.empty_cache()

    return results


def load_model_from_dir(output_dir, device):
    """Load a trained model from its output directory."""
    config_path = os.path.join(output_dir, 'config.yaml')
    ckpt_path = os.path.join(output_dir, 'best_model.pt')

    with open(config_path) as f:
        config = OmegaConf.create(yaml.load(f, Loader=yaml.FullLoader))

    hps = dict(config.arch.hparams)
    ddconfig = dict(config.arch.ddconfig)
    model = RQVAE(**hps, ddconfig=ddconfig, checkpointing=False).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    return model, config


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate RQ-VAE reconstruction metrics')
    parser.add_argument('--output-dirs', nargs='+', required=True,
                        help='Paths to trained model output directories')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--no-fid', action='store_true',
                        help='Skip FID computation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_results = OrderedDict()

    for out_dir in args.output_dirs:
        name = os.path.basename(out_dir)
        print(f'\n{"="*60}')
        print(f'Evaluating: {name}')
        print(f'{"="*60}')

        model, config = load_model_from_dir(out_dir, device)

        # Load test set
        dataset_cfg = OmegaConf.create(
            {'transforms': config.dataset.transforms})
        transforms_test = create_transforms(
            dataset_cfg, split='val', is_eval=True)
        root = config.dataset.get('root', '../EuroSAT_RGB')
        split_path = config.dataset.get(
            'split_indices_path', '../eurosat_split_indices.pt')
        dataset_test = EuroSAT(root, split='test', transform=transforms_test,
                               split_indices_path=split_path)
        loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

        results = compute_all_metrics(model, loader, device,
                                      compute_fid_flag=not args.no_fid)
        all_results[name] = results

        code_shape = list(config.arch.hparams.code_shape)
        print(f'  Code shape: {code_shape[0]}x{code_shape[1]}x{code_shape[2]}')
        print(f'  PSNR:  {results["psnr"]:.2f} dB')
        print(f'  SSIM:  {results["ssim"]:.4f}')
        print(f'  LPIPS: {results["lpips"]:.4f}')
        if results['fid'] is not None:
            print(f'  FID:   {results["fid"]:.2f}')

        del model
        torch.cuda.empty_cache()

    # Summary table
    print(f'\n{"="*70}')
    print(f'{"Model":<20} {"PSNR(dB)":<10} {"SSIM":<10} {"LPIPS":<10} {"FID":<10}')
    print(f'{"-"*70}')
    for name, r in all_results.items():
        fid_str = f'{r["fid"]:.2f}' if r['fid'] is not None else 'N/A'
        print(
            f'{name:<20} {r["psnr"]:<10.2f} {r["ssim"]:<10.4f} {r["lpips"]:<10.4f} {fid_str:<10}')


if __name__ == '__main__':
    main()
