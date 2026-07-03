"""
Evaluate whether the depth-16 RQ-VAE model can be reused at shallower depths
by truncating to the first k codebook stages (forward_partial_code, decode_type='add').

Run from rq-vae/:
    python3 evaluate_truncation.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from collections import OrderedDict
from torch.utils.data import DataLoader

from evaluate_metrics import compute_all_metrics, load_model_from_dir
from rqvae.img_datasets import create_dataset_split


class TruncatedModel(torch.nn.Module):
    """Routes forward() to forward_partial_code at a fixed depth."""

    def __init__(self, base_model, depth):
        super().__init__()
        self.base = base_model
        self.code_idx = depth - 1  # 0-indexed

    def forward(self, xs):
        out = self.base.forward_partial_code(xs, self.code_idx, decode_type='add')
        return out, None, None


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'output/flair-rqvae-8x8x16'
    depths = [1, 2, 4, 8, 16]

    print(f'Loading depth-16 model from {output_dir} ...')
    model, config = load_model_from_dir(output_dir, device)
    model.eval()

    dataset = create_dataset_split(config, split='val', is_eval=True)
    num_workers = config.experiment.get('num_workers', 4)
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    all_results = OrderedDict()

    for d in depths:
        label = f'd16→depth{d}'
        print(f'\n{"="*60}')
        print(f'Truncated to depth {d}  (first {d} of 16 codebook stages)')
        print(f'{"="*60}')

        wrapped = TruncatedModel(model, d)
        wrapped.eval()

        results = compute_all_metrics(wrapped, loader, device, compute_fid_flag=True)
        all_results[label] = results

        print(f'  PSNR:  {results["psnr"]:.2f} dB')
        print(f'  SSIM:  {results["ssim"]:.4f}')
        print(f'  LPIPS: {results["lpips"]:.4f}')
        if results['fid'] is not None:
            print(f'  FID:   {results["fid"]:.2f}')

    print(f'\n{"="*70}')
    print('Summary — depth-16 model truncated vs. dedicated models')
    print(f'{"Model":<22} {"PSNR(dB)":<10} {"SSIM":<10} {"LPIPS":<10} {"FID":<10}')
    print(f'{"-"*62}')
    for name, r in all_results.items():
        fid_str = f'{r["fid"]:.2f}' if r['fid'] is not None else 'N/A'
        print(f'{name:<22} {r["psnr"]:<10.2f} {r["ssim"]:<10.4f} {r["lpips"]:<10.4f} {fid_str:<10}')

    print('\nReference (dedicated models trained at each depth):')
    print(f'  {"8x8x1":<20} {"20.63":<10} {"0.4560":<10} {"0.4595":<10} {"71.33"}')
    print(f'  {"8x8x8":<20} {"21.02":<10} {"0.4643":<10} {"0.4779":<10} {"73.19"}')
    print(f'  {"8x8x16":<20} {"21.06":<10} {"0.4899":<10} {"0.5039":<10} {"125.78"}')


if __name__ == '__main__':
    main()
