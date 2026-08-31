"""Reconstruct FLAIR-1 val images at RQ-VAE depth q and write 5-band GeoTIFFs
for downstream segmentation.

Hybrid 5-band design: RQ-VAE only compresses RGB, while FLAIR's baseline
segmenter takes RGB + NIR + Elevation. We therefore write

    bands 1-3  = the depth-q RECONSTRUCTION
    bands 4-5  = the ORIGINAL NIR and Elevation, copied through untouched

which keeps IGNF's published 5-band U-Net/ResNet34 checkpoint usable as-is
(no segmentation training) and is physically honest: only the optical bands
went through the codec.

Two conditions beyond q in {1,2,4,8,16}:

    --depth orig    no files written; the CSV points at the originals.
                    Gives mIoU_ref, the uncompressed reference.
    --depth blank   bands 1-3 filled with the per-band dataset means, i.e.
                    RGB carries no information at all. Gives mIoU_floor:
                    what the segmenter still achieves from NIR + Elevation
                    alone. That is the decision-theoretic zero for a
                    scheduler -- "what you get by not delivering the image"
                    -- and anchoring s_q there instead of at mIoU = 0 is the
                    difference between a ~12%-wide quality spread and a wide
                    one. See oec_sim/utility.py:load_quality_table.

SERVER ONLY: needs the RQ-VAE checkpoint (rq-vae/output/flair-rqvae-8x8x16),
the FLAIR GeoTIFFs and CUDA. Run inside tmux.

    python3 downstream/recon_to_geotiff.py --depth 8 \
        --csv downstream/flair-1-paths-val-7050.csv --data-root .. \
        --out-root /scratch/flair_recon_q8 --min-free-gb 50
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

import rasterio

from evaluate_metrics import load_model_from_dir
from rqvae.img_datasets.flair import FLAIR
from rqvae.img_datasets.transforms import create_transforms

# FLAIR-1 per-band means on raw 0-255 values (configs/flair-1-config.yaml).
# Only the first three are used, for the --depth blank condition.
FLAIR_RGB_MEANS = (105, 111, 102)


def build_dataset(csv_path, data_root, image_size, max_samples=None):
    """FLAIR(..., return_path=True) directly -- create_dataset_split does not
    forward that kwarg. png_root is deliberately NOT set: the PNGs are 8-bit
    RGB only, and we need the original TIFF for NIR/Elevation and for the
    rasterio profile (CRS/transform/nodata) we copy onto the output."""
    from types import SimpleNamespace
    cfg = SimpleNamespace(transforms=SimpleNamespace(
        type='flair', image_size=image_size, channels=[1, 2, 3],
        get=lambda k, d=None: {'image_size': image_size,
                               'channels': [1, 2, 3]}.get(k, d)))
    tf = create_transforms(cfg, split='val', is_eval=True)
    return FLAIR(csv_path=csv_path, split='val', transform=tf,
                 channels=[1, 2, 3], data_root=data_root,
                 max_samples=max_samples, return_path=True)


def out_path_for(src, data_root, out_root):
    """Mirror the source tree under out_root (same pattern as
    convert_flair_to_png.py)."""
    src = Path(src).resolve()
    try:
        rel = src.relative_to(Path(data_root).resolve())
    except ValueError:
        rel = Path(*src.parts[-4:])
    return Path(out_root) / rel


def check_disk(out_root, min_free_gb):
    Path(out_root).mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out_root).free / 1e9
    if free_gb < min_free_gb:
        raise SystemExit(
            f'ABORT before writing anything: only {free_gb:.1f} GB free at '
            f'{out_root}, need >= {min_free_gb} GB. 5-band uint8 512x512 is '
            f'~0.6-0.9 GB per 1000 images after LZW. Free space or lower '
            f'--min-free-gb deliberately.')
    return free_gb


def write_tif(dst, rgb_u8, src_path):
    with rasterio.open(src_path) as src:
        if src.count < 5:
            raise SystemExit(
                f'{src_path} has {src.count} bands, expected 5 '
                f'(RGB + NIR + Elevation). The hybrid pipeline needs the '
                f'original NIR/Elevation bands.')
        nir, elev = src.read(4), src.read(5)
        profile = src.profile
    profile.update(count=5, dtype='uint8', compress='lzw', predictor=2,
                   tiled=True, blockxsize=256, blockysize=256)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst, 'w', **profile) as out:
        for b in range(3):
            out.write(rgb_u8[b], b + 1)
        out.write(nir.astype('uint8'), 4)
        out.write(elev.astype('uint8'), 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', required=True,
                    help="codebook stages, or 'orig' / 'blank'")
    ap.add_argument('--csv', required=True, help='frozen 2-column FLAIR CSV')
    ap.add_argument('--data-root', default='..')
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--model-dir', default='output/flair-rqvae-8x8x16')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--max-samples', type=int, default=None)
    ap.add_argument('--min-free-gb', type=float, default=50.0)
    args = ap.parse_args()

    if args.depth == 'orig':
        print('--depth orig: nothing to write, point the CSV at the originals')
        return

    check_disk(args.out_root, args.min_free_gb)
    ds = build_dataset(args.csv, args.data_root, args.image_size,
                       args.max_samples)
    print(f'{len(ds)} images from {args.csv}')

    if args.depth == 'blank':
        for i in range(len(ds)):
            src = ds.get_image_path(i)
            flat = np.stack([np.full((args.image_size, args.image_size), m,
                                     dtype=np.uint8) for m in FLAIR_RGB_MEANS])
            write_tif(out_path_for(src, args.data_root, args.out_root),
                      flat, src)
            if (i + 1) % 500 == 0:
                print(f'  {i + 1}/{len(ds)}', flush=True)
        print(f'wrote {len(ds)} blank-RGB tiles -> {args.out_root}')
        return

    q = int(args.depth)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'loading {args.model_dir} on {device} ...')
    model, _cfg = load_model_from_dir(args.model_dir, device)
    model.eval()

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    done = 0
    with torch.no_grad():
        for xs, paths in loader:
            xs = xs.to(device, non_blocking=True)
            # same truncation semantics as evaluate_truncation.py
            out = model.forward_partial_code(xs, q - 1, decode_type='add')
            if isinstance(out, (tuple, list)):
                out = out[0]
            rgb = (out.clamp(-1, 1) * 0.5 + 0.5)          # [-1,1] -> [0,1]
            rgb = np.round(rgb.cpu().numpy() * 255).astype(np.uint8)
            for j, src in enumerate(paths):
                write_tif(out_path_for(src, args.data_root, args.out_root),
                          rgb[j], src)
            done += len(paths)
            if done % 500 < args.batch_size:
                print(f'  {done}/{len(ds)}', flush=True)
    print(f'wrote {done} depth-{q} tiles -> {args.out_root}')


if __name__ == '__main__':
    main()
