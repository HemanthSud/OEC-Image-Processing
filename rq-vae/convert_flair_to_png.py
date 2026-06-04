"""
Convert FLAIR GeoTIFF images to PNG for faster training.

Reads channels 1,2,3 (RGB) from each TIFF using rasterio, scales to uint8,
and saves as PNG in a mirrored directory under flair_aerial_train_png/.
Skips files that already exist. Uses multiprocessing for speed.

Usage:
    python convert_flair_to_png.py \
        --csv ../flair-1-paths-train.csv \
        --data-root .. \
        --out-root ../flair_aerial_train_png \
        --workers 32
"""
import argparse
import csv
import os
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import numpy as np

try:
    import rasterio
except ImportError:
    raise ImportError("pip install rasterio")

from PIL import Image


def resolve_path(raw_path, csv_dir, data_root):
    path = Path(raw_path.strip())
    candidates = [
        Path.cwd() / path,
        csv_dir / path,
        Path(data_root) / path,
        Path(data_root) / raw_path.strip().lstrip("./"),
        Path(data_root) / raw_path.strip().removeprefix("../"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])


def convert_one(args):
    src_path, dst_path = args
    if os.path.exists(dst_path):
        return 'skip'
    try:
        with rasterio.open(src_path) as src:
            arr = src.read([1, 2, 3]).astype(np.float32)
            dtypes = src.dtypes
        first_dtype = np.dtype(dtypes[0])
        if np.issubdtype(first_dtype, np.integer):
            arr = arr / np.iinfo(first_dtype).max
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr.transpose(1, 2, 0), 'RGB')
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        img.save(dst_path, format='PNG')
        return 'ok'
    except Exception as e:
        return f'err:{e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='FLAIR CSV file')
    parser.add_argument('--data-root', default='..', help='Root containing flair_aerial_train/')
    parser.add_argument('--out-root', required=True, help='Output root for PNG files')
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    csv_dir = csv_path.parent
    out_root = Path(args.out_root)
    flair_train = Path(args.data_root) / 'flair_aerial_train'

    pairs = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if not row:
                continue
            raw = row[0].strip()
            src = resolve_path(raw, csv_dir, args.data_root)
            # Mirror path under out_root, replacing flair_aerial_train prefix
            src_p = Path(src)
            try:
                rel = src_p.relative_to(flair_train)
            except ValueError:
                rel = Path(src_p.name)
            dst = str(out_root / rel.with_suffix('.png'))
            pairs.append((src, dst))

    total = len(pairs)
    print(f"Found {total} images to convert")

    with Pool(args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(convert_one, pairs, chunksize=16)):
            results.append(r)
            if (i + 1) % 500 == 0:
                ok = results.count('ok')
                skip = results.count('skip')
                errs = sum(1 for r in results if r.startswith('err'))
                print(f"  {i+1}/{total} — converted: {ok}, skipped: {skip}, errors: {errs}")

    ok = results.count('ok')
    skip = results.count('skip')
    errs = sum(1 for r in results if r.startswith('err'))
    print(f"\nDone. Converted: {ok}, Skipped (already existed): {skip}, Errors: {errs}")


if __name__ == '__main__':
    main()
