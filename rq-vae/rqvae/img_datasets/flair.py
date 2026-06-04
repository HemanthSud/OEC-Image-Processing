"""
FLAIR-1 aerial image loader for RQ-VAE reconstruction training.

The official FLAIR CSVs contain image and mask paths. For compression we only
need the image path, so this dataset returns (image, 0) to match the EuroSAT
training loop.
"""
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
except ImportError as exc:  # pragma: no cover - exercised only on missing deps
    rasterio = None
    _RASTERIO_IMPORT_ERROR = exc
else:
    _RASTERIO_IMPORT_ERROR = None


class FLAIR(Dataset):
    def __init__(self, csv_path, split='train', transform=None, channels=None,
                 data_root=None, max_samples=None, return_path=False,
                 png_root=None):
        if rasterio is None:
            raise ImportError(
                "FLAIR loading requires rasterio. Install it with `pip install rasterio`."
            ) from _RASTERIO_IMPORT_ERROR

        self.csv_path = Path(csv_path).expanduser()
        self.split = split
        self.transform = transform
        self.channels = channels or [1, 2, 3]
        self.data_root = Path(data_root).expanduser() if data_root else None
        self.return_path = return_path
        self.png_root = Path(png_root).resolve() if png_root else None

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"FLAIR CSV not found: {self.csv_path}")

        tiff_files = self._read_image_paths()
        if max_samples is not None and max_samples > 0:
            tiff_files = tiff_files[:max_samples]

        # Pre-compute PNG paths once at init so __getitem__ has zero overhead
        if self.png_root is not None:
            self.files = []
            self.use_png = []
            for p in tiff_files:
                png = self._png_path_for(p)
                if png is not None:
                    self.files.append(png)
                    self.use_png.append(True)
                else:
                    self.files.append(p)
                    self.use_png.append(False)
            n_png = sum(self.use_png)
            print(f"[FLAIR] {len(self.files)} images ({split}): "
                  f"{n_png} PNG, {len(self.files)-n_png} TIFF fallback")
        else:
            self.files = tiff_files
            self.use_png = [False] * len(self.files)
            if max_samples is not None and max_samples > 0:
                print(f"[FLAIR] Limited to {len(self.files)} images ({split})")
            else:
                print(f"[FLAIR] Found {len(self.files)} images ({split})")

    def _read_image_paths(self):
        files = []
        csv_dir = self.csv_path.parent
        with self.csv_path.open(newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                files.append(self._resolve_path(row[0].strip(), csv_dir))
        return files

    def _resolve_path(self, raw_path, csv_dir):
        path = Path(os.path.expandvars(os.path.expanduser(raw_path)))
        if path.is_absolute():
            return str(path)

        candidates = [
            Path.cwd() / path,
            csv_dir / path,
        ]
        if self.data_root is not None:
            candidates.extend([
                self.data_root / path,
                self.data_root / raw_path.lstrip("./"),
                self.data_root / raw_path.removeprefix("../"),
            ])

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0])

    def _png_path_for(self, tiff_path):
        """Compute PNG path at init time. Returns path string or None."""
        p = Path(tiff_path).resolve()
        parts = p.parts
        # Try nested path first (mirrors flair_aerial_train subdirectory structure)
        try:
            idx = parts.index('flair_aerial_train')
            rel = Path(*parts[idx + 1:]).with_suffix('.png')
            candidate = self.png_root / rel
            if candidate.exists():
                return str(candidate)
        except ValueError:
            pass
        # Fallback: flat file (conversion script may have dropped subdirs)
        flat = self.png_root / (p.stem + '.png')
        return str(flat) if flat.exists() else None

    def __len__(self):
        return len(self.files)

    def _read_image(self, img_path, use_png):
        if use_png:
            from PIL import Image as PILImage
            img = PILImage.open(img_path).convert('RGB')
            array = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(np.clip(array, 0.0, 1.0).transpose(2, 0, 1))

        with rasterio.open(img_path) as src:
            array = src.read(self.channels)
            dtypes = src.dtypes

        array = array.astype(np.float32)
        first_dtype = np.dtype(dtypes[self.channels[0] - 1])
        if np.issubdtype(first_dtype, np.integer):
            max_value = np.iinfo(first_dtype).max
            array = array / max(max_value, 1)
        else:
            array = np.clip(array, 0.0, 1.0)

        array = np.clip(array, 0.0, 1.0)
        return torch.from_numpy(array)

    def __getitem__(self, idx):
        img = self._read_image(self.files[idx], self.use_png[idx])

        if self.transform is not None:
            img = self.transform(img)

        if self.return_path:
            return img, self.files[idx]
        return img, 0

    def get_image_path(self, idx):
        return self.files[idx]
