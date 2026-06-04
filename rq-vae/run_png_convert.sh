#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 convert_flair_to_png.py --csv ../flair-1-paths-train.csv --data-root .. --out-root ../flair_aerial_train_png --workers 32
