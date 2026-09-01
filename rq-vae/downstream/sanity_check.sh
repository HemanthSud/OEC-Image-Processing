#!/usr/bin/env bash
# Small-scale (50-image) sanity check of the reconstruction -> GeoTIFF ->
# FLAIR predict+metrics chain, for blank and one depth (q8), before
# committing to the full 7050 x 6-condition sweep.
set -euo pipefail

RQ_VAE_DIR="$HOME/Research-clean/rq-vae"
FLAIR_DIR="$HOME/Research-clean/FLAIR-1-main"
SCRATCH="$HOME/scratch"
CSV="downstream/flair-1-paths-val-50.csv"

run_cond () {
  local cond="$1" depth="$2"
  echo "=== SANITY $cond ==================================================="
  local recon="$SCRATCH/flair_recon_sanity_$cond"
  local csv_out="$FLAIR_DIR/csv_recon/val-sanity-$cond.csv"

  cd "$RQ_VAE_DIR"
  python3 downstream/recon_to_geotiff.py --depth "$depth" --csv "$CSV" \
      --data-root .. --out-root "$recon" --min-free-gb 10
  python3 downstream/make_flair_csv.py --base-csv "$CSV" \
      --recon-root "$recon" --out "$csv_out"

  cd "$FLAIR_DIR"
  python3 make_eval_config.py --test-csv "$csv_out" \
      --out-folder "outputs/sanity_$cond" --out "configs/eval-sanity-$cond.yaml"
  flair --conf="configs/eval-sanity-$cond.yaml"

  rm -rf "$recon"
  echo "=== SANITY $cond done =============================================="
}

run_cond blank blank
run_cond q8 8

echo "SANITY_CHECK_ALL_DONE"
