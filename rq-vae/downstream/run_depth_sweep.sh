#!/usr/bin/env bash
# Recon -> segment -> mIoU, one depth at a time.
#
# SERVER ONLY. Run inside tmux:  tmux new -s flair_sweep
#
#   ./downstream/run_depth_sweep.sh                    # full 7,050 val set
#   MAX_SAMPLES=500 ./downstream/run_depth_sweep.sh    # fast subset first
#
# One condition at a time, deleting reconstructions after harvesting, so peak
# disk stays ~6 GB instead of the ~36 GB all six conditions would need at
# once. The disk has hit 100% on this box before -- recon_to_geotiff.py also
# refuses to start below --min-free-gb.
set -euo pipefail

RQ_VAE_DIR="${RQ_VAE_DIR:-$HOME/Research-clean/rq-vae}"
FLAIR_DIR="${FLAIR_DIR:-$HOME/Research-clean/FLAIR-1-main}"
SCRATCH="${SCRATCH:-/scratch}"
CSV="${CSV:-downstream/flair-1-paths-val-7050.csv}"
DEPTHS="${DEPTHS:-1 2 4 8 16}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MIN_FREE_GB="${MIN_FREE_GB:-50}"
OUT_ROOT="${OUT_ROOT:-$FLAIR_DIR/outputs}"

cd "$RQ_VAE_DIR"
MAXARG=""
[ -n "$MAX_SAMPLES" ] && MAXARG="--max-samples $MAX_SAMPLES"

run_condition () {
  local cond="$1" depth="$2"
  echo "=== $cond ==================================================="
  local recon="$SCRATCH/flair_recon_$cond"
  local csv_out="$FLAIR_DIR/csv_recon/val-$cond.csv"

  if [ "$depth" = "orig" ]; then
    python3 downstream/make_flair_csv.py --base-csv "$CSV" \
        --recon-root orig --out "$csv_out"
  else
    python3 downstream/recon_to_geotiff.py --depth "$depth" --csv "$CSV" \
        --data-root .. --out-root "$recon" --min-free-gb "$MIN_FREE_GB" $MAXARG
    python3 downstream/make_flair_csv.py --base-csv "$CSV" \
        --recon-root "$recon" --out "$csv_out" --allow-missing 100000
  fi

  sed -e "s|__TEST_CSV__|$csv_out|" -e "s|__OUT__|$OUT_ROOT/$cond|" \
      downstream/configs/eval-recon-template.yaml \
      > "$FLAIR_DIR/configs/eval-$cond.yaml"
  ( cd "$FLAIR_DIR" && flair --conf="configs/eval-$cond.yaml" )

  # delete BEFORE the next condition -- this is the disk guard that matters
  [ "$depth" != "orig" ] && rm -rf "$recon"
  echo "=== $cond done ============================================="
}

run_condition orig  orig      # mIoU_ref  -- also the checkpoint gate (~0.5443)
run_condition blank blank     # mIoU_floor -- NIR+Elevation only, no usable RGB
for d in $DEPTHS; do run_condition "q$d" "$d"; done

python3 downstream/harvest_metrics.py \
    --results-root "$OUT_ROOT" \
    --out downstream/results/downstream_summary.json \
    --quality-table ../hypatia_sim/oec_sim/quality_table.json \
    --depths "$(echo $DEPTHS | tr ' ' ',')"
