#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"

extra_args=()
if [[ -n "${EPOCHS:-}" ]]; then
  extra_args+=(--epochs "$EPOCHS")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  extra_args+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "${LR:-}" ]]; then
  extra_args+=(--lr "$LR")
fi
if [[ -n "${N_EMBED:-}" ]]; then
  extra_args+=(--n-embed "$N_EMBED")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  extra_args+=(--max-samples "$MAX_SAMPLES")
fi
if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
  extra_args+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi
if [[ -n "${MAX_VAL_SAMPLES:-}" ]]; then
  extra_args+=(--max-val-samples "$MAX_VAL_SAMPLES")
fi
if [[ -n "${MAX_TEST_SAMPLES:-}" ]]; then
  extra_args+=(--max-test-samples "$MAX_TEST_SAMPLES")
fi

read -r -a depths <<< "${DEPTHS:-1 2 4 8 16}"

for depth in "${depths[@]}"; do
  config="configs/flair/stage1/flair-rqvae-8x8x${depth}.yaml"
  output="output/${OUTPUT_PREFIX:-flair-rqvae}-8x8x${depth}"

  echo "==> Training ${output}"
  "$PYTHON_BIN" train_eurosat.py \
    -m "$config" \
    -o "$output" \
    "${extra_args[@]}"
done
