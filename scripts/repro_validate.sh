#!/usr/bin/env bash
set -euo pipefail

: "${CATK_CACHE_ROOT:?Set CATK_CACHE_ROOT to the cache containing validation/}"
: "${CATK_CKPT:?Set CATK_CKPT to the checkpoint to evaluate}"

VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TASK_NAME="${TASK_NAME:-repro_val_fast}"

python -m src.run \
  experiment=repro_val_fast \
  trainer=default \
  paths.cache_root="$CATK_CACHE_ROOT" \
  ckpt_path="$CATK_CKPT" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.num_workers="$NUM_WORKERS" \
  task_name="$TASK_NAME"
