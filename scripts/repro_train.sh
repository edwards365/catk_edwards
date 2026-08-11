#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "bc" && "$MODE" != "topk" && "$MODE" != "catk" ]]; then
  echo "Usage: CATK_CACHE_ROOT=... [BC_CKPT=...] $0 {bc|topk|catk}"
  exit 2
fi
USER_ARGS=("${@:2}")

: "${CATK_CACHE_ROOT:?Set CATK_CACHE_ROOT to the cache containing training/ and validation/}"

GPUS="${GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATE="${ACCUMULATE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EXPERIMENT="repro_${MODE}_fast"
TASK_NAME="${TASK_NAME:-$EXPERIMENT}"

EXTRA_ARGS=()
if [[ "$MODE" != "bc" ]]; then
  : "${BC_CKPT:?Set BC_CKPT for Top-K or CAT-K fine-tuning}"
  EXTRA_ARGS+=("ckpt_path=$BC_CKPT")
fi

COMMON_ARGS=(
  "experiment=$EXPERIMENT"
  "paths.cache_root=$CATK_CACHE_ROOT"
  "trainer.devices=$GPUS"
  "data.train_batch_size=$BATCH_SIZE"
  "data.val_batch_size=$BATCH_SIZE"
  "data.num_workers=$NUM_WORKERS"
  "trainer.accumulate_grad_batches=$ACCUMULATE"
  "task_name=$TASK_NAME"
)

if [[ "$GPUS" -eq 1 ]]; then
  python -m src.run trainer=default "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" \
    "${USER_ARGS[@]}"
else
  torchrun --standalone --nproc_per_node="$GPUS" -m src.run \
    trainer=ddp "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" "${USER_ARGS[@]}"
fi
