#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="${1:-${CATK_CACHE_ROOT:-}}"

echo "== GPU =="
nvidia-smi

echo "== Filesystems =="
df -h

echo "== Python =="
python --version
python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.version.cuda); print("cuda_available", torch.cuda.is_available())'

if [[ -n "$CACHE_ROOT" ]]; then
  echo "== CATK cache =="
  du -sh "$CACHE_ROOT" || true
  for split in training validation testing validation_tfrecords_splitted; do
    path="$CACHE_ROOT/$split"
    if [[ -d "$path" ]]; then
      size=$(du -sh "$path" | cut -f1)
      count=$(find "$path" -maxdepth 1 -type f | wc -l)
      echo "$split: size=$size files=$count"
    fi
  done
fi
