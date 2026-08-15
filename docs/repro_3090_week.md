# CATK Fast Reproduction on 4070 and 4x3090

This setup targets a method-level reproduction under limited storage. It keeps
the upstream configs unchanged and adds four experiment overrides:

- `repro_bc_fast`: SMART-mini behavior-cloning baseline.
- `repro_topk_fast`: four-epoch Top-K closed-loop fine-tuning.
- `repro_catk_fast`: four-epoch CAT-K fine-tuning from the same BC checkpoint.
- `repro_val_fast`: common open-loop and eight-rollout closed-loop validation.

## 1. Inventory

```bash
bash scripts/repro_inventory.sh /path/to/cache
```

Keep at least 30% free space on the active training filesystem.

## 2. Deterministic subsets

Use separate cache roots whose `training` and `validation` directories contain
only links to the selected scenarios. The manifest is written next to the
subset directory so `MultiDataset` will not try to load it as a sample.

```bash
python scripts/make_repro_subset.py /cache/full/training /cache/repro/training \
  --count 5000 --seed 817 --method symlink
python scripts/make_repro_subset.py /cache/full/validation /cache/repro/validation \
  --count 500 --seed 817 --method symlink
```

## 3. BC baseline

Single-GPU smoke test:

```bash
CATK_CACHE_ROOT=/cache/repro GPUS=1 BATCH_SIZE=1 \
  TASK_NAME=mini_bc_smoke bash scripts/repro_train.sh bc
```

Four-GPU run:

```bash
CATK_CACHE_ROOT=/cache/repro GPUS=4 BATCH_SIZE=1 ACCUMULATE=2 \
  TASK_NAME=mini_bc_fixed bash scripts/repro_train.sh bc
```

## 4. Matched Top-K and CAT-K runs

Both runs must use the same `BC_CKPT`, cache root, seed, batch size, gradient
accumulation, and epoch count.

```bash
CATK_CACHE_ROOT=/cache/repro BC_CKPT=/logs/mini_bc_fixed/checkpoints/last.ckpt \
  GPUS=4 BATCH_SIZE=1 ACCUMULATE=4 TASK_NAME=mini_topk_fixed \
  bash scripts/repro_train.sh topk

CATK_CACHE_ROOT=/cache/repro BC_CKPT=/logs/mini_bc_fixed/checkpoints/last.ckpt \
  GPUS=4 BATCH_SIZE=1 ACCUMULATE=4 TASK_NAME=mini_catk_fixed \
  bash scripts/repro_train.sh catk
```

Top-K uses `topk_prob`, `K=32`, and temperature `1.0`. CAT-K uses
`topk_prob_sampled_with_dist`, `K=32`, and temperature `1e-5`.

## 5. Common validation

This validation computes open-loop loss/accuracy and closed-loop minADE. It
does not open validation TFRecords because WOSAC metrics and videos are off.

```bash
CATK_CACHE_ROOT=/cache/repro CATK_CKPT=/path/to/model.ckpt \
  TASK_NAME=val_bc bash scripts/repro_validate.sh
```

Run it for BC, Top-K, and CAT-K checkpoints. Only after the ordering is stable
should `n_rollout_closed_val` be increased from 8 to 32 and selected validation
TFRecords be prepared for official WOSAC metrics.

## Weekly gates

1. BC loss decreases and its checkpoint reloads.
2. Top-K and CAT-K each complete a 50-step smoke run without OOM or NaN.
3. Matched runs complete with equal optimizer-step counts.
4. All three checkpoints are evaluated with the same validation config.
5. Rent larger storage only if the medium subset cannot leave 30% free space,
   measured training ETA exceeds 36 hours, or full WOSAC evaluation is needed.
