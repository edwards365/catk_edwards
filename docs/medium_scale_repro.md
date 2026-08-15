# CATK Medium-Scale Reproduction Plan

This stage originally targeted a deterministic 10,000/500 split and was
expanded on the rental server to 19,611 training and 1,151 validation
scenarios. It follows the upstream CATK sequence: WOMD preprocessing, BC
pre-training, matched Top-K and CAT-K fine-tuning, common closed-loop
validation, then WOSAC evaluation. Measured results are recorded in
`experiments/medium_20k_results.md`.

## Rental Target

Preferred configuration:

- 2x RTX 4090 24 GB, RTX 3090 24 GB, or better CUDA GPUs.
- 1 TB usable NVMe minimum; choose 2 TB if a later full-WOMD run is likely.
- 16 or more CPU cores and 128 GB RAM.
- Ubuntu 22.04, Docker support, and persistent storage independent of the
  instance lifecycle.
- Reliable access to GitHub, the Waymo download source, and W&B if online
  logging is enabled.

The two GPUs let Top-K and CAT-K run concurrently. More GPUs are optional for
this 3.4M-parameter model; storage throughput and data availability matter more
at this scale. Keep at least 30% of the data volume free.

## Storage Layout

Use a dedicated mount such as `/mnt/catk_data`:

```text
/mnt/catk_data/
|-- womd/scenario/{training,validation,testing}/
|-- processed_full/{training,validation,testing,validation_tfrecords_splitted}/
|-- repro_10k/{training,validation}/
|-- logs/
`-- manifests/
```

Do not place WOMD or processed caches on the current 99%-used server overlay.
Checkpoints and manifests should be copied to a second persistent location
after every completed stage.

## Phase 1: Provision and Verify

1. Mount the persistent volume and confirm at least 500 GB free.
2. Install the upstream-compatible Python 3.11, PyTorch 2.4.1, Lightning 2.4.0,
   PyG CUDA 12.1, and Waymo 1.6.4 environment.
3. Apply the PyG `forward` and AMP dtype compatibility commits from this branch.
4. Run a five-batch BC smoke test before downloading the full medium subset.

## Phase 2: Data

1. Download WOMD v1.2.1 scenario shards to the dedicated volume.
2. Process enough training shards to exceed 10,000 scenarios and enough
   validation shards to exceed 500 scenarios.
3. Create deterministic subsets with seed 817 and symlinks:

```bash
python scripts/make_repro_subset.py \
  /mnt/catk_data/processed_full/training \
  /mnt/catk_data/repro_10k/training \
  --count 10000 --seed 817 --method symlink

python scripts/make_repro_subset.py \
  /mnt/catk_data/processed_full/validation \
  /mnt/catk_data/repro_10k/validation \
  --count 500 --seed 817 --method symlink

ln -s /mnt/catk_data/processed_full/validation_tfrecords_splitted \
  /mnt/catk_data/repro_10k/validation_tfrecords_splitted
```

Commit only manifests and aggregate counts, never WOMD files or checkpoints.

## Phase 3: Matched Training

Use SMART-mini, seed 817, and the same cache for all runs:

1. BC: 8 epochs, batch size 4, one GPU.
2. Verify and freeze the BC `last.ckpt`.
3. Top-K: 4 epochs, batch size 1, accumulation 4, GPU 0.
4. CAT-K: 4 epochs, batch size 1, accumulation 4, GPU 1.
5. Require equal optimizer-step counts for Top-K and CAT-K.

Based on the 1,522-scenario measurements, budget about 3 hours for BC and
10-14 hours for concurrent Top-K/CAT-K, plus preprocessing and validation.
These are planning estimates and must be replaced by measured rental-server
throughput after the smoke test.

## Phase 4: Evaluation

1. Common open-loop and 32-rollout validation on 100 scenarios.
2. WOSAC-50 pipeline check.
3. WOSAC-500 if every validation TFRecord is present and storage remains safe.
4. Record accuracy, cross-entropy, internal minADE, WOSAC minADE, realism,
   kinematic, interactive, and map-based metrics.

The primary gate is whether CAT-K consistently improves closed-loop and WOSAC
metrics over BC and Top-K. Training loss alone is not used to rank models.

## Stop Conditions

Stop and inspect before continuing if:

- free disk falls below 30%;
- a run reports OOM, NaN, unequal fine-tuning steps, or a missing checkpoint;
- the selected manifests differ between methods;
- estimated remaining runtime exceeds 36 hours;
- validation TFRecords do not match processed scenario IDs.

## Daily Artifacts

Upload one experiment log per day containing the rental instance specification,
environment versions, manifests, exact commands, wall-clock time, peak memory,
checkpoint identifiers, metrics, failures, and the next decision.
