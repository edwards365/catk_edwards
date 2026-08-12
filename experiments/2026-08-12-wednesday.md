# 2026-08-12 Wednesday: Matched Fine-Tuning Complete

## Status

The fixed SMART-mini BC, Top-K, and CAT-K training runs all completed on the
Linux 4x RTX 3090 server. Top-K and CAT-K ran concurrently on one GPU each and
used the same BC initialization, processed dataset, seed, batch size, gradient
accumulation, and epoch count.

| Run | Epoch | Global step | Final train loss | Checkpoint |
|---|---:|---:|---:|---|
| BC | 7 | Pending metadata read | Pending log extraction | `logs/mini_bc_fixed_seed817/runs/2026-08-11_22-18-47/checkpoints/epoch_007.ckpt` |
| Top-K | 3 | 1523 | 2.27297 | `logs/mini_topk_fixed_seed817/runs/2026-08-11_22-43-15/checkpoints/epoch_003.ckpt` |
| CAT-K | 3 | 1523 | 1.33530 | `logs/mini_catk_fixed_seed817/runs/2026-08-11_22-43-15/checkpoints/epoch_003.ckpt` |

Top-K completed at 00:19 and CAT-K at 00:21. Both logs ended with
`run.py DONE!!!`; neither run reported an exception, OOM, or NaN. The
checkpoint files are 32 MB each, while the BC checkpoint is 40 MB.

## Interpretation

The lower CAT-K training loss is not yet an evaluation result because each
method induces a different closed-loop training-state distribution. Model
quality must be compared by evaluating all three checkpoints with the same
open-loop and closed-loop configuration.

## Environment Note

The checkpoint metadata helper first ran in the Conda `base` environment,
which does not contain PyTorch. A later PyTorch import in the `catk` environment
was manually interrupted. These inspection-command failures do not affect the
completed training runs or checkpoint files.

## Next Gate

- [x] Validate that all three checkpoints load.
- [x] Run identical open-loop loss and token accuracy evaluation.
- [x] Run identical eight-rollout closed-loop minADE evaluation.
- [x] Compare BC, Top-K, and CAT-K without WOSAC protobuf generation first.

## Fast Validation Results

The common validation used 20 batches, eight closed-loop rollouts, seed 817,
Top-K probability sampling with `K=48`, and no WOSAC protobuf generation.

| Model | Open-loop accuracy | Open-loop loss | Closed-loop ADE |
|---|---:|---:|---:|
| BC | 0.73969 | 3.02632 | 0.73187 |
| Top-K | 0.64372 | 3.84451 | 1.54520 |
| CAT-K | **0.74374** | 3.14561 | **0.69320** |

CAT-K reduced closed-loop ADE by about 5.3% relative to BC and by about 55.1%
relative to Top-K. Top-K substantially degraded both open-loop and closed-loop
performance in this small-data run. CAT-K slightly improved token accuracy over
BC, although its open-loop cross-entropy was about 3.9% higher.

WOSAC fields were NaN with `scenario_counter=0`, as expected: the fast config
sets `n_batch_wosac_metric=0`. These NaNs are disabled metrics, not failed
rollouts. The next run increases closed-loop rollouts from 8 to 32 on the same
20 batches to test whether the method ordering is stable.

## 32-Rollout Stability Check

| Model | Open-loop accuracy | Open-loop loss | Closed-loop ADE |
|---|---:|---:|---:|
| BC | 0.73981 | 3.02632 | 0.51370 |
| Top-K | 0.64372 | 3.84451 | 1.11559 |
| CAT-K | **0.74374** | 3.14562 | **0.45407** |

All three jobs ended with `run.py DONE!!!`. CAT-K reduced 32-rollout ADE by
about 11.6% relative to BC and 59.3% relative to Top-K, preserving the same
ordering as the eight-rollout check. Increasing the rollout count lowered all
reported minADE values, which is expected because the metric selects the best
sample from a larger set.

The next stage enables WOSAC for 10 scenarios using the existing split
validation TFRecords. This is a pipeline and directional-metric check, not a
claim of full-validation statistical significance.

## WOSAC-10 Pipeline Check

| Model | Internal ADE | WOSAC minADE | Realism | Kinematic | Interactive | Map-based |
|---|---:|---:|---:|---:|---:|---:|
| BC | 0.52713 | 3.04823 | 0.63502 | 0.40561 | 0.64812 | 0.74927 |
| Top-K | 1.19706 | 5.43904 | 0.41148 | 0.28533 | 0.41630 | 0.47735 |
| CAT-K | **0.43947** | **2.33511** | **0.65274** | **0.41658** | **0.66353** | **0.77383** |

Each run evaluated 10 scenarios and ended with `run.py DONE!!!`. CAT-K improved
all reported WOSAC buckets over BC, while Top-K degraded all of them. Relative
to BC, CAT-K reduced WOSAC minADE by about 23.4% and increased the realism meta
metric by about 2.8%. Because 10 scenarios are too few for a robust claim, the
next gate evaluates a fixed prefix of up to 50 available validation scenarios.

## WOSAC-50 Results

| Model | Internal ADE | WOSAC minADE | Realism | Kinematic | Interactive | Map-based |
|---|---:|---:|---:|---:|---:|---:|
| BC | 0.40880 | 2.90008 | 0.63274 | 0.45459 | 0.68090 | 0.67263 |
| Top-K | 0.97641 | 5.47954 | 0.42266 | 0.30521 | 0.45353 | 0.45009 |
| CAT-K | **0.35203** | **2.60756** | **0.64844** | **0.46061** | **0.68125** | **0.71360** |

All runs evaluated 50 scenarios and ended normally. Relative to BC, CAT-K
reduced internal ADE by about 13.9%, reduced WOSAC minADE by about 10.1%, and
increased the realism meta metric by about 2.5%. CAT-K also led every reported
WOSAC bucket. The method ordering is now consistent at 8 rollouts, 32
rollouts, WOSAC-10, and WOSAC-50.

## Storage Gate

The server root filesystem is 99% used, with 22 GB free out of 1.9 TB. The
available source data contains only three training TFRecord shards, already
expanded into all 1,522 processed training scenarios. The current cache also
contains 287 validation and 307 testing scenarios. Medium-scale reproduction
therefore requires additional Waymo shards and a separate data volume.

The planned medium run must use a new mount with at least 500 GB free; 1 TB is
preferred. Raw Waymo files and processed caches must live on that mount. The
existing server remains suitable for code, checkpoints, and GPU training after
the new volume is attached. Do not download or preprocess more data into the
current 22 GB overlay.
