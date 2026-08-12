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
