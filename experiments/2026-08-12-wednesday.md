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

- Validate that all three checkpoints load.
- Run identical open-loop loss and token accuracy evaluation.
- Run identical eight-rollout closed-loop minADE evaluation.
- Compare BC, Top-K, and CAT-K without WOSAC protobuf generation first.
