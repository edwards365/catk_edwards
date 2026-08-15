# SMART-mini Medium-20k Reproduction Results

## Scope

- WOMD v1.2.1 scenario data: 19,611 training scenarios and 1,151 validation
  scenarios, processed from 40 training shards and 4 validation shards.
- SMART-mini, 3.4M parameters, seed 817, one NVIDIA A100-PCIE-40GB.
- BC trained for 8 epochs with batch size 4.
- Top-K and CAT-K both started from the same final BC checkpoint and trained
  for 4 epochs with batch size 4.
- Top-K and CAT-K used BF16 mixed precision after FP16 produced NaN logits
  during closed-loop rollout. BC used FP16 mixed precision.
- Results are local fixed-prefix evaluations, not official leaderboard scores.

## Training

| Run | Epochs | Precision | Batch | Wall time |
|---|---:|---|---:|---:|
| BC | 8 | FP16 mixed | 4 | About 3.6 hours |
| Top-K | 4 | BF16 mixed | 4 | 5 h 33 min |
| CAT-K | 4 | BF16 mixed | 4 | 5 h 36 min |

The matched fine-tuning runs each used 4,903 batches per epoch. A 250-batch
BF16 stress test passed for both methods before the full runs were launched.

## Fast Closed-Loop Validation

### Four Rollouts, 20 Batches

| Model | Open accuracy | Open loss | Closed ADE |
|---|---:|---:|---:|
| BC | 0.79568 | **2.55982** | 0.55231 |
| Top-K | 0.65307 | 3.61175 | 0.90488 |
| CAT-K | **0.79928** | 2.82498 | **0.52160** |

### Eight Rollouts, 50 Batches

| Model | Open accuracy | Open loss | Closed ADE |
|---|---:|---:|---:|
| BC | 0.77944 | **2.74388** | 0.44384 |
| Top-K | 0.61983 | 3.90996 | 0.89750 |
| CAT-K | **0.78022** | 2.99963 | **0.42888** |

On the 8-rollout/50-batch check, CAT-K reduced closed-loop ADE by about 3.4%
relative to BC. Top-K increased ADE by about 102%.

## WOSAC-20

Each model used 32 rollouts on the same 20 validation scenarios.

| Model | Internal ADE | WOSAC minADE | Realism | Kinematic | Interactive | Map-based |
|---|---:|---:|---:|---:|---:|---:|
| BC | 0.27328 | **1.73797** | **0.71116** | 0.47079 | 0.71826 | **0.83939** |
| Top-K | 0.62331 | 3.09012 | 0.50198 | 0.31847 | 0.47453 | 0.64215 |
| CAT-K | **0.25530** | 1.79663 | 0.71113 | **0.47400** | **0.72060** | 0.83445 |

Relative to BC, CAT-K reduced internal ADE by about 6.6%, improved the
kinematic bucket by 0.7%, and improved the interactive bucket by 0.3%.
Realism was effectively tied. CAT-K's WOSAC minADE was 3.4% higher and its
map-based score was 0.6% lower, so this 20-scenario sample does not support a
claim that CAT-K wins every official submetric. Top-K was substantially worse
on every reported metric.

## Engineering Findings

- FP16 closed-loop fine-tuning failed at Top-K epoch 0, batch 149 because
  `Categorical` received NaN next-token logits for a batch containing 227
  agents. This was a numerical-range failure, not an OOM or corrupt sample.
- A100-native BF16 eliminated the NaNs while preserving roughly 0.98 batches
  per second. Both full fine-tuning runs then completed without errors.
- Batch size 4 used about 10.1 GiB during the 100-batch rollout stress test and
  kept the effective batch size shared by Top-K and CAT-K.
- Official WOSAC computation is CPU-heavy. WOSAC-20 took roughly 15 minutes
  per model because each validation batch performs 32 rollouts, protobuf
  conversion, TFRecord loading, and TensorFlow metric computation.

## Interpretation

The medium-scale run reproduces the main qualitative SMART-mini result:
CAT-K is stable and improves the internal closed-loop displacement metric over
BC, while naive Top-K fine-tuning degrades strongly. The WOSAC-20 sample is
too small to establish small differences between BC and CAT-K in every
official bucket. A larger fixed-prefix WOSAC evaluation is the next statistical
check if rental time permits.
