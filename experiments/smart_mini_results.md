# SMART-mini Reproduction Results

## Scope

- 1,522 processed training scenarios and 287 validation scenarios.
- SMART-mini, 3.4M trainable parameters, seed 817.
- BC for 8 epochs; matched Top-K and CAT-K fine-tuning for 4 epochs.
- Top-K and CAT-K used the same BC checkpoint and each completed 1,523
  optimizer steps.
- Local WOSAC results are method-level evidence on fixed prefixes, not official
  leaderboard submissions.

## Training

| Run | Final epoch | Final train loss |
|---|---:|---:|
| BC | 7 | Not extracted |
| Top-K | 3 | 2.27297 |
| CAT-K | 3 | 1.33530 |

## Common 32-Rollout Validation

| Model | Open-loop accuracy | Open-loop loss | Closed-loop ADE |
|---|---:|---:|---:|
| BC | 0.73981 | **3.02632** | 0.51370 |
| Top-K | 0.64372 | 3.84451 | 1.11559 |
| CAT-K | **0.74374** | 3.14562 | **0.45407** |

## WOSAC-50

| Model | Internal ADE | WOSAC minADE | Realism | Kinematic | Interactive | Map-based |
|---|---:|---:|---:|---:|---:|---:|
| BC | 0.40880 | 2.90008 | 0.63274 | 0.45459 | 0.68090 | 0.67263 |
| Top-K | 0.97641 | 5.47954 | 0.42266 | 0.30521 | 0.45353 | 0.45009 |
| CAT-K | **0.35203** | **2.60756** | **0.64844** | **0.46061** | **0.68125** | **0.71360** |

Relative to BC, CAT-K reduced 32-rollout ADE by about 11.6%, WOSAC-50
internal ADE by about 13.9%, and WOSAC minADE by about 10.1%. Its realism meta
metric improved by about 2.5%. Top-K degraded every reported closed-loop and
WOSAC metric. The CAT-K ordering was consistent in 8-rollout, 32-rollout,
WOSAC-10, and WOSAC-50 evaluations.

## Engineering Findings

- Current PyG requires `BaseTransform.forward`; the upstream target builders
  implemented `__call__`.
- Mixed precision requires agent-token buffers to inherit the embedding output
  dtype rather than the float32 input-position dtype.
- The existing server has only 22 GB free on a 99%-used root filesystem, so
  medium-scale data must use a new persistent volume or rented server.

## Next Experiment

Repeat the matched comparison on deterministic 10,000/500 training/validation
subsets stored on a dedicated volume, followed by 32-rollout and WOSAC-50/500
evaluation.
