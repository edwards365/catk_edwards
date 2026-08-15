# Experiment Log

Daily records make the BC, Top-K, and CAT-K comparisons auditable. Add one file
per experiment day using `YYYY-MM-DD-day.md` and include:

1. Git commit and exact command.
2. Dataset manifest names and scenario counts.
3. GPU model, count, peak memory, and wall-clock time.
4. Seed, batch size, gradient accumulation, epochs, and optimizer steps.
5. Checkpoint path or external artifact identifier (never commit checkpoints).
6. Open-loop loss/accuracy and closed-loop minADE/WOSAC metrics when available.
7. Failures, deviations from the plan, and the next decision.

Current schedule:

- Tuesday: inventory, fixed subset, and BC baseline freeze.
- Wednesday: BC calibration and stable baseline.
- Thursday: matched Top-K and CAT-K fine-tuning.
- Friday: common validation and rollout inspection.
- Saturday: medium-data run or rental-server expansion decision.
- Sunday: final evaluation, tables, and reproduction summary.

Completed result summaries:

- `smart_mini_results.md`: 1,522/287 SMART-mini comparison.
- `medium_20k_results.md`: 19,611/1,151 A100 comparison and WOSAC-20.
