# AgentForge Training Experiments

## Core principle

- Agents make experiment decisions.
- The Platform enforces Recipes, budgets, security and audit policy.
- Trainer manages container lifecycle.
- PyTorch/sklearn perform the actual training computation.

## Resources

`Dataset` → `DatasetVersion` → `Experiment` → `TrainingJob` → `TrainingAttempt` → `Checkpoint` → `ModelVersion`.

Each Experiment has a trusted BaselineStrategy. DummyClassifier, DummyRegressor and image-prior baselines are built in. Specialized baseline strategies can only be registered by the platform.

Selection is deterministic: Baseline first, registered Recipe candidates in round one, Top-K refinement in later rounds, then `SelectionPolicy`. If no candidate reaches `minimum_improvement`, the Baseline is retained and `baseline_retained` is recorded.

## Leakage prevention

- Split is fixed before training.
- Imputers, encoders and scalers fit on train only.
- Validation/test run transform only.
- Image augmentation is train-only.
- Selection uses validation metrics only.
- Final test evaluation runs after selection and is stored as metadata for human review.

## Budget accounting

`ExperimentBudget` limits jobs, rounds, per-job time, total time and GPU time. Admission atomically reserves the Job hard limit. Completion releases unused reservation and records actual consumption. `required_gpu`, `preferred_gpu` and `cpu_only` are supported.

The ledger stores `reserved_total_seconds`, `consumed_total_seconds`, `reserved_gpu_seconds` and `consumed_gpu_seconds`. `preferred_gpu` releases the GPU reservation immediately when it falls back to CPU. Final evaluation remains platform-owned and is not part of Agent model selection.

## Recipes

RecipeRegistry is trusted and immutable. It defines allowed hyperparameters, ranges, resources and run limits. Agents cannot create or register Recipes.

## Reproducibility and lineage

TrainingAttempt stores the DatasetVersion/split checksums, Recipe and config checksums, random seed, runtime versions, Docker digest, Git commit when available, and hardware metadata.

Artifact records own SHA-256 checksums. Lineage edges connect input/output Artifact nodes and record operation, time and actor. Model Bundles contain an internal `checksums.json` and an external zip SHA-256.

Checkpoints record epoch/step, validation metric, Artifact ID and checksum. v0.2 does not resume training from a Checkpoint; retries start from a fresh Attempt. Model Bundles contain `model.pt`, `preprocessor.joblib`, `schema.json`, `metrics.json`, `training-config.json`, `reproducibility.json`, `lineage.json` and `checksums.json`.

## API and review boundary

- `GET /api/v1/workspaces/{workspace_id}/training/baselines` returns trusted baselines.
- `GET /api/v1/experiments/{experiment_id}/leaderboard` returns validation-only ranking.
- `GET /api/v1/experiments/{experiment_id}/report` returns budget, selection reason and the human final-test summary.
- Human sessions may inspect test metrics; Agent API keys receive a hidden or rejected response.

## Cancellation

Docker receives SIGTERM with a grace period, then SIGKILL. The training entrypoint flushes metrics and a non-resumable checkpoint on SIGTERM. v0.2 does not resume from checkpoints.

## Boundaries

No pretrained models, transfer learning, Hugging Face, LLM fine-tuning, multi-GPU, distributed training, Kubernetes, online inference, automatic Production promotion, arbitrary training code generation or complex AutoML.

The Trainer defaults to `python:3.12-slim`; CUDA runtime libraries come from the PyTorch wheels, so Ubuntu `apt` repositories are not required. When Docker Hub is not directly reachable, set `AGENTFORGE_TRAINER_BASE_IMAGE` to a trusted Python 3.12 mirror.
