# ADR 0005: First-class TrainingJobs with an independent Trainer

- Status: Accepted
- Date: 2026-09-27

## Context

Training can run for minutes and must survive Worker restarts, budget admission, cancellation and GPU queueing. A long-running Capability that directly owns training would mix DAG execution with compute lifecycle.

## Decision

DatasetVersion, Experiment, TrainingJob, TrainingAttempt, Checkpoint and ModelVersion are first-class resources. A separate Trainer service claims Jobs, launches isolated PyTorch/sklearn containers and monitors them. Agent capabilities submit, wait and inspect Jobs but never run training in the Agent process.

## Consequences

- Training state is persisted independently of DAG Worker leases.
- Agents can be interrupted without stopping the training container.
- GPU concurrency and budget reservations are controlled by the Platform.
- A dedicated optional Compose profile and CUDA image are required.