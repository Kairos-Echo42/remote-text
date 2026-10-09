# ADR 0007: Budget Reservation and Final-Test Isolation

- Status: Accepted
- Date: 2026-09-27

## Context

Concurrent Agent submissions can oversubscribe GPU time. Test metrics are also easy to misuse as a search signal even when the intended selection metric is validation.

## Decision

Admission reserves each Job's hard `max_job_seconds` against total and GPU budgets inside the same database transaction that creates the reservation. Completion reconciles actual usage and releases unused reservation. `preferred_gpu` releases its GPU reservation immediately on CPU fallback.

Test metrics remain platform-owned final evaluation metadata. They are produced only after selection, stored separately from validation metrics, displayed only to human reviewers, and hidden or rejected for Agent API keys.

## Consequences

- Concurrent submissions cannot oversell total or GPU budgets.
- Budget rejection, validation selection and final evaluation are fully auditable.
- Test metrics cannot influence ranking, Promotion, later searches or Agent context.
