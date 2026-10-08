# ADR 0006: Recipe-driven experiments with leakage-safe selection

- Status: Accepted
- Date: 2026-09-27

## Context

Allowing Agents to generate arbitrary training code creates security, reproducibility and review risks. Allowing test metrics into search creates data leakage and overfitting.

## Decision

RecipeRegistry is a trusted, immutable Platform component. Agents may only select registered Recipes and hyperparameters inside declared ranges. Preprocessing fits only on train; validation/test are transform-only. Model selection uses validation metrics only. Final test evaluation is Platform-owned and occurs after selection.

## Consequences

- Agent decisions remain auditable and reproducible.
- Test metrics cannot influence ranking, search, Promotion or Agent context.
- New model families require a reviewed Recipe version.
- Budget and policy rejections are persisted through DecisionRecord.