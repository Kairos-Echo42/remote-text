from __future__ import annotations

import asyncio
import hashlib
import io
import json
import uuid
import zipfile
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from agentforge.db import get_session_factory
from agentforge.models import (
    Artifact,
    Dataset,
    DatasetVersion,
    Experiment,
    ModelVersion,
    TrainingAttempt,
    TrainingEvent,
    TrainingJob,
)
from agentforge.storage.artifacts import LocalArtifactStore
from agentforge.training.audit import (
    PolicyViolation,
    ensure_lineage_node,
    link_lineage,
    record_decision,
    record_rejection,
)
from agentforge.training.baselines import BaselineRegistry, build_baseline_registry
from agentforge.training.budget import BudgetExceededError, reserve_job_budget
from agentforge.training.recipes import RecipeRegistry, get_recipe_registry
from agentforge.training.types import (
    DatasetKind,
    DecisionRecordData,
    DecisionResult,
    ExperimentBudget,
    ExperimentSearchStrategy,
    ExperimentStatus,
    ModelStage,
    ObjectiveDirection,
    SelectionPolicy,
    TaskType,
    TrainingJobSpec,
    TrainingJobStatus,
    stable_config_checksum,
)


class TrainingService:
    def __init__(
        self,
        recipe_registry: RecipeRegistry | None = None,
        baseline_registry: BaselineRegistry | None = None,
    ):
        self.recipes = recipe_registry or get_recipe_registry()
        self.baselines = baseline_registry or build_baseline_registry()

    async def create_experiment(
        self,
        session: AsyncSession,
        *,
        workspace_id: uuid.UUID,
        dataset_version_id: uuid.UUID,
        name: str,
        baseline_strategy_id: str | None,
        budget: ExperimentBudget,
        search_strategy: dict[str, Any] | None = None,
        selection_policy: SelectionPolicy | None = None,
        created_by: uuid.UUID | None = None,
        run_id: uuid.UUID | None = None,
    ) -> Experiment:
        dataset_version = await session.get(DatasetVersion, dataset_version_id)
        if dataset_version is None:
            raise ValueError("dataset version not found")
        dataset = await session.get(Dataset, dataset_version.dataset_id)
        if dataset is None or dataset.workspace_id != workspace_id:
            raise ValueError("dataset version not found in workspace")
        if dataset_version.status != "ready":
            raise ValueError("dataset version is not ready")

        baseline = (
            self.baselines.get(baseline_strategy_id)
            if baseline_strategy_id
            else self.baselines.resolve(dataset_kind=DatasetKind(dataset.kind), task_type=TaskType(dataset.task_type))
        )
        if not baseline.supports(dataset_kind=DatasetKind(dataset.kind), task_type=TaskType(dataset.task_type)):
            await record_rejection(
                session,
                workspace_id=workspace_id,
                action="experiment.create",
                reason_code="baseline_dataset_mismatch",
                reason="baseline strategy does not support this dataset",
                actor="platform",
                request={
                    "dataset_version_id": str(dataset_version_id),
                    "baseline_strategy_id": baseline.spec.qualified_id,
                },
            )
            await session.commit()
            raise PolicyViolation("baseline_dataset_mismatch", "baseline strategy does not support this dataset")
        baseline_recipe = self.recipes.get(baseline.build_recipe())
        policy = selection_policy or SelectionPolicy(
            validation_metric=baseline_recipe.default_metric,
            direction=baseline_recipe.default_direction,
        )
        if policy.validation_metric.startswith("test_") or policy.validation_metric == "test":
            await record_rejection(
                session,
                workspace_id=workspace_id,
                action="experiment.create",
                reason_code="test_metric_selection_forbidden",
                reason="test metrics cannot drive selection",
                actor="platform",
                request={"selection_policy": policy.model_dump(mode="json")},
            )
            await session.commit()
            raise PolicyViolation("test_metric_selection_forbidden", "test metrics cannot drive selection")
        strategy_data = dict(search_strategy or {})
        strategy_data["baseline_strategy_id"] = baseline.spec.qualified_id
        strategy = ExperimentSearchStrategy(**strategy_data)
        for allowed_recipe in strategy.round_one_recipes:
            registered_ids = {
                value for recipe in self.recipes.list() for value in (recipe.qualified_id, recipe.recipe_id)
            }
            if allowed_recipe not in registered_ids:
                await record_rejection(
                    session,
                    workspace_id=workspace_id,
                    action="experiment.create",
                    reason_code="search_strategy_recipe_invalid",
                    reason=f"round-one Recipe is not registered: {allowed_recipe}",
                    actor="platform",
                    request={"round_one_recipes": strategy.round_one_recipes},
                )
                await session.commit()
                raise PolicyViolation(
                    "search_strategy_recipe_invalid",
                    f"round-one Recipe is not registered: {allowed_recipe}",
                )
        experiment = Experiment(
            workspace_id=workspace_id,
            dataset_version_id=dataset_version.id,
            name=name,
            objective_metric=policy.validation_metric,
            objective_direction=policy.direction.value,
            status=ExperimentStatus.RUNNING.value,
            created_by=created_by,
            run_id=run_id,
            baseline_strategy={
                **baseline.spec.model_dump(mode="json"),
                "qualified_id": baseline.spec.qualified_id,
            },
            search_strategy=strategy.model_dump(mode="json"),
            budget=budget.model_dump(mode="json"),
            selection_policy=policy.model_dump(mode="json"),
            started_at=datetime.now(UTC),
        )
        session.add(experiment)
        await session.flush()
        try:
            baseline_job = await self.submit_job(
                session,
                workspace_id=workspace_id,
                experiment=experiment,
                job_spec=TrainingJobSpec(
                    recipe_id=baseline.build_recipe(),
                    dataset_version_id=dataset_version.id,
                    target_column=dataset_version.target_column,
                    parameters={},
                    device_policy="cpu_only",
                    max_job_seconds=min(
                        budget.max_job_seconds,
                        baseline_recipe.resources.max_job_seconds,
                    ),
                    idempotency_key=f"baseline:{experiment.id}",
                ),
                job_kind="baseline",
                submitted_by=created_by,
            )
        except PolicyViolation as exc:
            experiment.status = ExperimentStatus.FAILED.value
            experiment.error = exc.reason
            experiment.finished_at = datetime.now(UTC)
            await session.commit()
            raise
        experiment.baseline_job_id = baseline_job.id
        await session.flush()
        await record_decision(
            session,
            workspace_id=workspace_id,
            experiment_id=experiment.id,
            data=DecisionRecordData(
                action="experiment.create",
                result=DecisionResult.ACCEPTED,
                reason_code="baseline_bound",
                reason=f"Experiment bound to {baseline.spec.qualified_id}",
                actor="platform",
                phase="baseline",
                run_id=run_id,
                budget_snapshot=budget.model_dump(mode="json"),
            ),
        )
        return experiment

    async def submit_batch(
        self,
        session: AsyncSession,
        *,
        workspace_id: uuid.UUID,
        experiment_id: uuid.UUID,
        jobs: list[TrainingJobSpec],
        round_number: int,
        submitted_by: uuid.UUID | None = None,
    ) -> list[TrainingJob]:
        experiment = await session.get(Experiment, experiment_id)
        if experiment is None or experiment.workspace_id != workspace_id:
            raise ValueError("experiment not found")
        strategy = ExperimentSearchStrategy.model_validate(experiment.search_strategy)
        budget = ExperimentBudget.model_validate(experiment.budget)
        max_rounds = min(strategy.refinement_rounds + 1, budget.max_rounds)
        if round_number > max_rounds:
            await record_rejection(
                session,
                workspace_id=workspace_id,
                experiment_id=experiment.id,
                action="training.submit_batch",
                reason_code="max_rounds_exceeded",
                reason=f"round {round_number} exceeds experiment round budget",
                actor="platform",
                request={"round_number": round_number, "max_rounds": max_rounds},
            )
            await session.commit()
            raise PolicyViolation("max_rounds_exceeded", "experiment round budget exceeded")
        max_jobs = strategy.max_round_one_jobs if round_number == 1 else strategy.top_k
        existing_round_jobs = (
            await session.scalar(
                select(func.count(TrainingJob.id)).where(
                    TrainingJob.experiment_id == experiment.id,
                    TrainingJob.round_number == round_number,
                    TrainingJob.job_kind.in_(["candidate", "refinement"]),
                )
            )
            or 0
        )
        if int(existing_round_jobs) + len(jobs) > max_jobs:
            await record_rejection(
                session,
                workspace_id=workspace_id,
                experiment_id=experiment.id,
                action="training.submit_batch",
                reason_code="round_job_limit_exceeded",
                reason=f"round {round_number} accepts at most {max_jobs} jobs in total",
                actor="platform",
                request={
                    "round_number": round_number,
                    "existing_jobs": int(existing_round_jobs),
                    "job_count": len(jobs),
                    "max_jobs": max_jobs,
                },
            )
            await session.commit()
            raise PolicyViolation("round_job_limit_exceeded", "round job limit exceeded")
        created: list[TrainingJob] = []
        experiment.current_round = max(experiment.current_round, round_number)
        for job_spec in jobs:
            created.append(
                await self.submit_job(
                    session,
                    workspace_id=workspace_id,
                    experiment=experiment,
                    job_spec=job_spec.model_copy(update={"round_number": round_number}),
                    job_kind="baseline" if round_number == 0 else "candidate",
                    submitted_by=submitted_by,
                )
            )
        return created

    async def submit_job(
        self,
        session: AsyncSession,
        *,
        workspace_id: uuid.UUID,
        experiment: Experiment,
        job_spec: TrainingJobSpec,
        job_kind: str,
        submitted_by: uuid.UUID | None = None,
    ) -> TrainingJob:
        dataset_version = await session.scalar(
            select(DatasetVersion)
            .join(Dataset, DatasetVersion.dataset_id == Dataset.id)
            .where(
                DatasetVersion.id == job_spec.dataset_version_id,
                Dataset.workspace_id == workspace_id,
            )
        )
        if (
            dataset_version is None
            or dataset_version.status != "ready"
            or dataset_version.id != experiment.dataset_version_id
        ):
            await record_rejection(
                session,
                workspace_id=workspace_id,
                experiment_id=experiment.id,
                action="training.submit_job",
                reason_code="dataset_version_mismatch",
                reason="jobs must use the immutable DatasetVersion bound to the experiment",
                actor="platform",
                request=job_spec.model_dump(mode="json"),
            )
            await session.commit()
            raise PolicyViolation(
                "dataset_version_mismatch", "job dataset version does not match the experiment"
            )
        try:
            recipe, parameters = self.recipes.validate_job(job_spec)
        except ValueError as exc:
            reason_code = (
                "recipe_not_registered"
                if "not registered" in str(exc)
                else (
                    "max_job_seconds_exceeded"
                    if "max_job_seconds" in str(exc)
                    else "recipe_parameter_invalid"
                )
            )
            await record_rejection(
                session,
                workspace_id=workspace_id,
                experiment_id=experiment.id,
                action="training.submit_job",
                reason_code=reason_code,
                reason=str(exc),
                actor="platform",
                request=job_spec.model_dump(mode="json"),
            )
            await session.commit()
            raise PolicyViolation(reason_code, str(exc)) from exc
        profile = dataset_version.profile or {}
        dataset_kind = str(profile.get("dataset_kind") or "")
        task_type = str(profile.get("task_type") or "")
        if dataset_kind not in {item.value for item in recipe.dataset_kinds} or task_type not in {
            item.value for item in recipe.task_types
        }:
            await record_rejection(
                session,
                workspace_id=workspace_id,
                experiment_id=experiment.id,
                action="training.submit_job",
                reason_code="recipe_dataset_mismatch",
                reason="Recipe does not support this DatasetVersion task type",
                actor="platform",
                request=job_spec.model_dump(mode="json"),
            )
            await session.commit()
            raise PolicyViolation("recipe_dataset_mismatch", "recipe does not support this dataset")
        if (
            dataset_version.target_column
            and job_spec.target_column
            and job_spec.target_column != dataset_version.target_column
        ):
            await record_rejection(
                session,
                workspace_id=workspace_id,
                experiment_id=experiment.id,
                action="training.submit_job",
                reason_code="target_column_immutable",
                reason="target_column is part of the immutable DatasetVersion",
                actor="platform",
                request=job_spec.model_dump(mode="json"),
            )
            await session.commit()
            raise PolicyViolation("target_column_immutable", "target column cannot be changed")
        if not recipe.resources.supports_gpu and job_spec.device_policy != "cpu_only":
            await record_rejection(
                session,
                workspace_id=workspace_id,
                experiment_id=experiment.id,
                action="training.submit_job",
                reason_code="recipe_device_policy_invalid",
                reason="Recipe does not support GPU execution",
                actor="platform",
                request=job_spec.model_dump(mode="json"),
            )
            await session.commit()
            raise PolicyViolation("recipe_device_policy_invalid", "recipe is CPU-only")
        strategy = ExperimentSearchStrategy.model_validate(experiment.search_strategy)
        if (
            job_kind == "candidate"
            and
            job_spec.round_number == 1
            and strategy.round_one_recipes
            and not {recipe.qualified_id, recipe.recipe_id}.intersection(strategy.round_one_recipes)
        ):
            await record_rejection(
                session,
                workspace_id=workspace_id,
                experiment_id=experiment.id,
                action="training.submit_job",
                reason_code="recipe_not_allowed_by_strategy",
                reason="Recipe is not allowed by the experiment round-one search strategy",
                actor="platform",
                request=job_spec.model_dump(mode="json"),
            )
            await session.commit()
            raise PolicyViolation("recipe_not_allowed_by_strategy", "recipe is outside the search strategy")
        if job_kind == "candidate" and job_spec.round_number > 1:
            if job_spec.source_job_id is None:
                await record_rejection(
                    session,
                    workspace_id=workspace_id,
                    experiment_id=experiment.id,
                    action="training.submit_job",
                    reason_code="refinement_source_required",
                    reason="refinement jobs must reference a Top-K job from the previous round",
                    actor="platform",
                    request=job_spec.model_dump(mode="json"),
                )
                await session.commit()
                raise PolicyViolation("refinement_source_required", "refinement source job is required")
            previous_jobs = list(
                (
                    await session.scalars(
                        select(TrainingJob).where(
                            TrainingJob.experiment_id == experiment.id,
                            TrainingJob.status == "succeeded",
                            TrainingJob.job_kind.in_(["candidate", "refinement"]),
                            TrainingJob.round_number == job_spec.round_number - 1,
                        )
                    )
                ).all()
            )
            scored_previous: list[tuple[TrainingJob, float]] = []
            for previous in previous_jobs:
                value = await self._metric_for_job(session, previous.id, experiment.objective_metric)
                if value is not None:
                    scored_previous.append((previous, value))
            selection_policy = SelectionPolicy.model_validate(experiment.selection_policy)
            scored_previous.sort(key=lambda item: self._score_sort_key(item[0], item[1], selection_policy))
            top_k_ids = {job.id for job, _ in scored_previous[: strategy.top_k]}
            if job_spec.source_job_id not in top_k_ids:
                await record_rejection(
                    session,
                    workspace_id=workspace_id,
                    experiment_id=experiment.id,
                    action="training.submit_job",
                    reason_code="refinement_source_not_top_k",
                    reason="refinement source must be one of the previous-round Top-K jobs",
                    actor="platform",
                    request=job_spec.model_dump(mode="json"),
                )
                await session.commit()
                raise PolicyViolation("refinement_source_not_top_k", "refinement source is not Top-K")
        if job_spec.max_job_seconds > min(
            recipe.resources.max_job_seconds,
            ExperimentBudget.model_validate(experiment.budget).max_job_seconds,
        ):
            await record_rejection(
                session,
                workspace_id=workspace_id,
                experiment_id=experiment.id,
                action="training.submit_job",
                reason_code="max_job_seconds_exceeded",
                reason="job max_job_seconds exceeds recipe or experiment hard limit",
                actor="platform",
                request=job_spec.model_dump(mode="json"),
            )
            await session.commit()
            raise PolicyViolation("max_job_seconds_exceeded", "job exceeds hard time limit")
        idempotency_key = job_spec.idempotency_key or stable_config_checksum(
            {
                "dataset": str(job_spec.dataset_version_id),
                "recipe": recipe.qualified_id,
                "parameters": parameters,
                "round": job_spec.round_number,
                "random_seed": job_spec.random_seed,
                "max_epochs": job_spec.max_epochs,
                "batch_size": job_spec.batch_size,
                "device_policy": job_spec.device_policy.value,
            }
        )
        existing = await session.scalar(
            select(TrainingJob).where(
                TrainingJob.experiment_id == experiment.id,
                TrainingJob.idempotency_key == idempotency_key,
            )
        )
        if existing is not None:
            return existing
        validated_spec = job_spec.model_copy(
            update={
                "recipe_id": recipe.qualified_id,
                "target_column": dataset_version.target_column,
                "parameters": parameters,
                "idempotency_key": idempotency_key,
            }
        )
        job_spec_payload = validated_spec.model_dump(mode="json")
        job_spec_payload["objective_metric"] = experiment.objective_metric
        job_spec_payload["objective_direction"] = experiment.objective_direction
        job_spec_payload["recipe_resources"] = recipe.resources.model_dump(mode="json")
        config_checksum = stable_config_checksum(job_spec_payload)
        queue = "cpu" if job_spec.device_policy == "cpu_only" or not recipe.resources.supports_gpu else "gpu"
        job = TrainingJob(
            workspace_id=workspace_id,
            experiment_id=experiment.id,
            dataset_version_id=job_spec.dataset_version_id,
            recipe_id=recipe.qualified_id,
            recipe_version=recipe.version,
            recipe_checksum=recipe.checksum,
            round_number=job_spec.round_number,
            job_kind=job_kind,
            status=TrainingJobStatus.QUEUED.value,
            device_policy=job_spec.device_policy,
            idempotency_key=idempotency_key,
            spec=job_spec_payload,
            config_checksum=config_checksum,
            queue_name=queue,
            submitted_by=submitted_by,
        )
        session.add(job)
        await session.flush()
        try:
            await reserve_job_budget(session, job=job)
        except BudgetExceededError as exc:
            job.status = TrainingJobStatus.FAILED.value
            job.error = str(exc)
            await record_rejection(
                session,
                workspace_id=workspace_id,
                experiment_id=experiment.id,
                job_id=job.id,
                action="training.submit_job",
                reason_code="budget_exceeded",
                reason=str(exc),
                actor="platform",
                request=validated_spec.model_dump(mode="json"),
                budget_snapshot=self._budget_snapshot(experiment),
                config_checksum=config_checksum,
            )
            await session.commit()
            raise PolicyViolation("budget_exceeded", str(exc)) from exc
        await self._append_job_event(session, job, "training.job.queued", {"queue": queue})
        await record_decision(
            session,
            workspace_id=workspace_id,
            experiment_id=experiment.id,
            job_id=job.id,
            data=DecisionRecordData(
                action="training.submit_job",
                result=DecisionResult.ACCEPTED,
                reason_code="job_queued",
                reason=f"{job_kind} job accepted into the {queue} queue",
                actor="platform" if job_kind in {"baseline", "final_test"} else "agent",
                phase="submission",
                run_id=experiment.run_id,
                request=validated_spec.model_dump(mode="json"),
                budget_snapshot=self._budget_snapshot(experiment),
                config_checksum=config_checksum,
            ),
        )
        dataset_node = await ensure_lineage_node(
            session,
            workspace_id=workspace_id,
            node_type="dataset_version",
            ref_id=job.dataset_version_id,
        )
        experiment_node = await ensure_lineage_node(
            session,
            workspace_id=workspace_id,
            node_type="experiment",
            ref_id=experiment.id,
        )
        job_node = await ensure_lineage_node(
            session,
            workspace_id=workspace_id,
            node_type="training_job",
            ref_id=job.id,
        )
        await link_lineage(
            session,
            workspace_id=workspace_id,
            input_node=dataset_node,
            output_node=experiment_node,
            operation="bind_dataset",
            actor_type="platform",
            actor_id=str(submitted_by) if submitted_by else None,
        )
        await link_lineage(
            session,
            workspace_id=workspace_id,
            input_node=experiment_node,
            output_node=job_node,
            operation="submit_job",
            actor_type="agent" if job_kind != "baseline" else "platform",
            actor_id=str(submitted_by) if submitted_by else None,
            metadata={"job_kind": job_kind, "round": job.round_number},
        )
        return job

    async def await_jobs(
        self,
        *,
        job_ids: list[uuid.UUID],
        timeout_seconds: float,
        poll_seconds: float = 1.0,
        workspace_id: uuid.UUID | None = None,
    ) -> list[TrainingJob]:
        if not job_ids:
            return []
        factory = get_session_factory()
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        terminal = {"succeeded", "failed", "cancelled"}
        while True:
            async with factory() as session:
                query = select(TrainingJob).where(TrainingJob.id.in_(job_ids))
                if workspace_id is not None:
                    query = query.where(TrainingJob.workspace_id == workspace_id)
                jobs = list((await session.scalars(query)).all())
            if len(jobs) != len(job_ids):
                raise ValueError("training job not found in workspace")
            if jobs and all(job.status in terminal for job in jobs):
                return jobs
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("training jobs did not finish before timeout")
            await asyncio.sleep(poll_seconds)

    async def validation_results(
        self,
        session: AsyncSession,
        job_ids: list[uuid.UUID],
        *,
        workspace_id: uuid.UUID | None = None,
    ) -> list[dict]:
        query = select(TrainingJob).where(TrainingJob.id.in_(job_ids))
        if workspace_id is not None:
            query = query.where(TrainingJob.workspace_id == workspace_id)
        jobs = list((await session.scalars(query)).all())
        if len(jobs) != len(job_ids):
            raise ValueError("training job not found in workspace")
        results = []
        for job in jobs:
            attempt = await self._latest_attempt(session, job.id)
            metrics = dict((attempt.metrics or {}).get("validation", {})) if attempt else {}
            results.append(
                {
                    "job_id": str(job.id),
                    "job_kind": job.job_kind,
                    "recipe_id": job.recipe_id,
                    "status": job.status,
                    "validation": metrics,
                    "config_checksum": job.config_checksum,
                    "round": job.round_number,
                    "actual_total_seconds": job.actual_total_seconds,
                    "actual_gpu_seconds": job.actual_gpu_seconds,
                    "bundle_artifact_id": (job.spec or {}).get("bundle_artifact_id"),
                }
            )
        return results

    async def select_best(
        self,
        session: AsyncSession,
        *,
        experiment_id: uuid.UUID,
        workspace_id: uuid.UUID,
        actor: str = "agent",
    ) -> TrainingJob:
        experiment = await session.get(Experiment, experiment_id, with_for_update=True)
        if experiment is None or experiment.workspace_id != workspace_id:
            raise ValueError("experiment not found")
        if experiment.best_job_id is not None and experiment.status in {"evaluating", "completed"}:
            existing = await session.get(TrainingJob, experiment.best_job_id)
            if existing is None:
                raise ValueError("selected training job not found")
            return existing
        policy = SelectionPolicy.model_validate(experiment.selection_policy)
        baseline = await session.get(TrainingJob, experiment.baseline_job_id)
        if baseline is None or baseline.status != "succeeded":
            raise PolicyViolation("baseline_not_ready", "baseline must succeed before selection")
        baseline_metric = await self._metric_for_job(session, baseline.id, policy.validation_metric)
        if baseline_metric is None:
            raise PolicyViolation("baseline_metric_missing", "baseline validation metric is missing")

        unfinished = await session.scalar(
            select(func.count(TrainingJob.id)).where(
                TrainingJob.experiment_id == experiment.id,
                TrainingJob.job_kind.in_(["candidate", "refinement"]),
                TrainingJob.status.in_(["queued", "reserved", "starting", "running", "cancelling"]),
            )
        )
        if unfinished:
            await record_rejection(
                session,
                workspace_id=workspace_id,
                experiment_id=experiment.id,
                action="model.select_best",
                reason_code="selection_before_jobs_terminal",
                reason="all candidate and refinement jobs must reach a terminal state before selection",
                actor="platform",
                request={"unfinished_jobs": int(unfinished)},
                budget_snapshot=self._budget_snapshot(experiment),
            )
            await session.commit()
            raise PolicyViolation(
                "selection_before_jobs_terminal", "candidate jobs are still running"
            )
        candidates = list(
            (
                await session.scalars(
                    select(TrainingJob).where(
                        TrainingJob.experiment_id == experiment.id,
                        TrainingJob.job_kind.in_(["candidate", "refinement"]),
                        TrainingJob.status == "succeeded",
                    )
                )
            ).all()
        )
        scored = []
        for candidate in candidates:
            value = await self._metric_for_job(session, candidate.id, policy.validation_metric)
            if value is not None:
                scored.append((candidate, value))
        scored.sort(
            key=lambda item: self._score_sort_key(item[0], item[1], policy),
        )
        scored = scored[: policy.top_k]
        best_job = baseline
        best_value = baseline_metric
        retained = True
        for candidate, value in scored:
            improved = (
                value >= baseline_metric + policy.minimum_improvement
                if policy.direction == ObjectiveDirection.MAXIMIZE
                else value <= baseline_metric - policy.minimum_improvement
            )
            if improved and self._is_better(value, best_value, policy.direction):
                best_job = candidate
                best_value = value
                retained = False
        best_job = self._tie_break(best_job, scored, best_value, policy) if not retained else baseline
        experiment.best_job_id = best_job.id
        experiment.status = ExperimentStatus.EVALUATING.value
        experiment.selection_reason = "baseline_retained" if retained else f"candidate_selected:{best_job.id}"
        await record_decision(
            session,
            workspace_id=workspace_id,
            experiment_id=experiment.id,
            job_id=best_job.id,
            data=DecisionRecordData(
                action="model.select_best",
                result=DecisionResult.ACCEPTED,
                reason_code="baseline_retained" if retained else "candidate_selected",
                reason=experiment.selection_reason,
                actor=actor,
                phase="selection",
                validation_metrics={policy.validation_metric: best_value},
                budget_snapshot=self._budget_snapshot(experiment),
            ),
        )
        await self._submit_final_evaluation_jobs(session, experiment=experiment, best_job=best_job)
        return best_job

    async def register_candidate(
        self,
        session: AsyncSession,
        *,
        experiment_id: uuid.UUID,
        workspace_id: uuid.UUID,
        artifact_id: uuid.UUID,
        actor: str = "agent",
    ) -> ModelVersion:
        experiment = await session.get(Experiment, experiment_id, with_for_update=True)
        if experiment is None or experiment.workspace_id != workspace_id:
            raise ValueError("experiment not found")
        if experiment.best_job_id is None:
            raise PolicyViolation("selection_not_complete", "model selection is not complete")
        if experiment.final_test_metrics is None:
            raise PolicyViolation("final_evaluation_not_complete", "final test evaluation is not complete")
        job = await session.get(TrainingJob, experiment.best_job_id)
        artifact = await session.get(Artifact, artifact_id)
        if job is None or artifact is None or artifact.workspace_id != workspace_id:
            raise ValueError("candidate artifact not found")
        existing_model = await session.scalar(
            select(ModelVersion).where(
                ModelVersion.experiment_id == experiment.id,
                ModelVersion.job_id == job.id,
            )
        )
        if existing_model is not None:
            return existing_model
        expected_artifact_id = (job.spec or {}).get("bundle_artifact_id")
        if expected_artifact_id is None or str(expected_artifact_id) != str(artifact.id):
            await record_rejection(
                session,
                workspace_id=workspace_id,
                experiment_id=experiment.id,
                job_id=job.id,
                action="model.register_candidate",
                reason_code="bundle_artifact_mismatch",
                reason="only the selected training job's Model Bundle can be registered",
                actor=actor,
                request={"artifact_id": str(artifact.id)},
            )
            await session.commit()
            raise PolicyViolation("bundle_artifact_mismatch", "candidate artifact does not match selection")
        if not await self._verify_bundle(artifact):
            await record_rejection(
                session,
                workspace_id=workspace_id,
                experiment_id=experiment.id,
                job_id=job.id,
                action="model.register_candidate",
                reason_code="bundle_checksum_invalid",
                reason="Model Bundle checksum verification failed",
                actor=actor,
                request={"artifact_id": str(artifact.id)},
            )
            await session.commit()
            raise PolicyViolation(
                "bundle_checksum_invalid", "Model Bundle checksum verification failed"
            )
        model_count = (
            await session.scalar(select(func.count(ModelVersion.id)).where(ModelVersion.workspace_id == workspace_id))
            or 0
        )
        version = f"v{int(model_count) + 1}"
        model_version = ModelVersion(
            workspace_id=workspace_id,
            experiment_id=experiment.id,
            job_id=job.id,
            name=experiment.name,
            version=version,
            stage=ModelStage.CANDIDATE.value,
            bundle_artifact_id=artifact.id,
            bundle_sha256=artifact.checksum,
            recipe_id=job.recipe_id,
            validation_metrics=(await self._job_metrics(session, job.id)).get("validation", {}),
            final_test_metrics=(
                experiment.final_test_metrics.get("best")
                or experiment.final_test_metrics.get("baseline")
                or {}
            ),
            selection_policy=experiment.selection_policy,
            reproducibility=await self._reproducibility(session, job.id),
        )
        session.add(model_version)
        await session.flush()
        artifact_node = await ensure_lineage_node(
            session,
            workspace_id=workspace_id,
            node_type="artifact",
            ref_id=artifact.id,
            artifact_id=artifact.id,
            metadata={"checksum": artifact.checksum},
        )
        model_node = await ensure_lineage_node(
            session,
            workspace_id=workspace_id,
            node_type="model_version",
            ref_id=model_version.id,
        )
        await link_lineage(
            session,
            workspace_id=workspace_id,
            input_node=artifact_node,
            output_node=model_node,
            operation="register_candidate",
            actor_type="agent",
            actor_id=actor,
        )
        await record_decision(
            session,
            workspace_id=workspace_id,
            experiment_id=experiment.id,
            job_id=job.id,
            data=DecisionRecordData(
                action="model.register_candidate",
                result=DecisionResult.ACCEPTED,
                reason_code="candidate_registered",
                reason=f"registered {model_version.name} {model_version.version}",
                actor=actor,
                phase="model_registration",
                validation_metrics=model_version.validation_metrics,
            ),
        )
        return model_version

    async def _verify_bundle(self, artifact: Artifact) -> bool:
        try:
            content = await LocalArtifactStore().get(artifact.storage_key)
        except Exception:
            return False
        if hashlib.sha256(content).hexdigest() != artifact.checksum:
            return False
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                checksums = json.loads(archive.read("checksums.json").decode("utf-8"))
                if not isinstance(checksums, dict) or not checksums:
                    return False
                names = [item.filename for item in archive.infolist() if not item.is_dir()]
                if len(names) != len(set(names)) or len(names) > 100:
                    return False
                if any(name.startswith("/") or ".." in name.split("/") for name in names):
                    return False
                required = {
                    "model.pt",
                    "preprocessor.joblib",
                    "schema.json",
                    "metrics.json",
                    "training-config.json",
                    "reproducibility.json",
                    "lineage.json",
                }
                if not required.issubset(set(checksums)):
                    return False
                for name, expected in checksums.items():
                    if not isinstance(name, str) or not isinstance(expected, str) or len(expected) != 64:
                        return False
                    actual = hashlib.sha256(archive.read(name)).hexdigest()
                    if actual != expected:
                        return False
        except Exception:
            return False
        return True

    async def reject_test_metric_access(
        self,
        session: AsyncSession,
        *,
        workspace_id: uuid.UUID,
        experiment_id: uuid.UUID | None,
        actor: str,
        run_id: uuid.UUID | None = None,
        node_run_id: uuid.UUID | None = None,
    ) -> None:
        await record_rejection(
            session,
            workspace_id=workspace_id,
            experiment_id=experiment_id,
            action="training.test_metrics",
            reason_code="test_metrics_forbidden",
            reason="test metrics are platform-only final evaluation metadata",
            actor=actor,
            run_id=run_id,
            node_run_id=node_run_id,
            phase="selection",
        )

    async def record_cancel_request(
        self,
        session: AsyncSession,
        *,
        job: TrainingJob,
        actor: str,
    ) -> None:
        await record_decision(
            session,
            workspace_id=job.workspace_id,
            experiment_id=job.experiment_id,
            job_id=job.id,
            data=DecisionRecordData(
                action="training.cancel",
                result=DecisionResult.ACCEPTED,
                reason_code="cancellation_requested",
                reason="graceful cancellation requested",
                actor=actor,
                phase="execution",
            ),
        )

    async def final_evaluation_job_ids(self, session: AsyncSession, experiment_id: uuid.UUID) -> list[uuid.UUID]:
        jobs = list(
            (
                await session.scalars(
                    select(TrainingJob).where(
                        TrainingJob.experiment_id == experiment_id,
                        TrainingJob.job_kind == "final_test",
                    )
                )
            ).all()
        )
        return [job.id for job in jobs]

    async def await_experiment_finalization(
        self,
        *,
        experiment_id: uuid.UUID,
        timeout_seconds: float,
        workspace_id: uuid.UUID | None = None,
    ) -> dict[str, Any]:
        factory = get_session_factory()
        async with factory() as session:
            if workspace_id is not None:
                experiment = await session.get(Experiment, experiment_id)
                if experiment is None or experiment.workspace_id != workspace_id:
                    raise ValueError("experiment not found in workspace")
            job_ids = await self.final_evaluation_job_ids(session, experiment_id)
        if not job_ids:
            raise PolicyViolation("final_evaluation_missing", "no final test evaluation jobs exist")
        jobs = await self.await_jobs(
            job_ids=job_ids,
            timeout_seconds=timeout_seconds,
            workspace_id=workspace_id,
        )
        if any(job.status != "succeeded" for job in jobs):
            async with factory() as session:
                async with session.begin():
                    experiment = await session.get(Experiment, experiment_id, with_for_update=True)
                    if experiment is not None:
                        experiment.status = ExperimentStatus.FAILED.value
                        experiment.error = "final test evaluation failed"
                        experiment.finished_at = datetime.now(UTC)
                        await record_decision(
                            session,
                            workspace_id=experiment.workspace_id,
                            experiment_id=experiment.id,
                            job_id=experiment.best_job_id,
                            data=DecisionRecordData(
                                action="experiment.final_test_evaluation",
                                result=DecisionResult.FAILED,
                                reason_code="final_evaluation_failed",
                                reason="At least one platform final test evaluation job failed",
                                actor="platform",
                                phase="final_evaluation",
                            ),
                        )
            raise RuntimeError("final test evaluation failed")
        async with factory() as session:
            async with session.begin():
                experiment = await session.get(Experiment, experiment_id, with_for_update=True)
                if experiment is None or (
                    workspace_id is not None and experiment.workspace_id != workspace_id
                ):
                    raise ValueError("experiment not found in workspace")
                metrics: dict[str, Any] = {"baseline": {}, "best": {}}
                for job in jobs:
                    source_job_id = (job.spec or {}).get("source_job_id")
                    if source_job_id is None:
                        continue
                    source = await session.get(TrainingJob, uuid.UUID(str(source_job_id)))
                    if source is None:
                        continue
                    key = "baseline" if source.job_kind == "baseline" else "best"
                    attempt = await self._latest_attempt(session, job.id)
                    metrics[key] = dict((attempt.metrics or {}).get("test", {})) if attempt else {}
                if metrics["best"] == {} and metrics["baseline"]:
                    metrics["best"] = dict(metrics["baseline"])
                experiment.final_test_metrics = metrics
                experiment.status = ExperimentStatus.COMPLETED.value
                experiment.finished_at = datetime.now(UTC)
                await record_decision(
                    session,
                    workspace_id=experiment.workspace_id,
                    experiment_id=experiment.id,
                    job_id=experiment.best_job_id,
                    data=DecisionRecordData(
                        action="experiment.final_test_evaluation",
                        result=DecisionResult.ACCEPTED,
                        reason_code="final_evaluation_completed",
                        reason="Final test metrics recorded as report metadata only",
                        actor="platform",
                        phase="final_evaluation",
                    ),
                )
        return {
            "experiment_id": str(experiment_id),
            "status": "completed",
            "final_evaluation": "stored_platform_metadata_only",
        }

    async def _submit_final_evaluation_jobs(
        self,
        session: AsyncSession,
        *,
        experiment: Experiment,
        best_job: TrainingJob,
    ) -> list[TrainingJob]:
        baseline = await session.get(TrainingJob, experiment.baseline_job_id)
        if baseline is None:
            raise ValueError("baseline job not found")
        created: list[TrainingJob] = []
        for source in [baseline, best_job]:
            payload = dict(source.spec)
            source_spec = TrainingJobSpec(
                recipe_id=source.recipe_id,
                dataset_version_id=source.dataset_version_id,
                round_number=source.round_number,
                target_column=payload.get("target_column"),
                parameters=payload.get("parameters", {}),
                random_seed=int(payload.get("random_seed", 42)),
                max_epochs=int(payload.get("max_epochs", 50)),
                batch_size=int(payload.get("batch_size", 32)),
                device_policy="cpu_only",
                max_job_seconds=int(payload.get("max_job_seconds", 900)),
                idempotency_key=f"final-test:{experiment.id}:{source.id}",
                evaluation_only=True,
                source_job_id=source.id,
            )
            created.append(
                await self.submit_job(
                    session,
                    workspace_id=experiment.workspace_id,
                    experiment=experiment,
                    job_spec=source_spec,
                    job_kind="final_test",
                    submitted_by=source.submitted_by,
                )
            )
        return created

    async def _metric_for_job(self, session: AsyncSession, job_id: uuid.UUID, metric: str) -> float | None:
        metrics = await self._job_metrics(session, job_id)
        return metrics.get("validation", {}).get(metric)

    async def _job_metrics(self, session: AsyncSession, job_id: uuid.UUID) -> dict:
        attempt = await self._latest_attempt(session, job_id)
        return dict(attempt.metrics or {}) if attempt else {}

    async def _latest_attempt(self, session: AsyncSession, job_id: uuid.UUID) -> TrainingAttempt | None:
        return await session.scalar(
            select(TrainingAttempt)
            .where(TrainingAttempt.job_id == job_id)
            .order_by(TrainingAttempt.attempt.desc())
            .limit(1)
        )

    async def _reproducibility(self, session: AsyncSession, job_id: uuid.UUID) -> dict:
        attempt = await self._latest_attempt(session, job_id)
        return dict(attempt.manifest or {}) if attempt else {}

    def _is_better(self, value: float, current: float, direction: ObjectiveDirection) -> bool:
        return value > current if direction == ObjectiveDirection.MAXIMIZE else value < current

    def _tie_break(
        self,
        current_best: TrainingJob,
        scored: list[tuple[TrainingJob, float]],
        best_value: float,
        policy: SelectionPolicy,
    ) -> TrainingJob:
        tied = [job for job, value in scored if value == best_value]
        if not tied:
            return current_best
        if policy.tie_break == "lower_job_seconds":
            return min(tied, key=lambda item: (item.actual_total_seconds, item.config_checksum))
        if policy.tie_break == "lower_gpu_seconds":
            return min(tied, key=lambda item: (item.actual_gpu_seconds, item.config_checksum))
        return min(tied, key=lambda item: item.config_checksum)

    def _score_sort_key(
        self,
        job: TrainingJob,
        value: float,
        policy: SelectionPolicy,
    ) -> tuple[float, float, float, str]:
        primary = -value if policy.direction == ObjectiveDirection.MAXIMIZE else value
        if policy.tie_break == "lower_gpu_seconds":
            return (primary, job.actual_gpu_seconds, job.actual_total_seconds, job.config_checksum)
        if policy.tie_break == "config_checksum":
            return (primary, 0.0, 0.0, job.config_checksum)
        return (primary, job.actual_total_seconds, job.actual_gpu_seconds, job.config_checksum)

    async def _append_job_event(
        self,
        session: AsyncSession,
        job: TrainingJob,
        event_type: str,
        payload: dict[str, Any],
    ) -> TrainingEvent:
        next_seq = (
            await session.scalar(
                select(func.coalesce(func.max(TrainingEvent.seq), 0)).where(TrainingEvent.job_id == job.id)
            )
            or 0
        ) + 1
        event = TrainingEvent(
            job_id=job.id,
            seq=next_seq,
            event_type=event_type,
            payload=payload,
        )
        session.add(event)
        await session.flush()
        return event

    def _budget_snapshot(self, experiment: Experiment) -> dict[str, Any]:
        return {
            "reserved_total_seconds": experiment.reserved_total_seconds,
            "consumed_total_seconds": experiment.consumed_total_seconds,
            "reserved_gpu_seconds": experiment.reserved_gpu_seconds,
            "consumed_gpu_seconds": experiment.consumed_gpu_seconds,
            "jobs_started": experiment.jobs_started,
        }

