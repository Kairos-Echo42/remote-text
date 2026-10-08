from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import select

from agentforge.capabilities import CapabilityContext, CapabilityRegistry
from agentforge.capabilities.builtin import register_builtin_capabilities
from agentforge.config import get_settings
from agentforge.db import get_session_factory
from agentforge.models import BudgetReservation, DecisionRecord, Experiment, TrainingJob, Workspace
from agentforge.storage.artifacts import LocalArtifactStore
from agentforge.training.audit import PolicyViolation
from agentforge.training.backend import FakeTrainingBackend
from agentforge.training.budget import release_gpu_reservation
from agentforge.training.datasets import create_dataset, upload_dataset_version
from agentforge.training.recipes import get_recipe_registry
from agentforge.training.runner import TrainingWorker
from agentforge.training.service import TrainingService
from agentforge.training.types import (
    DatasetKind,
    ExperimentBudget,
    ObjectiveDirection,
    SelectionPolicy,
    TaskType,
    TrainingJobSpec,
)


def _csv_dataset() -> bytes:
    rows = ["feature_a,feature_b,target"]
    for index in range(60):
        rows.append(f"{index % 7},{index % 5},{index % 2}")
    return "\n".join(rows).encode("utf-8")


@pytest.mark.asyncio
async def test_recipe_registry_is_immutable_and_validates_ranges():
    registry = get_recipe_registry()
    with pytest.raises(RuntimeError):
        registry.register(next(iter(registry.list())))
    recipe = registry.get("tabular.classification.logistic@1.0.0")
    with pytest.raises(ValueError):
        recipe.validate_hyperparameters({"C": 999999})
    validated = recipe.validate_hyperparameters({"C": 0.5})
    assert validated["C"] == 0.5


@pytest.mark.asyncio
async def test_agent_experiment_uses_validation_selection_and_final_test(monkeypatch):
    monkeypatch.setattr(get_settings(), "training_cpu_concurrency", 4)
    factory = get_session_factory()
    async with factory() as session:
        workspace = await session.scalar(select(Workspace).limit(1))
        dataset = await create_dataset(
            session,
            workspace_id=workspace.id,
            name="training-e2e",
            kind=DatasetKind.TABULAR,
            task_type=TaskType.TABULAR_CLASSIFICATION,
        )
        await session.flush()
        version = await upload_dataset_version(
            session,
            dataset=dataset,
            filename="training.csv",
            content=_csv_dataset(),
            artifact_store=LocalArtifactStore(get_settings().artifact_root),
            target_column="target",
            split_seed=7,
        )
        await session.commit()

    service = TrainingService()
    backend = FakeTrainingBackend(step_delay=0)
    worker = TrainingWorker(backend)
    async with factory() as session:
        experiment = await service.create_experiment(
            session,
            workspace_id=workspace.id,
            dataset_version_id=version.id,
            name="validation-only-selection",
            baseline_strategy_id=None,
            budget=ExperimentBudget(
                max_jobs=6, max_rounds=2, max_job_seconds=30, max_total_seconds=180, max_gpu_seconds=0
            ),
            selection_policy=SelectionPolicy(
                validation_metric="f1_macro",
                direction=ObjectiveDirection.MAXIMIZE,
                minimum_improvement=0.01,
            ),
        )
        await session.commit()
        experiment_id = experiment.id
        baseline_job_id = experiment.baseline_job_id

    await _drain_worker(worker, job_ids=[baseline_job_id])
    async with factory() as session:
        created = await service.submit_batch(
            session,
            workspace_id=workspace.id,
            experiment_id=experiment_id,
            jobs=[
                TrainingJobSpec(
                    recipe_id="tabular.classification.logistic@1.0.0",
                    dataset_version_id=version.id,
                    target_column="target",
                    parameters={"C": 1.0},
                    device_policy="cpu_only",
                    max_job_seconds=30,
                    idempotency_key="round-1-logistic",
                ),
                TrainingJobSpec(
                    recipe_id="tabular.classification.random_forest@1.0.0",
                    dataset_version_id=version.id,
                    target_column="target",
                    parameters={"n_estimators": 20, "max_depth": 4},
                    device_policy="cpu_only",
                    max_job_seconds=30,
                    idempotency_key="round-1-rf",
                ),
            ],
            round_number=1,
            submitted_by=None,
        )
        await session.commit()
        candidate_ids = [job.id for job in created]

    await _drain_worker(worker, job_ids=candidate_ids)
    async with factory() as session:
        best = await service.select_best(
            session,
            experiment_id=experiment_id,
            workspace_id=workspace.id,
            actor="agent",
        )
        await session.commit()
        assert best.job_kind in {"candidate", "refinement"}

    async with factory() as session:
        final_ids = await service.final_evaluation_job_ids(session, experiment_id)
    await _drain_worker(worker, job_ids=final_ids)
    await service.await_experiment_finalization(experiment_id=experiment_id, timeout_seconds=10)

    async with factory() as session:
        experiment = await session.get(Experiment, experiment_id)
        assert experiment is not None
        assert experiment.final_test_metrics is not None
        assert experiment.selection_reason == f"candidate_selected:{best.id}"
        bundle_id = (best.spec or {}).get("bundle_artifact_id")
        model_version = await service.register_candidate(
            session,
            experiment_id=experiment.id,
            workspace_id=workspace.id,
            artifact_id=__import__("uuid").UUID(str(bundle_id)),
            actor="agent",
        )
        await session.commit()
        assert model_version.stage == "candidate"
        assert model_version.final_test_metrics


async def _drain_worker(worker: TrainingWorker, *, job_ids, timeout_seconds: float = 10):
    factory = get_session_factory()
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        await worker.tick()
        async with factory() as session:
            jobs = list((await session.scalars(select(TrainingJob).where(TrainingJob.id.in_(job_ids)))).all())
        if jobs and all(job.status in {"succeeded", "failed", "cancelled"} for job in jobs):
            return jobs
        await asyncio.sleep(0.01)
    raise TimeoutError("training worker did not drain")


@pytest.mark.asyncio
async def test_budget_rejection_and_test_metric_request_are_audited(monkeypatch):
    factory = get_session_factory()
    async with factory() as session:
        workspace = await session.scalar(select(Workspace).limit(1))
        dataset = await create_dataset(
            session,
            workspace_id=workspace.id,
            name="budget-audit",
            kind=DatasetKind.TABULAR,
            task_type=TaskType.TABULAR_CLASSIFICATION,
        )
        await session.flush()
        version = await upload_dataset_version(
            session,
            dataset=dataset,
            filename="budget.csv",
            content=_csv_dataset(),
            artifact_store=LocalArtifactStore(get_settings().artifact_root),
            target_column="target",
        )
        await session.commit()
    service = TrainingService()
    async with factory() as session:
        experiment = await service.create_experiment(
            session,
            workspace_id=workspace.id,
            dataset_version_id=version.id,
            name="one-job-budget",
            baseline_strategy_id=None,
            budget=ExperimentBudget(
                max_jobs=1,
                max_rounds=1,
                max_job_seconds=10,
                max_total_seconds=10,
                max_gpu_seconds=0,
            ),
            selection_policy=SelectionPolicy(validation_metric="f1_macro", direction=ObjectiveDirection.MAXIMIZE),
        )
        await session.commit()
        experiment_id = experiment.id

    with pytest.raises(PolicyViolation):
        async with factory() as session:
            await service.submit_batch(
                session,
                workspace_id=workspace.id,
                experiment_id=experiment_id,
                jobs=[
                    TrainingJobSpec(
                        recipe_id="tabular.classification.logistic@1.0.0",
                        dataset_version_id=version.id,
                        target_column="target",
                        parameters={"C": 1.0},
                        device_policy="cpu_only",
                        max_job_seconds=10,
                    )
                ],
                round_number=1,
            )

    registry = CapabilityRegistry()
    register_builtin_capabilities(registry)
    context = CapabilityContext(
        workspace_id=workspace.id,
        run_id=__import__("uuid").uuid4(),
        node_run_id=__import__("uuid").uuid4(),
        actor_id=None,
    )
    with pytest.raises(PolicyViolation):
        await registry.invoke(
            "training.test_metrics",
            context,
            {"experiment_id": str(experiment_id)},
        )

    async with factory() as session:
        records = list(
            (
                await session.scalars(
                    select(DecisionRecord)
                    .where(DecisionRecord.experiment_id == experiment_id)
                    .order_by(DecisionRecord.created_at)
                )
            ).all()
        )
    assert any(item.reason_code == "budget_exceeded" for item in records)
    assert any(item.reason_code == "test_metrics_forbidden" for item in records)
    assert all(item.action and item.result and item.reason and item.created_at for item in records)


@pytest.mark.asyncio
async def test_recipe_range_and_unfinished_candidate_selection_are_rejected():
    factory = get_session_factory()
    async with factory() as session:
        workspace = await session.scalar(select(Workspace).limit(1))
        dataset = await create_dataset(
            session,
            workspace_id=workspace.id,
            name="policy-guard",
            kind=DatasetKind.TABULAR,
            task_type=TaskType.TABULAR_CLASSIFICATION,
        )
        await session.flush()
        version = await upload_dataset_version(
            session,
            dataset=dataset,
            filename="policy.csv",
            content=_csv_dataset(),
            artifact_store=LocalArtifactStore(get_settings().artifact_root),
            target_column="target",
        )
        await session.commit()

    service = TrainingService()
    async with factory() as session:
        experiment = await service.create_experiment(
            session,
            workspace_id=workspace.id,
            dataset_version_id=version.id,
            name="policy-guard",
            baseline_strategy_id=None,
            budget=ExperimentBudget(
                max_jobs=3,
                max_rounds=1,
                max_job_seconds=30,
                max_total_seconds=90,
                max_gpu_seconds=0,
            ),
        )
        await session.commit()
        experiment_id = experiment.id
        baseline_job_id = experiment.baseline_job_id

    worker = TrainingWorker(FakeTrainingBackend(step_delay=0))
    await _drain_worker(worker, job_ids=[baseline_job_id])

    with pytest.raises(PolicyViolation) as invalid_recipe:
        async with factory() as session:
            await service.submit_batch(
                session,
                workspace_id=workspace.id,
                experiment_id=experiment_id,
                jobs=[
                    TrainingJobSpec(
                        recipe_id="tabular.classification.logistic@1.0.0",
                        dataset_version_id=version.id,
                        parameters={"C": 999999},
                        device_policy="cpu_only",
                        max_job_seconds=30,
                    )
                ],
                round_number=1,
            )
    assert invalid_recipe.value.reason_code == "recipe_parameter_invalid"

    async with factory() as session:
        running_candidate = await service.submit_batch(
            session,
            workspace_id=workspace.id,
            experiment_id=experiment_id,
            jobs=[
                TrainingJobSpec(
                    recipe_id="tabular.classification.logistic@1.0.0",
                    dataset_version_id=version.id,
                    parameters={"C": 0.5},
                    device_policy="cpu_only",
                    max_job_seconds=30,
                )
            ],
            round_number=1,
        )
        await session.commit()
        running_candidate_id = running_candidate[0].id

    with pytest.raises(PolicyViolation) as unfinished:
        async with factory() as session:
            await service.select_best(
                session,
                experiment_id=experiment_id,
                workspace_id=workspace.id,
                actor="agent",
            )
    assert unfinished.value.reason_code == "selection_before_jobs_terminal"

    async with factory() as session:
        records = list(
            (
                await session.scalars(
                    select(DecisionRecord)
                    .where(DecisionRecord.experiment_id == experiment_id)
                    .order_by(DecisionRecord.created_at)
                )
            ).all()
        )
    assert any(item.reason_code == "recipe_parameter_invalid" for item in records)
    assert any(item.reason_code == "selection_before_jobs_terminal" for item in records)

    await _drain_worker(worker, job_ids=[running_candidate_id])


@pytest.mark.asyncio
async def test_baseline_retained_when_minimum_improvement_is_not_met():
    factory = get_session_factory()
    async with factory() as session:
        workspace = await session.scalar(select(Workspace).limit(1))
        dataset = await create_dataset(
            session,
            workspace_id=workspace.id,
            name="baseline-retention",
            kind=DatasetKind.TABULAR,
            task_type=TaskType.TABULAR_CLASSIFICATION,
        )
        await session.flush()
        version = await upload_dataset_version(
            session,
            dataset=dataset,
            filename="retention.csv",
            content=_csv_dataset(),
            artifact_store=LocalArtifactStore(get_settings().artifact_root),
            target_column="target",
        )
        await session.commit()

    service = TrainingService()
    async with factory() as session:
        experiment = await service.create_experiment(
            session,
            workspace_id=workspace.id,
            dataset_version_id=version.id,
            name="retain-baseline",
            baseline_strategy_id=None,
            budget=ExperimentBudget(
                max_jobs=3,
                max_rounds=1,
                max_job_seconds=30,
                max_total_seconds=90,
                max_gpu_seconds=0,
            ),
            selection_policy=SelectionPolicy(
                validation_metric="f1_macro",
                direction=ObjectiveDirection.MAXIMIZE,
                minimum_improvement=1.0,
            ),
        )
        await session.commit()
        experiment_id = experiment.id
        baseline_job_id = experiment.baseline_job_id

    worker = TrainingWorker(FakeTrainingBackend(step_delay=0))
    await _drain_worker(worker, job_ids=[baseline_job_id])
    async with factory() as session:
        candidate = await service.submit_batch(
            session,
            workspace_id=workspace.id,
            experiment_id=experiment_id,
            jobs=[
                TrainingJobSpec(
                    recipe_id="tabular.classification.logistic@1.0.0",
                    dataset_version_id=version.id,
                    parameters={"C": 0.5},
                    device_policy="cpu_only",
                    max_job_seconds=30,
                )
            ],
            round_number=1,
        )
        await session.commit()
        candidate_id = candidate[0].id
    await _drain_worker(worker, job_ids=[candidate_id])

    async with factory() as session:
        best = await service.select_best(
            session,
            experiment_id=experiment_id,
            workspace_id=workspace.id,
            actor="agent",
        )
        await session.commit()
        experiment = await session.get(Experiment, experiment_id)
    assert best.id == baseline_job_id
    assert experiment is not None
    assert experiment.selection_reason == "baseline_retained"

    async with factory() as session:
        final_ids = await service.final_evaluation_job_ids(session, experiment_id)
    await _drain_worker(worker, job_ids=final_ids)
    await service.await_experiment_finalization(
        experiment_id=experiment_id,
        timeout_seconds=10,
        workspace_id=workspace.id,
    )
    async with factory() as session:
        experiment = await session.get(Experiment, experiment_id)
        assert experiment is not None
        assert experiment.final_test_metrics is not None
        assert experiment.final_test_metrics["best"] == experiment.final_test_metrics["baseline"]


@pytest.mark.asyncio
async def test_preferred_gpu_reservation_is_released_on_cpu_fallback():
    factory = get_session_factory()
    async with factory() as session:
        workspace = await session.scalar(select(Workspace).limit(1))
        dataset = await create_dataset(
            session,
            workspace_id=workspace.id,
            name="gpu-reservation",
            kind=DatasetKind.TABULAR,
            task_type=TaskType.TABULAR_CLASSIFICATION,
        )
        await session.flush()
        version = await upload_dataset_version(
            session,
            dataset=dataset,
            filename="gpu.csv",
            content=_csv_dataset(),
            artifact_store=LocalArtifactStore(get_settings().artifact_root),
            target_column="target",
        )
        await session.commit()

    service = TrainingService()
    async with factory() as session:
        experiment = await service.create_experiment(
            session,
            workspace_id=workspace.id,
            dataset_version_id=version.id,
            name="gpu-reservation",
            baseline_strategy_id=None,
            budget=ExperimentBudget(
                max_jobs=3,
                max_rounds=1,
                max_job_seconds=30,
                max_total_seconds=90,
                max_gpu_seconds=30,
            ),
        )
        created = await service.submit_batch(
            session,
            workspace_id=workspace.id,
            experiment_id=experiment.id,
            jobs=[
                TrainingJobSpec(
                    recipe_id="tabular.classification.mlp@1.0.0",
                    dataset_version_id=version.id,
                    parameters={"hidden_sizes": [64], "dropout": 0.0},
                    device_policy="preferred_gpu",
                    max_job_seconds=30,
                    max_epochs=2,
                )
            ],
            round_number=1,
        )
        await session.commit()
        experiment_id = experiment.id
        candidate_id = created[0].id

    async with factory() as session:
        await release_gpu_reservation(
            session,
            job_id=candidate_id,
            fallback_reason="test fallback",
        )
        await session.commit()

    async with factory() as session:
        experiment = await session.get(Experiment, experiment_id)
        reservation = await session.scalar(
            select(BudgetReservation).where(BudgetReservation.job_id == candidate_id)
        )
        assert experiment is not None
        assert reservation is not None
        assert experiment.reserved_gpu_seconds == 0
        assert reservation.gpu_seconds == 0
        assert reservation.status == "total_only"


@pytest.mark.asyncio
async def test_policy_gated_capabilities_record_rejections():
    factory = get_session_factory()
    async with factory() as session:
        workspace = await session.scalar(select(Workspace).limit(1))
    registry = CapabilityRegistry()
    register_builtin_capabilities(registry)
    context = CapabilityContext(
        workspace_id=workspace.id,
        run_id=uuid.uuid4(),
        node_run_id=uuid.uuid4(),
        actor_id=None,
    )
    with pytest.raises(PolicyViolation):
        await registry.invoke("ml.split.modify", context, {"dataset_version_id": str(uuid.uuid4())})
    with pytest.raises(PolicyViolation):
        await registry.invoke("model.promote", context, {"model_version_id": str(uuid.uuid4())})
    async with factory() as session:
        records = list((await session.scalars(select(DecisionRecord))).all())
    assert {item.reason_code for item in records} >= {
        "split_mutation_forbidden",
        "automatic_promotion_forbidden",
    }


@pytest.mark.asyncio
async def test_training_jobs_cannot_cross_workspace_dataset_boundaries():
    factory = get_session_factory()
    async with factory() as session:
        workspace_one = await session.scalar(select(Workspace).order_by(Workspace.created_at).limit(1))
        workspace_two = Workspace(name="isolated-training", slug="isolated-training")
        session.add(workspace_two)
        await session.flush()
        dataset_one = await create_dataset(
            session,
            workspace_id=workspace_one.id,
            name="workspace-one-data",
            kind=DatasetKind.TABULAR,
            task_type=TaskType.TABULAR_CLASSIFICATION,
        )
        dataset_two = await create_dataset(
            session,
            workspace_id=workspace_two.id,
            name="workspace-two-data",
            kind=DatasetKind.TABULAR,
            task_type=TaskType.TABULAR_CLASSIFICATION,
        )
        await session.flush()
        version_one = await upload_dataset_version(
            session,
            dataset=dataset_one,
            filename="one.csv",
            content=_csv_dataset(),
            artifact_store=LocalArtifactStore(get_settings().artifact_root),
            target_column="target",
        )
        version_two = await upload_dataset_version(
            session,
            dataset=dataset_two,
            filename="two.csv",
            content=_csv_dataset(),
            artifact_store=LocalArtifactStore(get_settings().artifact_root),
            target_column="target",
        )
        await session.commit()

    service = TrainingService()
    async with factory() as session:
        experiment = await service.create_experiment(
            session,
            workspace_id=workspace_one.id,
            dataset_version_id=version_one.id,
            name="workspace-isolation",
            baseline_strategy_id=None,
            budget=ExperimentBudget(
                max_jobs=2,
                max_rounds=1,
                max_job_seconds=30,
                max_total_seconds=60,
                max_gpu_seconds=0,
            ),
        )
        await session.commit()
        experiment_id = experiment.id

    with pytest.raises(PolicyViolation) as boundary:
        async with factory() as session:
            await service.submit_batch(
                session,
                workspace_id=workspace_one.id,
                experiment_id=experiment_id,
                jobs=[
                    TrainingJobSpec(
                        recipe_id="tabular.classification.logistic@1.0.0",
                        dataset_version_id=version_two.id,
                        parameters={"C": 1.0},
                        device_policy="cpu_only",
                        max_job_seconds=30,
                    )
                ],
                round_number=1,
            )
    assert boundary.value.reason_code == "dataset_version_mismatch"
    async with factory() as session:
        record = await session.scalar(
            select(DecisionRecord).where(
                DecisionRecord.workspace_id == workspace_one.id,
                DecisionRecord.reason_code == "dataset_version_mismatch",
            )
        )
        assert record is not None


@pytest.mark.asyncio
async def test_cancelling_a_queued_job_finalizes_budget_without_starting_a_container():
    factory = get_session_factory()
    async with factory() as session:
        workspace = await session.scalar(select(Workspace).limit(1))
        dataset = await create_dataset(
            session,
            workspace_id=workspace.id,
            name="queued-cancel",
            kind=DatasetKind.TABULAR,
            task_type=TaskType.TABULAR_CLASSIFICATION,
        )
        await session.flush()
        version = await upload_dataset_version(
            session,
            dataset=dataset,
            filename="cancel.csv",
            content=_csv_dataset(),
            artifact_store=LocalArtifactStore(get_settings().artifact_root),
            target_column="target",
        )
        await session.commit()

    service = TrainingService()
    async with factory() as session:
        experiment = await service.create_experiment(
            session,
            workspace_id=workspace.id,
            dataset_version_id=version.id,
            name="queued-cancel",
            baseline_strategy_id=None,
            budget=ExperimentBudget(
                max_jobs=1,
                max_rounds=1,
                max_job_seconds=30,
                max_total_seconds=30,
                max_gpu_seconds=0,
            ),
        )
        await session.commit()
        experiment_id = experiment.id
        job_id = experiment.baseline_job_id

    async with factory() as session:
        job = await session.get(TrainingJob, job_id, with_for_update=True)
        job.status = "cancelling"
        await session.commit()

    await TrainingWorker(FakeTrainingBackend(step_delay=0)).tick()
    async with factory() as session:
        job = await session.get(TrainingJob, job_id)
        experiment = await session.get(Experiment, experiment_id)
        reservation = await session.scalar(
            select(BudgetReservation).where(BudgetReservation.job_id == job_id)
        )
        assert job is not None
        assert experiment is not None
        assert reservation is not None
        assert job.status == "cancelled"
        assert reservation.status == "finalized"
        assert experiment.reserved_total_seconds == 0
