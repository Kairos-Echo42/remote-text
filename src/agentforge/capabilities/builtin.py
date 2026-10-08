from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from agentforge.capabilities.base import (
    CapabilityContext,
    CapabilityRegistry,
    CapabilityResult,
    FunctionCapability,
    SideEffect,
)
from agentforge.db import get_session_factory
from agentforge.models import Dataset, DatasetVersion, Experiment, TrainingJob
from agentforge.runtime.queue import get_redis
from agentforge.training.audit import PolicyViolation, record_rejection
from agentforge.training.recipes import get_recipe_registry
from agentforge.training.service import TrainingService
from agentforge.training.types import ExperimentBudget, SelectionPolicy, TrainingJobSpec


def register_builtin_capabilities(registry: CapabilityRegistry) -> None:
    registry.register(
        FunctionCapability(
            "fixture.echo",
            "Return the provided payload; used for deterministic tests and demos.",
            _echo,
            input_schema={"type": "object", "additionalProperties": True},
        )
    )
    registry.register(
        FunctionCapability(
            "web.search",
            "Search the configured web provider and return ranked snippets.",
            _web_search,
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["query"],
            },
        )
    )
    registry.register(
        FunctionCapability(
            "knowledge.search",
            "Retrieve evidence from workspace knowledge bases using hybrid search.",
            _knowledge_search,
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["query"],
            },
        )
    )
    registry.register(
        FunctionCapability(
            "memory.search",
            "Search durable workspace and agent memory.",
            _memory_search,
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["query"],
            },
        )
    )
    registry.register(
        FunctionCapability(
            "memory.write",
            "Write a curated durable memory record.",
            _memory_write,
            input_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "agent_name": {"type": "string"},
                    "kind": {"type": "string"},
                },
                "required": ["content"],
            },
            side_effect=SideEffect.WORKSPACE,
        )
    )
    registry.register(
        FunctionCapability(
            "workspace.write",
            "Write a UTF-8 file into the node sandbox workspace.",
            _workspace_write,
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            side_effect=SideEffect.WORKSPACE,
        )
    )
    registry.register(
        FunctionCapability(
            "workspace.read",
            "Read a UTF-8 file from the node sandbox workspace.",
            _workspace_read,
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
    )
    registry.register(
        FunctionCapability(
            "workspace.list",
            "List files in the node sandbox workspace.",
            _workspace_list,
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        )
    )
    registry.register(
        FunctionCapability(
            "sandbox.execute",
            "Execute a command inside the isolated node sandbox.",
            _sandbox_execute,
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "minimum": 1},
                    "cwd": {"type": "string"},
                    "capture": {"type": "array", "items": {"type": "string"}},
                    "files": {"type": "object", "additionalProperties": {"type": "string"}},
                },
                "required": ["command"],
            },
            side_effect=SideEffect.WORKSPACE,
            idempotent=False,
        )
    )
    registry.register(
        FunctionCapability(
            "ml.dataset.inspect",
            "Inspect an immutable DatasetVersion and its split/reproducibility metadata.",
            _ml_dataset_inspect,
            input_schema={
                "type": "object",
                "properties": {"dataset_version_id": {"type": "string"}},
                "required": ["dataset_version_id"],
            },
        )
    )
    registry.register(
        FunctionCapability(
            "ml.recipe.list",
            "List trusted, immutable training recipes and their allowed hyperparameters.",
            _ml_recipe_list,
            input_schema={"type": "object", "properties": {}},
        )
    )
    registry.register(
        FunctionCapability(
            "ml.experiment.create",
            "Create a budgeted training experiment with a mandatory baseline.",
            _ml_experiment_create,
            input_schema={"type": "object", "additionalProperties": True},
            side_effect=SideEffect.WORKSPACE,
        )
    )
    registry.register(
        FunctionCapability(
            "training.submit_batch",
            "Submit a budgeted batch of trusted Recipe training jobs.",
            _training_submit_batch,
            input_schema={"type": "object", "additionalProperties": True},
            side_effect=SideEffect.EXTERNAL,
            idempotent=True,
        )
    )
    registry.register(
        FunctionCapability(
            "training.await_batch",
            "Wait for training jobs and return validation metrics only.",
            _training_await_batch,
            input_schema={
                "type": "object",
                "properties": {
                    "job_ids": {"type": "array", "items": {"type": "string"}},
                    "timeout_seconds": {"type": "integer", "minimum": 1},
                },
                "required": ["job_ids"],
            },
        )
    )
    registry.register(
        FunctionCapability(
            "training.await_experiment",
            "Wait for platform final test evaluation and return status without test metrics.",
            _training_await_experiment,
            input_schema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "minimum": 1},
                },
                "required": ["experiment_id"],
            },
        )
    )
    registry.register(
        FunctionCapability(
            "training.results",
            "Read validation and resource results; test metrics are never exposed.",
            _training_results,
            input_schema={
                "type": "object",
                "properties": {"job_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["job_ids"],
            },
        )
    )
    registry.register(
        FunctionCapability(
            "training.test_metrics",
            "Policy-gated endpoint that always rejects Agent access to test metrics.",
            _training_test_metrics,
            input_schema={
                "type": "object",
                "properties": {"experiment_id": {"type": "string"}},
            },
        )
    )
    registry.register(
        FunctionCapability(
            "model.select_best",
            "Apply the platform SelectionPolicy using validation metrics only.",
            _model_select_best,
            input_schema={"type": "object", "additionalProperties": True},
            side_effect=SideEffect.WORKSPACE,
        )
    )
    registry.register(
        FunctionCapability(
            "model.register_candidate",
            "Register the selected model bundle as a candidate ModelVersion.",
            _model_register_candidate,
            input_schema={"type": "object", "additionalProperties": True},
            side_effect=SideEffect.WORKSPACE,
        )
    )
    registry.register(
        FunctionCapability(
            "ml.split.modify",
            "Policy-gated mutation attempt; DatasetVersion splits are immutable.",
            _policy_denial(
                action="ml.split.modify",
                reason_code="split_mutation_forbidden",
                reason="DatasetVersion split metadata is immutable",
            ),
            input_schema={"type": "object", "additionalProperties": True},
            side_effect=SideEffect.WORKSPACE,
        )
    )
    registry.register(
        FunctionCapability(
            "model.promote",
            "Policy-gated attempt; Production promotion requires human review.",
            _policy_denial(
                action="model.promote",
                reason_code="automatic_promotion_forbidden",
                reason="Production promotion requires an authorized human review",
            ),
            input_schema={"type": "object", "additionalProperties": True},
            side_effect=SideEffect.WORKSPACE,
        )
    )
    registry.register(
        FunctionCapability(
            "recipe.register",
            "Policy-gated attempt; RecipeRegistry is platform-owned and immutable.",
            _policy_denial(
                action="recipe.register",
                reason_code="recipe_registration_forbidden",
                reason="Agents cannot create, modify or register Recipes",
            ),
            input_schema={"type": "object", "additionalProperties": True},
            side_effect=SideEffect.WORKSPACE,
        )
    )
    registry.register(
        FunctionCapability(
            "training.code.generate",
            "Policy-gated attempt; arbitrary training code generation is disabled.",
            _policy_denial(
                action="training.code.generate",
                reason_code="training_code_generation_forbidden",
                reason="v0.2 only executes trusted Recipes and Trainer code",
            ),
            input_schema={"type": "object", "additionalProperties": True},
            side_effect=SideEffect.WORKSPACE,
        )
    )
    registry.register(
        FunctionCapability(
            "training.cancel",
            "Request graceful cancellation of one training job.",
            _training_cancel,
            input_schema={"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
            side_effect=SideEffect.EXTERNAL,
            idempotent=True,
        )
    )
    registry.register(
        FunctionCapability(
            "artifact.write",
            "Create a downloadable artifact from text output.",
            _artifact_write,
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                    "content_type": {"type": "string"},
                },
                "required": ["name", "content"],
            },
            side_effect=SideEffect.WORKSPACE,
        )
    )


async def _echo(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    return CapabilityResult(output=arguments)


def _policy_denial(
    *,
    action: str,
    reason_code: str,
    reason: str,
) -> Callable[[CapabilityContext, dict[str, Any]], Awaitable[CapabilityResult]]:
    async def handler(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
        experiment_id = None
        raw_experiment_id = arguments.get("experiment_id")
        if raw_experiment_id:
            try:
                experiment_id = uuid.UUID(str(raw_experiment_id))
            except ValueError:
                experiment_id = None
        async with get_session_factory()() as session:
            await record_rejection(
                session,
                workspace_id=context.workspace_id,
                experiment_id=experiment_id,
                action=action,
                reason_code=reason_code,
                reason=reason,
                actor="agent",
                run_id=context.run_id,
                node_run_id=context.node_run_id,
                request=dict(arguments),
            )
            await session.commit()
        raise PolicyViolation(reason_code, reason)

    return handler


async def _web_search(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    provider = context.service("search")
    results = await provider.search(arguments["query"], limit=arguments.get("limit", 5))
    return CapabilityResult(output={"results": results})


async def _knowledge_search(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    retrieval = context.service("retrieval")
    results = await retrieval.search(
        query=arguments["query"],
        knowledge_base_ids=context.service("knowledge_base_ids"),
        limit=arguments.get("limit", 8),
    )
    return CapabilityResult(output={"results": [item.as_dict() for item in results]})


async def _memory_search(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    memory = context.service("memory")
    async with get_session_factory()() as session:
        records = await memory.search(
            session,
            query=arguments["query"],
            workspace_id=context.workspace_id,
            agent_name=arguments.get("agent_name"),
            limit=arguments.get("limit", 8),
        )
    return CapabilityResult(
        output={
            "results": [{"id": str(record.id), "content": record.content, "kind": record.kind} for record in records]
        }
    )


async def _memory_write(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    memory = context.service("memory")
    async with get_session_factory()() as session:
        async with session.begin():
            record = await memory.create(
                session,
                workspace_id=context.workspace_id,
                content=arguments["content"],
                agent_name=arguments.get("agent_name"),
                kind=arguments.get("kind", "fact"),
                source_run_id=context.run_id,
            )
    return CapabilityResult(output={"id": str(record.id), "content": record.content})


async def _workspace_write(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    sandbox = context.service("sandbox")
    await sandbox.write_file(arguments["path"], arguments["content"])
    return CapabilityResult(output={"path": arguments["path"], "written": True})


async def _workspace_read(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    sandbox = context.service("sandbox")
    content = await sandbox.read_file(arguments["path"])
    return CapabilityResult(output={"path": arguments["path"], "content": content})


async def _workspace_list(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    sandbox = context.service("sandbox")
    files = await sandbox.list_dir(arguments.get("path", "."))
    return CapabilityResult(output={"files": files})


async def _sandbox_execute(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    sandbox = context.service("sandbox")
    for path, content in arguments.get("files", {}).items():
        await sandbox.write_file(path, content)
    result = await sandbox.execute(
        arguments["command"],
        timeout_seconds=arguments.get("timeout_seconds", 120),
        cwd=arguments.get("cwd"),
    )
    artifacts: list[dict[str, Any]] = []
    for path in arguments.get("capture", []):
        try:
            content = await sandbox.read_file(path)
        except Exception:
            continue
        artifacts.append(
            {
                "name": path.split("/")[-1],
                "content": content,
                "content_type": _guess_content_type(path),
            }
        )
    return CapabilityResult(
        output={
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
        },
        metrics={"exit_code": result.exit_code, "timed_out": result.timed_out},
        artifacts=artifacts,
    )


async def _artifact_write(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    return CapabilityResult(
        output={"name": arguments["name"], "size": len(arguments["content"].encode("utf-8"))},
        artifacts=[
            {
                "name": arguments["name"],
                "content": arguments["content"],
                "content_type": arguments.get("content_type", "text/plain; charset=utf-8"),
            }
        ],
    )


def _guess_content_type(path: str) -> str:
    suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return {
        "md": "text/markdown; charset=utf-8",
        "txt": "text/plain; charset=utf-8",
        "json": "application/json",
        "csv": "text/csv; charset=utf-8",
        "svg": "image/svg+xml",
        "html": "text/html; charset=utf-8",
    }.get(suffix, "application/octet-stream")


async def _ml_dataset_inspect(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    async with get_session_factory()() as session:
        version = await session.get(DatasetVersion, uuid.UUID(arguments["dataset_version_id"]))
        if version is None:
            raise ValueError("dataset version not found")
        dataset = await session.get(Dataset, version.dataset_id)
        if dataset is None or dataset.workspace_id != context.workspace_id:
            raise ValueError("dataset version not found in workspace")
        return CapabilityResult(
            output={
                "id": str(version.id),
                "dataset_id": str(dataset.id),
                "name": dataset.name,
                "kind": dataset.kind,
                "task_type": dataset.task_type,
                "status": version.status,
                "checksum": version.checksum,
                "split_checksum": version.split_checksum,
                "split_algorithm": version.split_algorithm,
                "split_seed": version.split_seed,
                "stratification": version.stratification,
                "profile": version.profile,
                "target_column": version.target_column,
            }
        )


async def _ml_recipe_list(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    del context, arguments
    return CapabilityResult(
        output={
            "recipes": [
                {
                    "recipe_id": recipe.qualified_id,
                    "display_name": recipe.display_name,
                    "backend": recipe.backend,
                    "task_types": [item.value for item in recipe.task_types],
                    "dataset_kinds": [item.value for item in recipe.dataset_kinds],
                    "default_metric": recipe.default_metric,
                    "default_direction": recipe.default_direction.value,
                    "checksum": recipe.checksum,
                    "hyperparameters": [item.model_dump(mode="json") for item in recipe.hyperparameters],
                    "resources": recipe.resources.model_dump(mode="json"),
                }
                for recipe in get_recipe_registry().list()
            ]
        }
    )


async def _ml_experiment_create(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    service = TrainingService()
    budget = ExperimentBudget.model_validate(arguments.get("budget", {}))
    policy_data = arguments.get("selection_policy")
    policy = SelectionPolicy.model_validate(policy_data) if policy_data else None
    async with get_session_factory()() as session:
        async with session.begin():
            experiment = await service.create_experiment(
                session,
                workspace_id=context.workspace_id,
                dataset_version_id=uuid.UUID(arguments["dataset_version_id"]),
                name=arguments["name"],
                baseline_strategy_id=arguments.get("baseline_strategy_id"),
                budget=budget,
                search_strategy=arguments.get("search_strategy"),
                selection_policy=policy,
                created_by=context.actor_id,
                run_id=context.run_id,
            )
        return CapabilityResult(
            output={
                "experiment_id": str(experiment.id),
                "status": experiment.status,
                "baseline_job_id": str(experiment.baseline_job_id),
                "budget": budget.model_dump(mode="json"),
            }
        )


async def _training_submit_batch(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    service = TrainingService()
    jobs = [TrainingJobSpec.model_validate(item) for item in arguments["jobs"]]
    async with get_session_factory()() as session:
        async with session.begin():
            experiment = await session.get(Experiment, uuid.UUID(arguments["experiment_id"]))
            if experiment is None or experiment.workspace_id != context.workspace_id:
                raise ValueError("experiment not found")
            created = await service.submit_batch(
                session,
                workspace_id=context.workspace_id,
                experiment_id=experiment.id,
                jobs=jobs,
                round_number=int(arguments.get("round_number", 1)),
                submitted_by=context.actor_id,
            )
        return CapabilityResult(
            output={
                "job_ids": [str(job.id) for job in created],
                "status": [job.status for job in created],
                "round": int(arguments.get("round_number", 1)),
            }
        )


async def _training_await_batch(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    service = TrainingService()
    job_ids = [uuid.UUID(item) for item in arguments["job_ids"]]
    jobs = await service.await_jobs(
        job_ids=job_ids,
        timeout_seconds=float(arguments.get("timeout_seconds", 900)),
        workspace_id=context.workspace_id,
    )
    async with get_session_factory()() as session:
        results = await service.validation_results(
            session,
            [job.id for job in jobs],
            workspace_id=context.workspace_id,
        )
    return CapabilityResult(output={"jobs": results})


async def _training_await_experiment(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    service = TrainingService()
    result = await service.await_experiment_finalization(
        experiment_id=uuid.UUID(arguments["experiment_id"]),
        timeout_seconds=float(arguments.get("timeout_seconds", 900)),
        workspace_id=context.workspace_id,
    )
    return CapabilityResult(output=result)


async def _training_results(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    service = TrainingService()
    job_ids = [uuid.UUID(item) for item in arguments["job_ids"]]
    async with get_session_factory()() as session:
        results = await service.validation_results(
            session,
            job_ids,
            workspace_id=context.workspace_id,
        )
    return CapabilityResult(output={"jobs": results})


async def _training_test_metrics(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    service = TrainingService()
    experiment_id = uuid.UUID(arguments["experiment_id"]) if arguments.get("experiment_id") else None
    async with get_session_factory()() as session:
        if experiment_id is not None:
            experiment = await session.get(Experiment, experiment_id)
            if experiment is None or experiment.workspace_id != context.workspace_id:
                experiment_id = None
        await service.reject_test_metric_access(
            session,
            workspace_id=context.workspace_id,
            experiment_id=experiment_id,
            actor="agent",
            run_id=context.run_id,
            node_run_id=context.node_run_id,
        )
        await session.commit()
    raise PolicyViolation(
        "test_metrics_forbidden",
        "test metrics are platform-only final evaluation metadata and cannot be read by Agent",
    )


async def _model_select_best(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    service = TrainingService()
    async with get_session_factory()() as session:
        async with session.begin():
            job = await service.select_best(
                session,
                experiment_id=uuid.UUID(arguments["experiment_id"]),
                workspace_id=context.workspace_id,
                actor="agent",
            )
            experiment = await session.get(Experiment, job.experiment_id)
            result = (
                await service.validation_results(
                    session,
                    [job.id],
                    workspace_id=context.workspace_id,
                )
            )[0]
        return CapabilityResult(
            output={
                "experiment_id": str(job.experiment_id),
                "best_job_id": str(job.id),
                "selection_reason": experiment.selection_reason if experiment else None,
                "validation_metrics": result["validation"],
                "bundle_artifact_id": result.get("bundle_artifact_id"),
            }
        )


async def _model_register_candidate(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    service = TrainingService()
    async with get_session_factory()() as session:
        async with session.begin():
            model_version = await service.register_candidate(
                session,
                experiment_id=uuid.UUID(arguments["experiment_id"]),
                workspace_id=context.workspace_id,
                artifact_id=uuid.UUID(arguments["artifact_id"]),
                actor="agent",
            )
        return CapabilityResult(
            output={
                "model_version_id": str(model_version.id),
                "name": model_version.name,
                "version": model_version.version,
                "stage": model_version.stage,
                "bundle_sha256": model_version.bundle_sha256,
            }
        )


async def _training_cancel(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
    job_id = uuid.UUID(arguments["job_id"])
    async with get_session_factory()() as session:
        async with session.begin():
            job = await session.get(TrainingJob, job_id, with_for_update=True)
            if job is None or job.workspace_id != context.workspace_id:
                raise ValueError("training job not found")
            if job.status in {"succeeded", "failed", "cancelled"}:
                return CapabilityResult(output={"job_id": str(job.id), "status": job.status})
            job.status = "cancelling"
            await TrainingService().record_cancel_request(
                session,
                job=job,
                actor="agent",
            )
            await get_redis().set(f"agentforge:training:cancel:{job.id}", "1", ex=86_400)
    return CapabilityResult(output={"job_id": str(job_id), "status": "cancelling"})
