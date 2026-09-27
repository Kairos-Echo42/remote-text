from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentforge.models import NodeRun, OutboxEvent, Run, WorkflowDefinition, WorkflowVersion
from agentforge.runtime.state import NodeStatus, RunStatus
from agentforge.workflow import WorkflowDocument


class RunNotFoundError(LookupError):
    pass


class InvalidRunState(ValueError):
    pass


async def create_run(
    session: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    workflow_version_id: uuid.UUID,
    input_data: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    created_by: uuid.UUID | None = None,
) -> Run:
    version = await session.scalar(
        select(WorkflowVersion)
        .join(WorkflowDefinition, WorkflowVersion.workflow_id == WorkflowDefinition.id)
        .where(
            WorkflowVersion.id == workflow_version_id,
            WorkflowDefinition.workspace_id == workspace_id,
        )
    )
    if version is None:
        raise RunNotFoundError("workflow version not found in workspace")

    document = WorkflowDocument.model_validate(version.definition)
    run = Run(
        workspace_id=workspace_id,
        workflow_version_id=version.id,
        created_by=created_by,
        status=RunStatus.QUEUED.value,
        input=input_data,
        run_metadata=metadata or {},
        started_at=datetime.now(UTC),
    )
    session.add(run)
    await session.flush()

    for node in document.spec.nodes:
        session.add(
            NodeRun(
                run_id=run.id,
                node_id=node.id,
                node_type=node.type,
                attempt=1,
                status=NodeStatus.PENDING.value,
            )
        )
    await session.flush()
    return run


async def request_pause(session: AsyncSession, run_id: uuid.UUID, workspace_id: uuid.UUID) -> Run:
    run = await get_run(session, run_id, workspace_id)
    if run.status != RunStatus.RUNNING.value:
        raise InvalidRunState("only running runs can be paused")
    run.status = RunStatus.PAUSED.value
    return run


async def request_resume(session: AsyncSession, run_id: uuid.UUID, workspace_id: uuid.UUID) -> Run:
    run = await get_run(session, run_id, workspace_id)
    if run.status != RunStatus.PAUSED.value:
        raise InvalidRunState("only paused runs can be resumed")
    run.status = RunStatus.QUEUED.value
    return run


async def request_cancel(session: AsyncSession, run_id: uuid.UUID, workspace_id: uuid.UUID) -> Run:
    run = await get_run(session, run_id, workspace_id)
    if run.status in {"succeeded", "failed", "cancelled"}:
        return run
    run.cancel_requested = True
    run.status = RunStatus.CANCELLING.value
    return run


async def request_retry(
    session: AsyncSession,
    run_id: uuid.UUID,
    workspace_id: uuid.UUID,
    *,
    node_id: str | None = None,
) -> NodeRun | Run:
    run = await get_run(session, run_id, workspace_id)
    if node_id is not None:
        failed = await session.scalar(
            select(NodeRun)
            .where(
                NodeRun.run_id == run.id,
                NodeRun.node_id == node_id,
                NodeRun.status == NodeStatus.FAILED.value,
            )
            .order_by(NodeRun.attempt.desc())
        )
        if failed is None:
            raise InvalidRunState(f"no failed attempt found for node {node_id}")
        next_attempt = NodeRun(
            run_id=run.id,
            node_id=failed.node_id,
            node_type=failed.node_type,
            attempt=failed.attempt + 1,
            status=NodeStatus.READY.value,
            ready_at=datetime.now(UTC),
        )
        session.add(next_attempt)
        await session.flush()
        session.add(
            OutboxEvent(
                topic="node.ready",
                aggregate_id=run.id,
                payload={"node_run_id": str(next_attempt.id), "node_id": node_id},
            )
        )
        run.status = RunStatus.QUEUED.value
        run.cancel_requested = False
        run.finished_at = None
        run.error = None
        return next_attempt
    if run.status != RunStatus.FAILED.value:
        raise InvalidRunState("only failed runs can be retried")
    run.status = RunStatus.QUEUED.value
    run.cancel_requested = False
    run.error = None
    run.finished_at = None
    return run


async def get_run(session: AsyncSession, run_id: uuid.UUID, workspace_id: uuid.UUID) -> Run:
    run = await session.scalar(select(Run).where(Run.id == run_id, Run.workspace_id == workspace_id))
    if run is None:
        raise RunNotFoundError("run not found")
    return run
