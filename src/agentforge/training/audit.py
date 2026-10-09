from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentforge.models import DecisionRecord, LineageEdge, LineageNode
from agentforge.training.types import DecisionRecordData, DecisionResult


class PolicyViolation(ValueError):
    def __init__(self, reason_code: str, reason: str):
        self.reason_code = reason_code
        self.reason = reason
        super().__init__(reason)


async def record_decision(
    session: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    data: DecisionRecordData,
    experiment_id: uuid.UUID | None = None,
    job_id: uuid.UUID | None = None,
) -> DecisionRecord:
    record = DecisionRecord(
        workspace_id=workspace_id,
        experiment_id=experiment_id,
        job_id=job_id,
        run_id=data.run_id,
        node_run_id=data.node_run_id,
        phase=data.phase,
        actor=data.actor,
        action=data.action,
        result=data.result.value,
        reason_code=data.reason_code,
        reason=data.reason,
        request=data.request,
        validation_metrics=data.validation_metrics,
        budget_snapshot=data.budget_snapshot,
        config_checksum=data.config_checksum,
        created_at=data.timestamp,
    )
    session.add(record)
    await session.flush()
    return record


async def record_rejection(
    session: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    action: str,
    reason_code: str,
    reason: str,
    actor: str,
    request: dict[str, Any] | None = None,
    experiment_id: uuid.UUID | None = None,
    job_id: uuid.UUID | None = None,
    run_id: uuid.UUID | None = None,
    node_run_id: uuid.UUID | None = None,
    phase: str | None = None,
    budget_snapshot: dict[str, Any] | None = None,
    config_checksum: str | None = None,
) -> DecisionRecord:
    return await record_decision(
        session,
        workspace_id=workspace_id,
        experiment_id=experiment_id,
        job_id=job_id,
        data=DecisionRecordData(
            action=action,
            result=DecisionResult.REJECTED,
            reason_code=reason_code,
            reason=reason,
            timestamp=datetime.now(UTC),
            actor=actor,
            phase=phase,
            run_id=run_id,
            node_run_id=node_run_id,
            request=request,
            budget_snapshot=budget_snapshot,
            config_checksum=config_checksum,
        ),
    )


async def ensure_lineage_node(
    session: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    node_type: str,
    ref_id: uuid.UUID,
    artifact_id: uuid.UUID | None = None,
    metadata: dict[str, Any] | None = None,
) -> LineageNode:
    node = await session.scalar(
        select(LineageNode).where(
            LineageNode.workspace_id == workspace_id,
            LineageNode.node_type == node_type,
            LineageNode.ref_id == ref_id,
        )
    )
    if node is None:
        node = LineageNode(
            workspace_id=workspace_id,
            node_type=node_type,
            ref_id=ref_id,
            artifact_id=artifact_id,
            metadata_=metadata or {},
        )
        session.add(node)
        await session.flush()
    return node


async def link_lineage(
    session: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    input_node: LineageNode,
    output_node: LineageNode,
    operation: str,
    actor_type: str,
    actor_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> LineageEdge:
    edge = LineageEdge(
        workspace_id=workspace_id,
        input_node_id=input_node.id,
        output_node_id=output_node.id,
        operation=operation,
        actor_type=actor_type,
        actor_id=actor_id,
        metadata_=metadata or {},
        created_at=datetime.now(UTC),
    )
    session.add(edge)
    await session.flush()
    return edge
