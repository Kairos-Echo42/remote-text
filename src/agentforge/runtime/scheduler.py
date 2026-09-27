from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentforge.config import get_settings
from agentforge.db import get_session_factory
from agentforge.logging import get_logger
from agentforge.models import NodeRun, OutboxEvent, Run, WorkflowVersion
from agentforge.runtime.events import EventService, OutboxPublisher
from agentforge.runtime.queue import RedisStreamQueue
from agentforge.runtime.state import NODE_TERMINAL_STATES, NodeStatus, RunStatus
from agentforge.workflow import WorkflowDocument
from agentforge.workflow.conditions import (
    build_evaluation_context,
    evaluate_condition,
    resolve_mapping,
)
from agentforge.workflow.validator import resolve_dependencies

logger = get_logger(__name__)


@dataclass(slots=True)
class PendingPublish:
    run_id: uuid.UUID
    event_type: str
    payload: dict[str, Any]
    node_run_id: uuid.UUID | None = None


class DAGScheduler:
    def __init__(
        self,
        *,
        queue: RedisStreamQueue | None = None,
        event_service: EventService | None = None,
    ):
        self.queue = queue or RedisStreamQueue()
        self.events = event_service or EventService(self.queue)
        self.outbox = OutboxPublisher(self.queue)

    async def tick(self, *, limit: int = 20) -> int:
        advanced = 0
        factory = get_session_factory()
        publications: list[PendingPublish] = []
        async with factory() as session:
            async with session.begin():
                runs = list(
                    (
                        await session.scalars(
                            select(Run)
                            .where(
                                Run.status.in_(
                                    [
                                        RunStatus.QUEUED.value,
                                        RunStatus.RUNNING.value,
                                        RunStatus.PAUSED.value,
                                        RunStatus.CANCELLING.value,
                                    ]
                                )
                            )
                            .order_by(Run.created_at)
                            .limit(limit)
                            .with_for_update(skip_locked=True)
                        )
                    ).all()
                )
                for run in runs:
                    changed = await self._advance_run(session, run, publications)
                    advanced += int(changed)
        if publications:
            await self._publish_events(publications)
        await self.outbox.publish_pending(limit=200)
        return advanced

    async def _advance_run(self, session: AsyncSession, run: Run, publications: list[PendingPublish]) -> bool:
        version = await session.get(WorkflowVersion, run.workflow_version_id)
        if version is None:
            run.status = RunStatus.FAILED.value
            run.error = "workflow version disappeared"
            run.finished_at = datetime.now(UTC)
            return True
        document = WorkflowDocument.model_validate(version.definition)
        await self._recover_expired_leases(session, run.id, publications)
        nodes = list(
            (await session.scalars(select(NodeRun).where(NodeRun.run_id == run.id).order_by(NodeRun.attempt))).all()
        )
        latest = _latest_node_attempts(nodes)
        changed = False

        if run.cancel_requested:
            for node in latest.values():
                if node.status not in NODE_TERMINAL_STATES:
                    node.status = NodeStatus.CANCELLED.value
                    node.finished_at = datetime.now(UTC)
                    _queue_event(
                        publications,
                        run.id,
                        "node.cancelled",
                        {"node_id": node.node_id},
                        node.id,
                    )
            run.status = RunStatus.CANCELLED.value
            run.finished_at = datetime.now(UTC)
            _queue_event(publications, run.id, "run.cancelled", {})
            return True

        if run.status == RunStatus.QUEUED.value:
            run.status = RunStatus.RUNNING.value
            _queue_event(publications, run.id, "run.started", {})
            changed = True

        dependencies = resolve_dependencies(document)
        specs = {node.id: node for node in document.spec.nodes}
        outputs = {
            node.node_id: node.output or {} for node in latest.values() if node.status == NodeStatus.SUCCEEDED.value
        }
        context = build_evaluation_context(run_input=run.input, node_outputs=outputs, run_metadata=run.run_metadata)
        limit = min(document.spec.max_concurrency, get_settings().run_max_concurrency)
        active_count = sum(
            node.status in {NodeStatus.RUNNING.value, NodeStatus.READY.value} for node in latest.values()
        )

        for node_id, node_run in latest.items():
            if node_run.status not in {NodeStatus.PENDING.value, NodeStatus.RETRY_WAIT.value}:
                continue
            if node_run.ready_at is not None and _as_utc(node_run.ready_at) > datetime.now(UTC):
                continue
            dependencies_for_node = dependencies.get(node_id, set())
            dependency_states = {
                dependency: latest[dependency].status for dependency in dependencies_for_node if dependency in latest
            }
            failed_dependencies = [
                dependency
                for dependency, state in dependency_states.items()
                if state in {NodeStatus.FAILED.value, NodeStatus.CANCELLED.value}
            ]
            if failed_dependencies and not specs[node_id].continue_on_error:
                node_run.status = NodeStatus.SKIPPED.value
                node_run.error = f"dependencies failed: {', '.join(sorted(failed_dependencies))}"
                node_run.finished_at = datetime.now(UTC)
                _queue_event(
                    publications,
                    run.id,
                    "node.skipped",
                    {"node_id": node_id, "reason": node_run.error},
                    node_run.id,
                )
                changed = True
                continue
            if any(
                state not in {NodeStatus.SUCCEEDED.value, NodeStatus.SKIPPED.value}
                for state in dependency_states.values()
            ):
                continue
            if run.status == RunStatus.PAUSED.value:
                continue
            if not evaluate_condition(specs[node_id].when, context):
                node_run.status = NodeStatus.SKIPPED.value
                node_run.finished_at = datetime.now(UTC)
                _queue_event(
                    publications,
                    run.id,
                    "node.skipped",
                    {"node_id": node_id, "reason": "condition is false"},
                    node_run.id,
                )
                changed = True
                continue
            if active_count >= limit:
                continue
            node_run.status = NodeStatus.READY.value
            node_run.ready_at = datetime.now(UTC)
            session.add(
                OutboxEvent(
                    topic="node.ready",
                    aggregate_id=run.id,
                    payload={"node_run_id": str(node_run.id), "node_id": node_id},
                )
            )
            _queue_event(publications, run.id, "node.ready", {"node_id": node_id}, node_run.id)
            active_count += 1
            changed = True

        await session.flush()
        refreshed = list(
            (await session.scalars(select(NodeRun).where(NodeRun.run_id == run.id).order_by(NodeRun.attempt))).all()
        )
        latest = _latest_node_attempts(refreshed)
        if latest and all(node.status in NODE_TERMINAL_STATES for node in latest.values()):
            failed = [node for node in latest.values() if node.status == NodeStatus.FAILED.value]
            final_outputs = {
                node.node_id: node.output or {} for node in latest.values() if node.status == NodeStatus.SUCCEEDED.value
            }
            final_context = build_evaluation_context(
                run_input=run.input,
                node_outputs=final_outputs,
                run_metadata=run.run_metadata,
            )
            run.output = resolve_mapping(document.spec.output, final_context)
            run.finished_at = datetime.now(UTC)
            if failed:
                run.status = RunStatus.FAILED.value
                run.error = failed[0].error or f"node {failed[0].node_id} failed"
                _queue_event(
                    publications,
                    run.id,
                    "run.failed",
                    {"error": run.error, "node_id": failed[0].node_id},
                )
            else:
                run.status = RunStatus.SUCCEEDED.value
                from agentforge.services.memory import MemoryService
                from agentforge.services.models import resolve_embedding_provider

                embedding_provider = await resolve_embedding_provider(session, run.workspace_id)
                await MemoryService(embedding_provider).extract_from_run(session, run)
                _queue_event(publications, run.id, "run.succeeded", {"output": run.output})
            changed = True
        elif run.status == RunStatus.CANCELLING.value:
            run.status = RunStatus.RUNNING.value
        return changed

    async def _recover_expired_leases(
        self,
        session: AsyncSession,
        run_id: uuid.UUID,
        publications: list[PendingPublish],
    ) -> None:
        now = datetime.now(UTC)
        nodes = list(
            (
                await session.scalars(
                    select(NodeRun)
                    .where(
                        NodeRun.run_id == run_id,
                        NodeRun.status == NodeStatus.RUNNING.value,
                        NodeRun.lease_expires_at.is_not(None),
                    )
                    .with_for_update(skip_locked=True)
                )
            ).all()
        )
        for node in nodes:
            if node.lease_expires_at is None or _as_utc(node.lease_expires_at) >= now:
                continue
            node.status = NodeStatus.READY.value
            node.ready_at = now
            node.lease_owner = None
            node.lease_expires_at = None
            node.heartbeat_at = None
            session.add(
                OutboxEvent(
                    topic="node.ready",
                    aggregate_id=run_id,
                    payload={"node_run_id": str(node.id), "node_id": node.node_id, "recovered": True},
                )
            )
            _queue_event(
                publications,
                run_id,
                "node.recovered",
                {"node_id": node.node_id, "attempt": node.attempt},
                node.id,
            )

    async def _publish_events(self, publications: list[PendingPublish]) -> None:
        for item in publications:
            try:
                await self.events.emit(
                    item.run_id,
                    item.event_type,
                    item.payload,
                    node_run_id=item.node_run_id,
                )
            except Exception:
                logger.exception("event_publish_failed", run_id=str(item.run_id))

    async def run_forever(self, *, interval_seconds: float = 0.5) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("scheduler_tick_failed")
            await asyncio.sleep(interval_seconds)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _latest_node_attempts(nodes: list[NodeRun]) -> dict[str, NodeRun]:
    latest: dict[str, NodeRun] = {}
    for node in nodes:
        current = latest.get(node.node_id)
        if current is None or node.attempt > current.attempt:
            latest[node.node_id] = node
    return latest


def _queue_event(
    publications: list[PendingPublish],
    run_id: uuid.UUID,
    event_type: str,
    payload: dict[str, Any],
    node_run_id: uuid.UUID | None = None,
) -> None:
    publications.append(PendingPublish(run_id, event_type, payload, node_run_id))


async def run_scheduler() -> None:
    scheduler = DAGScheduler()
    await scheduler.run_forever()
