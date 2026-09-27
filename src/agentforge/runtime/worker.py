from __future__ import annotations

import asyncio
import os
import socket
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from agentforge.config import get_settings
from agentforge.db import get_session_factory
from agentforge.logging import get_logger
from agentforge.models import NodeRun, Run, WorkflowVersion
from agentforge.runtime.events import EventService
from agentforge.runtime.queue import RedisStreamQueue
from agentforge.runtime.state import NodeStatus
from agentforge.workflow import WorkflowDocument
from agentforge.workflow.dsl import NodeSpec

logger = get_logger(__name__)


@dataclass(slots=True)
class NodeExecutionResult:
    output: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


class NodeExecutor(Protocol):
    async def execute(
        self,
        *,
        run: Run,
        node_run: NodeRun,
        node_spec: NodeSpec,
        document: WorkflowDocument,
    ) -> NodeExecutionResult: ...

    async def cancel(self, node_run_id: uuid.UUID) -> None: ...


class Worker:
    def __init__(
        self,
        executor: NodeExecutor,
        *,
        queue: RedisStreamQueue | None = None,
        event_service: EventService | None = None,
        consumer_name: str | None = None,
    ):
        self.executor = executor
        self.queue = queue or RedisStreamQueue()
        self.events = event_service or EventService(self.queue)
        self.consumer_name = consumer_name or f"{socket.gethostname()}-{os.getpid()}"

    async def run_forever(self) -> None:
        await self.queue.ensure_worker_group()
        while True:
            try:
                stale = await self.queue.claim_stale(
                    self.consumer_name,
                    min_idle_ms=get_settings().worker_claim_idle_ms,
                )
                if stale:
                    for message_id, fields in stale:
                        await self._handle(message_id, fields)
                    continue
                messages = await self.queue.consume_node(self.consumer_name, block_ms=5_000)
                for message_id, fields in messages:
                    await self._handle(message_id, fields)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("worker_loop_failed")
                await asyncio.sleep(1)

    async def _handle(self, message_id: str, fields: dict[str, str]) -> None:
        node_run_id = uuid.UUID(fields["node_run_id"])
        try:
            should_execute = await self._claim(node_run_id)
            if should_execute:
                await self._execute(node_run_id)
        except asyncio.CancelledError:
            await self._fail(node_run_id, "worker cancelled", cancelled=True)
            raise
        except Exception as exc:
            logger.exception("node_execution_failed", node_run_id=str(node_run_id))
            await self._fail(node_run_id, str(exc))
        finally:
            await self.queue.acknowledge_node(message_id)

    async def _claim(self, node_run_id: uuid.UUID) -> bool:
        now = datetime.now(UTC)
        factory = get_session_factory()
        async with factory() as session:
            async with session.begin():
                node_run = await session.get(NodeRun, node_run_id)
                if node_run is None or node_run.status != NodeStatus.READY.value:
                    return False
                await session.get(Run, node_run.run_id, with_for_update=True)
                node_run = await session.get(NodeRun, node_run_id, with_for_update=True)
                if node_run is None or node_run.status != NodeStatus.READY.value:
                    return False
                node_run.status = NodeStatus.RUNNING.value
                node_run.lease_owner = self.consumer_name
                node_run.lease_expires_at = now + timedelta(seconds=get_settings().worker_lease_seconds)
                node_run.heartbeat_at = now
                node_run.started_at = node_run.started_at or now
                await self.events.append(
                    session,
                    node_run.run_id,
                    "node.started",
                    {"node_id": node_run.node_id, "attempt": node_run.attempt},
                    node_run_id=node_run.id,
                )
        return True

    async def _execute(self, node_run_id: uuid.UUID) -> None:
        factory = get_session_factory()
        async with factory() as session:
            node_run = await session.get(NodeRun, node_run_id)
            if node_run is None:
                return
            run = await session.get(Run, node_run.run_id)
            if run is None:
                raise RuntimeError("run disappeared")
            version = await session.get(WorkflowVersion, run.workflow_version_id)
            if version is None:
                raise RuntimeError("workflow version disappeared")
            document = WorkflowDocument.model_validate(version.definition)
            node_spec = next(node for node in document.spec.nodes if node.id == node_run.node_id)
        task = asyncio.create_task(
            self.executor.execute(
                run=run,
                node_run=node_run,
                node_spec=node_spec,
                document=document,
            )
        )
        cancellation_key = f"agentforge:cancel:{run.id}"
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.5)
            if task in done:
                break
            try:
                if await self.queue.client.get(cancellation_key):
                    await self.executor.cancel(node_run_id)
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    raise asyncio.CancelledError
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        result = await task
        await self._succeed(node_run_id, result)

    async def _succeed(self, node_run_id: uuid.UUID, result: NodeExecutionResult) -> None:
        factory = get_session_factory()
        async with factory() as session:
            async with session.begin():
                node_run = await session.get(NodeRun, node_run_id)
                if node_run is None or node_run.status != NodeStatus.RUNNING.value:
                    return
                await session.get(Run, node_run.run_id, with_for_update=True)
                node_run = await session.get(NodeRun, node_run_id, with_for_update=True)
                if node_run is None or node_run.status != NodeStatus.RUNNING.value:
                    return
                node_run.status = NodeStatus.SUCCEEDED.value
                node_run.output = result.output
                node_run.metrics = result.metrics
                node_run.finished_at = datetime.now(UTC)
                node_run.lease_owner = None
                node_run.lease_expires_at = None
                await self.events.append(
                    session,
                    node_run.run_id,
                    "node.succeeded",
                    {"node_id": node_run.node_id, "metrics": result.metrics},
                    node_run_id=node_run.id,
                )

    async def _fail(
        self,
        node_run_id: uuid.UUID,
        error: str,
        *,
        cancelled: bool = False,
    ) -> None:
        factory = get_session_factory()
        async with factory() as session:
            async with session.begin():
                node_run = await session.get(NodeRun, node_run_id)
                if node_run is None or node_run.status != NodeStatus.RUNNING.value:
                    return
                run = await session.get(Run, node_run.run_id, with_for_update=True)
                node_run = await session.get(NodeRun, node_run_id, with_for_update=True)
                if node_run is None or node_run.status != NodeStatus.RUNNING.value:
                    return
                if run is None:
                    return
                version = await session.get(WorkflowVersion, run.workflow_version_id)
                if version is None:
                    raise RuntimeError("workflow version disappeared")
                document = WorkflowDocument.model_validate(version.definition)
                spec = next(node for node in document.spec.nodes if node.id == node_run.node_id)
                node_run.lease_owner = None
                node_run.lease_expires_at = None
                if cancelled or run.cancel_requested:
                    node_run.status = NodeStatus.CANCELLED.value
                    node_run.error = error
                    node_run.finished_at = datetime.now(UTC)
                    await self.events.append(
                        session,
                        run.id,
                        "node.cancelled",
                        {"node_id": node_run.node_id, "error": error},
                        node_run_id=node_run.id,
                    )
                elif node_run.attempt < spec.retry.max_attempts:
                    delay = min(
                        spec.retry.base_delay_seconds * (2 ** (node_run.attempt - 1)),
                        spec.retry.max_delay_seconds,
                    )
                    node_run.status = NodeStatus.FAILED.value
                    node_run.error = error
                    node_run.finished_at = datetime.now(UTC)
                    retry = NodeRun(
                        run_id=node_run.run_id,
                        node_id=node_run.node_id,
                        node_type=node_run.node_type,
                        attempt=node_run.attempt + 1,
                        status=NodeStatus.RETRY_WAIT.value,
                        ready_at=datetime.now(UTC) + timedelta(seconds=delay),
                    )
                    session.add(retry)
                    await self.events.append(
                        session,
                        run.id,
                        "node.retry_scheduled",
                        {
                            "node_id": node_run.node_id,
                            "attempt": retry.attempt,
                            "delay_seconds": delay,
                        },
                        node_run_id=node_run.id,
                    )
                else:
                    node_run.status = NodeStatus.FAILED.value
                    node_run.error = error
                    node_run.finished_at = datetime.now(UTC)
                    await self.events.append(
                        session,
                        run.id,
                        "node.failed",
                        {"node_id": node_run.node_id, "error": error},
                        node_run_id=node_run.id,
                    )


class NodeExecutorRegistry:
    """Tracks active node executions so cancellation can be best-effort propagated."""

    def __init__(self) -> None:
        self._tasks: dict[uuid.UUID, asyncio.Task[Any]] = {}

    def register(self, node_run_id: uuid.UUID, task: asyncio.Task[Any]) -> None:
        self._tasks[node_run_id] = task

    def unregister(self, node_run_id: uuid.UUID) -> None:
        self._tasks.pop(node_run_id, None)

    async def cancel(self, node_run_id: uuid.UUID) -> None:
        task = self._tasks.get(node_run_id)
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
