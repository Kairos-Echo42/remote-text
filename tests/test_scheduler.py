from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from agentforge.db import get_session_factory
from agentforge.models import NodeRun, Run, User, Workspace
from agentforge.runtime.events import EventService
from agentforge.runtime.scheduler import DAGScheduler
from agentforge.runtime.state import NodeStatus
from agentforge.runtime.worker import NodeExecutionResult, Worker
from agentforge.services.runs import create_run
from agentforge.services.workflows import sync_workflow
from agentforge.workflow import WorkflowBuilder


class FakeQueue:
    def __init__(self):
        self.messages: list[str] = []

    async def ensure_worker_group(self):
        return None

    async def publish_node(self, node_run_id: str, payload=None):
        self.messages.append(node_run_id)
        return "1-0"

    async def acknowledge_node(self, message_id: str):
        return 1

    async def publish_event(self, run_id: str, fields, **kwargs):
        return "1-0"

    async def claim_stale(self, *args, **kwargs):
        return []

    async def consume_node(self, *args, **kwargs):
        return []


class StubExecutor:
    def __init__(self, failures: int = 0):
        self.failures = failures
        self.calls: list[str] = []

    async def execute(self, *, run, node_run, node_spec, document):
        self.calls.append(node_run.node_id)
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("transient failure")
        return NodeExecutionResult(
            output={"node": node_run.node_id, "value": 1},
            metrics={"attempt": node_run.attempt},
        )

    async def cancel(self, node_run_id: uuid.UUID) -> None:
        return None


@pytest.mark.asyncio
async def test_scheduler_and_worker_execute_parallel_fanout_without_duplicates():
    factory = get_session_factory()
    async with factory() as session:
        user = await session.scalar(select(User).limit(1))
        workspace = await session.scalar(select(Workspace).limit(1))
        document = (
            WorkflowBuilder("fanout", "1.0.0")
            .max_concurrency(2)
            .node("start", "capability", "fixture.echo", inputs={"value": "$.input.value"})
            .node(
                "left",
                "capability",
                "fixture.echo",
                needs=["start"],
                inputs={"value": "$.nodes.start.output.value"},
            )
            .node(
                "right",
                "capability",
                "fixture.echo",
                needs=["start"],
                inputs={"value": "$.nodes.start.output.value"},
            )
            .node(
                "join",
                "capability",
                "fixture.echo",
                needs=["left", "right"],
                inputs={"value": "$.input.value"},
            )
            .output({"left": "$.nodes.left.output.node", "right": "$.nodes.right.output.node"})
            .build()
        )
        _, version = await sync_workflow(session, workspace.id, document)
        run = await create_run(
            session,
            workspace_id=workspace.id,
            workflow_version_id=version.id,
            input_data={"value": 7},
            created_by=user.id,
        )
        await session.commit()
        run_id = run.id

    queue = FakeQueue()
    events = EventService(queue, publish=False)
    scheduler = DAGScheduler(queue=queue, event_service=events)
    executor = StubExecutor()
    worker = Worker(executor, queue=queue, event_service=events, consumer_name="test-worker")

    for index in range(30):
        await scheduler.tick()
        async with factory() as session:
            current = await session.get(Run, run_id)
            if current.status in {"succeeded", "failed", "cancelled"}:
                break
            ready = list(
                (
                    await session.scalars(
                        select(NodeRun).where(
                            NodeRun.run_id == run_id,
                            NodeRun.status == NodeStatus.READY.value,
                        )
                    )
                ).all()
            )
        for node in ready:
            await worker._handle(f"{index}-{node.id}", {"node_run_id": str(node.id)})
        await asyncio.sleep(0)

    async with factory() as session:
        current = await session.get(Run, run_id)
        nodes = list(
            (await session.scalars(select(NodeRun).where(NodeRun.run_id == run_id).order_by(NodeRun.created_at))).all()
        )
    assert current is not None and current.status == "succeeded"
    assert current.output == {"left": "left", "right": "right"}
    assert sorted(executor.calls) == ["join", "left", "right", "start"]
    assert len(executor.calls) == len(set(executor.calls))
    assert all(node.status == "succeeded" for node in nodes)


@pytest.mark.asyncio
async def test_worker_creates_retry_attempt_then_succeeds():
    factory = get_session_factory()
    async with factory() as session:
        user = await session.scalar(select(User).limit(1))
        workspace = await session.scalar(select(Workspace).limit(1))
        document = (
            WorkflowBuilder("retry-once", "1.0.0")
            .node(
                "flaky",
                "capability",
                "fixture.echo",
                inputs={"value": "$.input.value"},
                max_attempts=2,
            )
            .output({"value": "$.nodes.flaky.output.value"})
            .build()
        )
        _, version = await sync_workflow(session, workspace.id, document)
        run = await create_run(
            session,
            workspace_id=workspace.id,
            workflow_version_id=version.id,
            input_data={"value": 9},
            created_by=user.id,
        )
        await session.commit()
        run_id = run.id

    queue = FakeQueue()
    events = EventService(queue, publish=False)
    scheduler = DAGScheduler(queue=queue, event_service=events)
    executor = StubExecutor(failures=1)
    worker = Worker(executor, queue=queue, event_service=events, consumer_name="retry-worker")

    await scheduler.tick()
    first = await _ready_node(factory, run_id)
    await worker._handle("1-0", {"node_run_id": str(first.id)})
    await asyncio.sleep(1.1)
    await scheduler.tick()
    second = await _ready_node(factory, run_id)
    assert second.attempt == 2
    await worker._handle("2-0", {"node_run_id": str(second.id)})
    await scheduler.tick()

    async with factory() as session:
        run = await session.get(Run, run_id)
        attempts = list(
            (
                await session.scalars(
                    select(NodeRun)
                    .where(NodeRun.run_id == run_id, NodeRun.node_id == "flaky")
                    .order_by(NodeRun.attempt)
                )
            ).all()
        )
    assert run is not None and run.status == "succeeded"
    assert [item.attempt for item in attempts] == [1, 2]
    assert [item.status for item in attempts] == ["failed", "succeeded"]


async def _ready_node(factory, run_id: uuid.UUID) -> NodeRun:
    async with factory() as session:
        node = await session.scalar(
            select(NodeRun)
            .where(NodeRun.run_id == run_id, NodeRun.status == NodeStatus.READY.value)
            .order_by(NodeRun.attempt.desc())
        )
        assert node is not None
        return node


@pytest.mark.asyncio
async def test_expired_running_lease_is_recovered_and_requeued():
    factory = get_session_factory()
    async with factory() as session:
        user = await session.scalar(select(User).limit(1))
        workspace = await session.scalar(select(Workspace).limit(1))
        document = (
            WorkflowBuilder("lease-recovery", "1.0.0")
            .node("work", "capability", "fixture.echo", inputs={"value": "$.input.value"})
            .output({"value": "$.nodes.work.output.value"})
            .build()
        )
        _, version = await sync_workflow(session, workspace.id, document)
        run = await create_run(
            session,
            workspace_id=workspace.id,
            workflow_version_id=version.id,
            input_data={"value": 3},
            created_by=user.id,
        )
        await session.commit()
        run_id = run.id

    queue = FakeQueue()
    events = EventService(queue, publish=False)
    scheduler = DAGScheduler(queue=queue, event_service=events)
    await scheduler.tick()
    node = await _ready_node(factory, run_id)

    async with factory() as session:
        stored = await session.get(NodeRun, node.id)
        stored.status = NodeStatus.RUNNING.value
        stored.lease_owner = "dead-worker"
        stored.lease_expires_at = datetime.now(UTC) - timedelta(seconds=10)
        await session.commit()

    await scheduler.tick()
    recovered = await _ready_node(factory, run_id)
    assert recovered.id == node.id
    assert recovered.lease_owner is None
