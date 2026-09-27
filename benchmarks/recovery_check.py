from __future__ import annotations

import asyncio
import json
import time
import uuid

from sqlalchemy import select

from agentforge.config import get_settings
from agentforge.db import create_all, get_session_factory
from agentforge.models import NodeRun, Run, User, Workspace
from agentforge.runtime.events import EventService
from agentforge.runtime.queue import RedisStreamQueue
from agentforge.runtime.scheduler import DAGScheduler
from agentforge.runtime.worker import NodeExecutionResult, Worker
from agentforge.services.auth import bootstrap_default_admin
from agentforge.services.runs import create_run
from agentforge.services.workflows import sync_workflow
from agentforge.workflow import WorkflowBuilder


class StubExecutor:
    def __init__(self):
        self.calls: list[str] = []

    async def execute(self, *, run, node_run, node_spec, document):
        self.calls.append(node_run.node_id)
        return NodeExecutionResult(output={"recovered": True})

    async def cancel(self, node_run_id):
        return None


async def main() -> None:
    settings = get_settings()
    await create_all()
    await bootstrap_default_admin()
    factory = get_session_factory()
    async with factory() as session:
        user = await session.scalar(select(User).limit(1))
        workspace = await session.scalar(select(Workspace).limit(1))
        document = (
            WorkflowBuilder(f"recovery-{uuid.uuid4().hex[:8]}", "1.0.0")
            .node("work", "capability", "fixture.echo", inputs={"value": "$.input.value"})
            .output({"recovered": "$.nodes.work.output.recovered"})
            .build()
        )
        _, version = await sync_workflow(session, workspace.id, document)
        run = await create_run(
            session,
            workspace_id=workspace.id,
            workflow_version_id=version.id,
            input_data={"value": 1},
            created_by=user.id,
        )
        await session.commit()
        run_id = run.id

    queue = RedisStreamQueue()
    await queue.client.flushdb()
    events = EventService(queue, publish=False)
    scheduler = DAGScheduler(queue=queue, event_service=events)
    executor = StubExecutor()

    await scheduler.tick()
    messages = await queue.consume_node("crashed-worker", block_ms=1000)
    if not messages:
        raise RuntimeError("scheduler did not dispatch a node")
    _, fields = messages[0]
    node_run_id = uuid.UUID(fields["node_run_id"])
    crashed_worker = Worker(
        executor,
        queue=queue,
        event_service=events,
        consumer_name="crashed-worker",
    )
    claimed = await crashed_worker._claim(node_run_id)
    if not claimed:
        raise RuntimeError("crashed worker could not claim node")

    await asyncio.sleep(settings.worker_lease_seconds + 0.5)
    await scheduler.tick()

    replacement = Worker(
        executor,
        queue=queue,
        event_service=events,
        consumer_name="replacement-worker",
    )
    stale = await queue.claim_stale("replacement-worker", min_idle_ms=0, count=10)
    if not stale:
        raise RuntimeError("replacement worker did not recover the unacknowledged message")
    await replacement._handle(stale[0][0], stale[0][1])
    await scheduler.tick()

    async with factory() as session:
        run = await session.get(Run, run_id)
        node = await session.get(NodeRun, node_run_id)
    result = {
        "run_id": str(run_id),
        "run_status": run.status if run else None,
        "node_status": node.status if node else None,
        "executor_calls": len(executor.calls),
        "passed": bool(
            run and run.status == "succeeded" and node and node.status == "succeeded" and executor.calls == ["work"]
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    started = time.perf_counter()
    asyncio.run(main())
    print(f"elapsed_seconds={time.perf_counter() - started:.2f}")
