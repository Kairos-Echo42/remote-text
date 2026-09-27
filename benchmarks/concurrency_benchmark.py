from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass

from sqlalchemy import select

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

NODE_DELAY_SECONDS = 0.2
MAX_CONCURRENCY = 4


@dataclass(slots=True)
class BenchmarkExecutor:
    delay: float
    calls: list[str]

    async def execute(self, *, run, node_run, node_spec, document):
        self.calls.append(node_run.node_id)
        await asyncio.sleep(self.delay)
        return NodeExecutionResult(output={"node": node_run.node_id})

    async def cancel(self, node_run_id):
        return None


async def main() -> None:
    await create_all()
    await bootstrap_default_admin()
    factory = get_session_factory()
    async with factory() as session:
        user = await session.scalar(select(User).limit(1))
        workspace = await session.scalar(select(Workspace).limit(1))
        if user is None or workspace is None:
            raise RuntimeError("run `agentforge bootstrap` first")
        builder = WorkflowBuilder(f"benchmark-{uuid.uuid4().hex[:8]}", "1.0.0").max_concurrency(MAX_CONCURRENCY)
        for index in range(8):
            builder.node(
                f"worker-{index}",
                "capability",
                "fixture.echo",
                inputs={"value": index},
            )
        document = builder.output({}).build()
        _, version = await sync_workflow(session, workspace.id, document)
        run = await create_run(
            session,
            workspace_id=workspace.id,
            workflow_version_id=version.id,
            input_data={},
            created_by=user.id,
        )
        await session.commit()
        run_id = run.id

    queue = RedisStreamQueue()
    await queue.client.flushdb()
    events = EventService(queue, publish=False)
    scheduler = DAGScheduler(queue=queue, event_service=events)
    executor = BenchmarkExecutor(NODE_DELAY_SECONDS, [])
    workers = [
        Worker(executor, queue=queue, event_service=events, consumer_name=f"benchmark-{index}")
        for index in range(MAX_CONCURRENCY)
    ]
    tasks = [asyncio.create_task(scheduler.run_forever(interval_seconds=0.02))]
    tasks.extend(asyncio.create_task(worker.run_forever()) for worker in workers)
    started = time.perf_counter()
    try:
        while time.perf_counter() - started < 30:
            async with factory() as session:
                current = await session.get(Run, run_id)
                if current and current.status in {"succeeded", "failed", "cancelled"}:
                    break
            await asyncio.sleep(0.05)
        workflow_wall_seconds = time.perf_counter() - started
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async with factory() as session:
        nodes = list(
            (await session.scalars(select(NodeRun).where(NodeRun.run_id == run_id).order_by(NodeRun.created_at))).all()
        )
    started_times = [node.started_at for node in nodes if node.started_at is not None]
    finished_times = [node.finished_at for node in nodes if node.finished_at is not None]
    node_span_seconds = (max(finished_times) - min(started_times)).total_seconds()
    serial_estimate = NODE_DELAY_SECONDS * 8
    ratio = node_span_seconds / serial_estimate
    result = {
        "run_id": str(run_id),
        "nodes": 8,
        "max_concurrency": MAX_CONCURRENCY,
        "executor_calls": len(executor.calls),
        "unique_executor_calls": len(set(executor.calls)),
        "node_span_seconds": round(node_span_seconds, 4),
        "workflow_wall_seconds": round(workflow_wall_seconds, 4),
        "serial_estimate_seconds": serial_estimate,
        "parallel_ratio": round(ratio, 4),
        "passed": ratio < 0.6 and len(executor.calls) == len(set(executor.calls)) == 8,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
