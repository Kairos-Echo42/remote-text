from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console

from agentforge.config import get_settings
from agentforge.db import create_all, get_session_factory
from agentforge.models import Run, Workspace
from agentforge.runtime.executor import DefaultNodeExecutor
from agentforge.runtime.sandbox import SandboxProvider
from agentforge.runtime.scheduler import DAGScheduler
from agentforge.runtime.state import RUN_TERMINAL_STATES
from agentforge.runtime.worker import Worker
from agentforge.services.auth import bootstrap_default_admin
from agentforge.services.runs import create_run
from agentforge.services.skills import sync_skill_catalog
from agentforge.services.workflows import sync_agent, sync_workflow_path
from agentforge.workflow import load_workflow, validate_workflow

app = typer.Typer(help="AgentForge command line interface", no_args_is_help=True)
console = Console()


@app.command()
def validate(path: Path) -> None:
    """Validate a workflow YAML file."""
    document = load_workflow(path)
    issues = validate_workflow(document)
    if issues:
        for issue in issues:
            console.print(f"[red]{issue.code}[/red] {issue.path}: {issue.message}")
        raise typer.Exit(1)
    console.print(f"[green]valid[/green] {document.metadata.name}@{document.metadata.version}")


@app.command()
def bootstrap() -> None:
    """Create the database tables and default administrator."""
    asyncio.run(_bootstrap())


@app.command()
def sync(
    directory: Path = typer.Option(None, help="Directory containing agents/ and workflows/"),
) -> None:
    """Sync code-defined Agents and Workflows into the database."""
    asyncio.run(_sync(directory))


@app.command()
def run(
    workflow: str,
    input_json: str = typer.Option("{}", "--input", "-i"),
    wait: bool = typer.Option(False, "--wait"),
    timeout: int = typer.Option(300, min=1),
) -> None:
    """Create a workflow run, optionally executing it with an in-process worker."""
    asyncio.run(_run(workflow, input_json, wait=wait, timeout_seconds=timeout))


@app.command()
def worker() -> None:
    """Start a durable Agent execution worker."""

    async def main() -> None:
        executor = DefaultNodeExecutor()
        await Worker(executor).run_forever()

    asyncio.run(main())


@app.command()
def scheduler() -> None:
    """Start the DAG scheduler and outbox dispatcher."""
    asyncio.run(DAGScheduler().run_forever())


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Start the FastAPI gateway."""
    import uvicorn

    uvicorn.run(
        "agentforge.gateway.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def demo(
    topic: str = typer.Option("如何构建可靠的异步多 Agent 平台？", "--topic"),
    timeout: int = typer.Option(300, min=10),
) -> None:
    """Run the bundled research workflow with local services and a deterministic model."""
    asyncio.run(_demo(topic, timeout))


async def _bootstrap() -> None:
    settings = get_settings()
    if settings.database_url.startswith("sqlite"):
        await create_all()
    await bootstrap_default_admin()
    console.print("[green]AgentForge database is ready[/green]")


async def _sync(directory: Path | None) -> None:
    settings = get_settings()
    await create_all()
    await bootstrap_default_admin()
    base = directory or settings.example_dir
    factory = get_session_factory()
    async with factory() as session:
        workspace = await _first_workspace(session)
        agent_count = 0
        workflow_count = 0
        skill_count = 0
        for path in sorted((base / "agents").glob("*.yaml")):
            import yaml

            await sync_agent(session, workspace.id, yaml.safe_load(path.read_text(encoding="utf-8")))
            agent_count += 1
        for path in sorted((base / "workflows").glob("*.yaml")):
            await sync_workflow_path(session, workspace.id, path)
            workflow_count += 1
        skill_count = await sync_skill_catalog(session, workspace_id=workspace.id, root=base / "skills")
        await session.commit()
    console.print(f"[green]synced[/green] {agent_count} agents, {workflow_count} workflows, {skill_count} skills")


async def _run(workflow: str, input_json: str, *, wait: bool, timeout_seconds: int) -> None:
    await create_all()
    await bootstrap_default_admin()
    try:
        payload = json.loads(input_json)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"input is not valid JSON: {exc}") from exc
    factory = get_session_factory()
    async with factory() as session:
        workspace = await _first_workspace(session)
        from sqlalchemy import select

        from agentforge.models import WorkflowDefinition, WorkflowVersion

        version = await session.scalar(
            select(WorkflowVersion)
            .join(WorkflowDefinition, WorkflowVersion.workflow_id == WorkflowDefinition.id)
            .where(
                WorkflowDefinition.workspace_id == workspace.id,
                WorkflowDefinition.name == workflow,
            )
            .order_by(WorkflowVersion.created_at.desc())
            .limit(1)
        )
        if version is None:
            raise typer.BadParameter(f"workflow {workflow!r} has not been synced")
        run = await create_run(
            session,
            workspace_id=workspace.id,
            workflow_version_id=version.id,
            input_data=payload,
        )
        await session.commit()
        run_id = run.id
    console.print(f"run: [cyan]{run_id}[/cyan]")
    if not wait:
        return

    settings = get_settings()
    original_redis_url = settings.redis_url
    if settings.redis_url.rstrip("/").endswith("/0"):
        settings.redis_url = settings.redis_url.rstrip("/")[:-1] + "1"

    executor = DefaultNodeExecutor(sandbox_provider=SandboxProvider(provider="local"))
    worker = Worker(executor)
    scheduler = DAGScheduler()
    tasks = [
        asyncio.create_task(worker.run_forever()),
        asyncio.create_task(scheduler.run_forever()),
    ]
    try:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            async with factory() as session:
                current = await session.get(Run, run_id)
                if current and current.status in RUN_TERMINAL_STATES:
                    console.print(f"status: [bold]{current.status}[/bold]")
                    if current.error:
                        console.print(f"[red]{current.error}[/red]")
                    if current.output:
                        console.print_json(json.dumps(current.output, ensure_ascii=False))
                    return
            await asyncio.sleep(0.5)
        console.print("[red]run timed out[/red]")
        raise typer.Exit(2)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        settings.redis_url = original_redis_url


async def _demo(topic: str, timeout_seconds: int) -> None:
    await _sync(None)
    await _run(
        "multi-agent-research",
        json.dumps({"topic": topic}, ensure_ascii=False),
        wait=True,
        timeout_seconds=timeout_seconds,
    )


async def _first_workspace(session) -> Workspace:
    from sqlalchemy import select

    workspace = await session.scalar(select(Workspace).order_by(Workspace.created_at).limit(1))
    if workspace is None:
        raise RuntimeError("no workspace exists; run agentforge bootstrap")
    return workspace
