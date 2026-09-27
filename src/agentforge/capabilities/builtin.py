from __future__ import annotations

from typing import Any

from agentforge.capabilities.base import (
    CapabilityContext,
    CapabilityRegistry,
    CapabilityResult,
    FunctionCapability,
    SideEffect,
)
from agentforge.db import get_session_factory


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
