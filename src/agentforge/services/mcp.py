from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import select

from agentforge.capabilities import (
    CapabilityContext,
    CapabilityRegistry,
    CapabilityResult,
    FunctionCapability,
    SideEffect,
)
from agentforge.db import get_session_factory
from agentforge.models import MCPServer

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:  # pragma: no cover - dependency error is surfaced when MCP servers are used
    ClientSession = None  # type: ignore[assignment,misc]
    StdioServerParameters = None  # type: ignore[assignment,misc]
    stdio_client = None  # type: ignore[assignment,misc]
    streamablehttp_client = None  # type: ignore[assignment,misc]


class MCPManager:
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds

    async def register_workspace_capabilities(self, registry: CapabilityRegistry, workspace_id) -> None:
        factory = get_session_factory()
        async with factory() as session:
            servers = list(
                (
                    await session.scalars(
                        select(MCPServer).where(
                            MCPServer.workspace_id == workspace_id,
                            MCPServer.enabled.is_(True),
                        )
                    )
                ).all()
            )
            server_ids = [server.id for server in servers]
        for server_id in server_ids:
            await self.register_server(registry, server_id)

    async def register_server(self, registry: CapabilityRegistry, server_id) -> None:
        server = await self._load_server(server_id)
        if server is None or ClientSession is None:
            return
        try:
            tools = await self.discover_tools(server)
        except Exception:
            return
        for tool in tools:
            name = f"mcp.{server.name}.{tool.name}"
            registry.register(
                FunctionCapability(
                    name,
                    tool.description or f"MCP tool {tool.name} from {server.name}",
                    self._make_handler(server.id, tool.name, server.name),
                    input_schema=tool.input_schema,
                    side_effect=SideEffect.EXTERNAL,
                    idempotent=False,
                )
            )
        factory = get_session_factory()
        async with factory() as session:
            entity = await session.get(MCPServer, server.id)
            if entity is not None:
                entity.tool_cache = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                    }
                    for tool in tools
                ]
                await session.commit()

    async def discover_tools(self, server: MCPServer):
        async with self._session_for(server) as session:
            response = await session.list_tools()
            result = []
            for tool in response.tools:
                result.append(
                    type(
                        "DiscoveredTool",
                        (),
                        {
                            "name": tool.name,
                            "description": getattr(tool, "description", None),
                            "input_schema": getattr(tool, "inputSchema", None) or {"type": "object"},
                        },
                    )()
                )
            return result

    def _make_handler(self, server_id, tool_name: str, server_name: str):
        async def invoke(context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
            del context
            server = await self._load_server(server_id)
            if server is None:
                raise RuntimeError(f"MCP server {server_name!r} is no longer configured")
            async with self._session_for(server) as session:
                result = await session.call_tool(tool_name, arguments=arguments)
            content = [
                item.model_dump() if hasattr(item, "model_dump") else str(item)
                for item in getattr(result, "content", [])
            ]
            structured = getattr(result, "structuredContent", None)
            return CapabilityResult(output={"content": content, "structured": structured})

        return invoke

    @asynccontextmanager
    async def _session_for(self, server: MCPServer) -> AsyncIterator[Any]:
        if ClientSession is None:
            raise RuntimeError("mcp package is not installed")
        if server.transport == "stdio":
            if not server.command:
                raise RuntimeError(f"MCP stdio server {server.name!r} has no command")
            params = StdioServerParameters(command=server.command[0], args=server.command[1:])
            async with asyncio.timeout(self.timeout_seconds):
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        yield session
        elif server.transport == "streamable_http":
            if not server.endpoint:
                raise RuntimeError(f"MCP HTTP server {server.name!r} has no endpoint")
            async with asyncio.timeout(self.timeout_seconds):
                async with streamablehttp_client(server.endpoint) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        yield session
        else:
            raise RuntimeError(f"unsupported MCP transport {server.transport!r}")

    async def _load_server(self, server_id) -> MCPServer | None:
        factory = get_session_factory()
        async with factory() as session:
            server = await session.get(MCPServer, server_id)
            if server is not None:
                _ = server.env_secret_ids
            return server
