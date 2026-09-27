from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from agentforge.capabilities import CapabilityContext, CapabilityRegistry
from agentforge.db import get_session_factory
from agentforge.models import MCPServer, Workspace
from agentforge.services.mcp import MCPManager


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_stdio_mcp_tool_is_discovered_and_invoked():
    fixture = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
    factory = get_session_factory()
    async with factory() as session:
        workspace = await session.scalar(select(Workspace).limit(1))
        server = MCPServer(
            workspace_id=workspace.id,
            name="echo",
            transport="stdio",
            command=[sys.executable, str(fixture)],
            enabled=True,
        )
        session.add(server)
        await session.commit()
        server_id = server.id

    registry = CapabilityRegistry()
    await MCPManager(timeout_seconds=10).register_server(registry, server_id)
    capability = registry.get("mcp.echo.echo")
    result = await capability.invoke(
        CapabilityContext(
            workspace_id=workspace.id,
            run_id=uuid.uuid4(),
            node_run_id=uuid.uuid4(),
        ),
        {"text": "hello"},
    )
    assert "echo:hello" in str(result.output)
