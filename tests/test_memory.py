from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from agentforge.capabilities import CapabilityContext, CapabilityRegistry
from agentforge.capabilities.builtin import register_builtin_capabilities
from agentforge.db import get_session_factory
from agentforge.models import Workspace
from agentforge.providers.fake import FakeEmbeddingProvider
from agentforge.services.memory import MemoryService


@pytest.mark.asyncio
async def test_memory_capabilities_write_and_search():
    factory = get_session_factory()
    async with factory() as session:
        workspace = await session.scalar(select(Workspace).limit(1))
        workspace_id = workspace.id
    memory = MemoryService(FakeEmbeddingProvider())
    context = CapabilityContext(
        workspace_id=workspace_id,
        run_id=uuid.uuid4(),
        node_run_id=uuid.uuid4(),
        services={"memory": memory},
    )
    registry = CapabilityRegistry()
    register_builtin_capabilities(registry)
    await registry.invoke(
        "memory.write",
        context,
        {"content": "AgentForge uses leases for durable node recovery."},
    )
    result = await registry.invoke(
        "memory.search",
        context,
        {"query": "How are nodes recovered?"},
    )
    assert result.output["results"]
    assert "leases" in result.output["results"][0]["content"]
