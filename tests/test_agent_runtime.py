from __future__ import annotations

import uuid

import pytest

from agentforge.capabilities import CapabilityContext, CapabilityRegistry
from agentforge.capabilities.builtin import register_builtin_capabilities
from agentforge.providers.base import Message
from agentforge.providers.fake import FakeModelProvider
from agentforge.runtime.agent import LangGraphAgentRuntime


class NullEventService:
    async def emit(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test_langgraph_agent_executes_tool_then_returns_final_output():
    registry = CapabilityRegistry()
    register_builtin_capabilities(registry)
    context = CapabilityContext(
        workspace_id=uuid.uuid4(),
        run_id=uuid.uuid4(),
        node_run_id=uuid.uuid4(),
        services={"events": NullEventService()},
    )
    runtime = LangGraphAgentRuntime(
        provider=FakeModelProvider(),
        registry=registry,
        capability_context=context,
        max_steps=4,
    )
    result = await runtime.run(
        [
            Message(role="system", content="You are repository-researcher"),
            Message(role="user", content='{"query":"AgentForge","tool_hint":"fixture.echo"}'),
        ]
    )
    assert result.steps == 1
    assert "summary" in result.output
