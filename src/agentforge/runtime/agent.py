from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agentforge.capabilities import CapabilityContext, CapabilityRegistry
from agentforge.providers.base import Message, ModelProvider, ModelResponse, ToolCall


class AgentState(TypedDict, total=False):
    messages: list[Message]
    step: int
    max_steps: int
    tool_calls: list[ToolCall]
    response: ModelResponse | None
    output: dict[str, Any]
    usage: dict[str, int]


@dataclass(slots=True)
class AgentRunResult:
    output: dict[str, Any]
    usage: dict[str, int]
    steps: int


class LangGraphAgentRuntime:
    """A compact plan-act-observe graph around any OpenAI-compatible provider."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        registry: CapabilityRegistry,
        capability_context: CapabilityContext,
        max_steps: int = 12,
        max_tool_steps: int = 1,
        temperature: float = 0.0,
        response_schema: dict[str, Any] | None = None,
    ):
        self.provider = provider
        self.registry = registry
        self.capability_context = capability_context
        self.max_steps = max_steps
        self.max_tool_steps = max_tool_steps
        self.temperature = temperature
        self.response_schema = response_schema

    async def run(self, messages: list[Message]) -> AgentRunResult:
        graph = self._build_graph()
        state: AgentState = {
            "messages": list(messages),
            "step": 0,
            "max_steps": self.max_steps,
            "tool_calls": [],
            "response": None,
            "output": {},
            "usage": {},
        }
        result = await graph.ainvoke(state)
        return AgentRunResult(
            output=result.get("output") or {},
            usage=result.get("usage") or {},
            steps=int(result.get("step", 0)),
        )

    def _build_graph(self):
        builder = StateGraph(AgentState)

        async def reason(state: AgentState) -> AgentState:
            available_tools = self.registry.openai_tools()
            can_call_tools = bool(available_tools) and int(state.get("step", 0)) < self.max_tool_steps
            tool_choice = "auto" if can_call_tools else ("none" if available_tools else None)
            response = await self.provider.complete(
                state["messages"],
                tools=available_tools,
                tool_choice=tool_choice,
                response_schema=self.response_schema,
                temperature=self.temperature,
            )
            selected_tool_calls = response.tool_calls[:1]
            assistant = Message(role="assistant", content=response.content, tool_calls=selected_tool_calls)
            usage = dict(state.get("usage") or {})
            for key, value in response.usage.items():
                usage[key] = usage.get(key, 0) + value
            if response.tool_calls:
                output = dict(state.get("output") or {})
            else:
                output = parse_output(response.content)
            return {
                "messages": [*state["messages"], assistant],
                "response": response,
                "tool_calls": selected_tool_calls,
                "usage": usage,
                "output": output,
            }

        async def act(state: AgentState) -> AgentState:
            messages = list(state["messages"])
            for call in state.get("tool_calls", []):
                await self.capability_context.service("events").emit(
                    self.capability_context.run_id,
                    "tool.started",
                    {"tool": call.name, "arguments": call.arguments},
                    node_run_id=self.capability_context.node_run_id,
                )
                try:
                    capability_name = self.registry.resolve_external_name(call.name)
                    result = await self.registry.invoke(capability_name, self.capability_context, call.arguments)
                    payload = {
                        "output": result.output,
                        "metrics": result.metrics,
                        "artifacts": result.artifacts,
                    }
                    status = "succeeded"
                except Exception as exc:
                    payload = {"error": str(exc)}
                    status = "failed"
                await self.capability_context.service("events").emit(
                    self.capability_context.run_id,
                    f"tool.{status}",
                    {"tool": call.name, "result": payload},
                    node_run_id=self.capability_context.node_run_id,
                )
                messages.append(
                    Message(
                        role="tool",
                        name=call.name,
                        tool_call_id=call.id,
                        content=json.dumps(payload, ensure_ascii=False),
                    )
                )
            return {
                "messages": messages,
                "step": int(state.get("step", 0)) + 1,
                "tool_calls": [],
                "response": None,
            }

        def route(state: AgentState) -> str:
            response = state.get("response")
            if response is not None and response.tool_calls:
                if int(state.get("step", 0)) >= self.max_tool_steps:
                    return "end"
                if int(state.get("step", 0)) >= int(state.get("max_steps", self.max_steps)):
                    return "end"
                return "act"
            return "end"

        builder.add_node("reason", reason)
        builder.add_node("act", act)
        builder.add_edge(START, "reason")
        builder.add_conditional_edges("reason", route, {"act": "act", "end": END})
        builder.add_edge("act", "reason")
        return builder.compile()


def parse_output(content: str | None) -> dict[str, Any]:
    if content is None:
        return {"content": ""}
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
        return {"content": content, "parsed": parsed}
    except json.JSONDecodeError:
        return {"content": content}
