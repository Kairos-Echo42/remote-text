from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from agentforge.providers.base import Message, ModelResponse, ToolCall, ToolDefinition


class FakeModelProvider:
    """Deterministic model that exercises the same agent loop as a real provider."""

    def __init__(self, *, dimension: int = 1024):
        self.dimension = dimension

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | None = None,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        del tool_choice, response_schema, temperature, max_tokens
        last = messages[-1] if messages else Message(role="user", content="")
        prompt = "\n".join(part for message in messages for part in [message.content or "", message.name or ""])
        tool_calls = [tool for message in messages for tool in message.tool_calls]
        if tools and not tool_calls and "tool_hint" in prompt:
            hinted = re.search(r'tool_hint["\']?\s*:\s*["\']?([a-zA-Z0-9_.:-]+)', prompt)
            tool_name = hinted.group(1) if hinted else tools[0].name
            selected = next(
                (tool for tool in tools if tool.name == tool_name or tool.name.replace("__", ".") == tool_name),
                tools[0],
            )
            arguments = _arguments_for_tool(selected, last.content or "", prompt)
            return ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=f"call-{hashlib.sha1(f'{prompt}{selected.name}'.encode()).hexdigest()[:12]}",
                        name=selected.name,
                        arguments=arguments,
                    )
                ],
                usage={"prompt_tokens": len(prompt.split()), "completion_tokens": 8},
                finish_reason="tool_calls",
            )

        system_prompt = next((message.content or "" for message in messages if message.role == "system"), "").lower()
        if "writer" in system_prompt:
            content = (
                "# AgentForge 研究报告\n\n"
                "## 结论\n\n"
                "多 Agent 并行研究能够缩短任务完成时间，并能通过统一事件模型保留可审计过程。\n\n"
                "## 依据\n\n"
                "- DAG 控制面提供可恢复的节点级状态。\n"
                "- Capability Runtime 将模型、MCP、检索与沙箱统一为工具边界。\n"
                "- Memory 与 RAG 分别承载跨运行知识和事实来源。\n"
            )
        elif "analyst" in system_prompt:
            content = json.dumps(
                {
                    "summary": "已完成证据分析",
                    "findings": ["核心主题具有跨来源一致性", "需要明确区分事实与推断"],
                    "citations": _extract_citations(prompt),
                },
                ensure_ascii=False,
            )
        elif "planner" in system_prompt:
            content = json.dumps(
                {
                    "summary": "已完成任务规划",
                    "steps": ["检索材料", "并行分析", "汇总证据", "生成报告"],
                },
                ensure_ascii=False,
            )
        else:
            content = json.dumps(
                {
                    "summary": "fake provider completed the task",
                    "query": _last_question(prompt),
                    "evidence": _extract_citations(prompt),
                },
                ensure_ascii=False,
            )
        return ModelResponse(
            content=content,
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(content.split())},
        )


class FakeEmbeddingProvider:
    def __init__(self, *, dimension: int = 1024):
        self.dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embedding(text, self.dimension) for text in texts]


class FixtureSearchProvider:
    def __init__(self, documents: list[dict[str, Any]] | None = None):
        self.documents = documents or [
            {
                "title": "AgentForge architecture",
                "url": "https://example.invalid/agentforge",
                "snippet": "DAG orchestration, capabilities, memory, MCP and sandbox execution.",
            },
            {
                "title": "Durable async workflows",
                "url": "https://example.invalid/durable-workflows",
                "snippet": "Leases, idempotency keys and checkpoints enable reliable recovery.",
            },
        ]

    async def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        terms = {term.lower() for term in re.findall(r"\w+", query)}
        ranked = sorted(
            self.documents,
            key=lambda item: len(terms & set(re.findall(r"\w+", f"{item['title']} {item['snippet']}".lower()))),
            reverse=True,
        )
        return ranked[:limit]


def _hash_embedding(text: str, dimension: int) -> list[float]:
    seed = hashlib.sha512(text.encode("utf-8")).digest()
    values: list[float] = []
    counter = 0
    while len(values) < dimension:
        block = hashlib.sha512(seed + counter.to_bytes(4, "big")).digest()
        for index in range(0, len(block), 4):
            number = int.from_bytes(block[index : index + 4], "big")
            values.append((number / 2**31) - 1.0)
            if len(values) == dimension:
                break
        counter += 1
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def _arguments_for_tool(tool: ToolDefinition, content: str, prompt: str) -> dict[str, Any]:
    parsed = _extract_json_object(content) or {}
    node_input = parsed.get("node_input", parsed) if isinstance(parsed, dict) else {}
    schema = tool.input_schema or {}
    required = schema.get("required", [])
    if not required:
        return node_input if isinstance(node_input, dict) else {"value": node_input}
    arguments: dict[str, Any] = {}
    for key in required:
        value = node_input.get(key) if isinstance(node_input, dict) else None
        if value is None and key == "query":
            value = _last_question(prompt)
        if value is None:
            value = _last_question(prompt)
        arguments[key] = value
    return arguments


def _extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _last_question(prompt: str) -> str:
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    return lines[-1][:500] if lines else "task"


def _extract_citations(prompt: str) -> list[str]:
    return [match for match in re.findall(r"\[(?:source|citation):([^\]]+)\]", prompt)][:10]
