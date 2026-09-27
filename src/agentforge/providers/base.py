from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class Message:
    role: str
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ModelResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"


class ModelProvider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | None = None,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ModelResponse: ...


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class SearchProvider(Protocol):
    async def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]: ...
