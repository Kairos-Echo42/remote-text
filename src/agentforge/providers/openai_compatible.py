from __future__ import annotations

import json
import re
from typing import Any

from openai import AsyncOpenAI

from agentforge.config import get_settings
from agentforge.providers.base import Message, ModelResponse, ToolCall, ToolDefinition


class OpenAICompatibleModelProvider:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_params: dict[str, Any] | None = None,
    ):
        settings = get_settings()
        self.model = model
        self.default_params = default_params or {}
        self.client = AsyncOpenAI(
            api_key=api_key or settings.openai_api_key or "not-configured",
            base_url=base_url or settings.openai_base_url,
        )

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
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_openai(message) for message in messages],
            "temperature": temperature,
            **self.default_params,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": re.sub(r"[^a-zA-Z0-9_-]", "__", tool.name),
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in tools
            ]
            payload["tool_choice"] = tool_choice or "auto"
            payload["parallel_tool_calls"] = tool_choice not in {"none", "required"}
        if response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "agent_output", "strict": False, "schema": response_schema},
            }
        tool_names = [item["function"]["name"] for item in payload.get("tools", [])]
        try:
            response = await self.client.chat.completions.create(**payload)
        except Exception as exc:
            raise RuntimeError(f"{exc}; outgoing tools={tool_names!r}") from exc
        choice = response.choices[0]
        calls = []
        for call in choice.message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"raw": call.function.arguments}
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))
        usage = response.usage
        return ModelResponse(
            content=choice.message.content,
            tool_calls=calls,
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            finish_reason=choice.finish_reason or "stop",
        )


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        settings = get_settings()
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key or settings.openai_api_key or "not-configured",
            base_url=base_url or settings.openai_base_url,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


def _message_to_openai(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name:
        payload["name"] = message.name
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return payload
