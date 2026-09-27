from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from agentforge.db import get_session_factory
from agentforge.models import AgentDefinition
from agentforge.plugins.skills import SkillCatalog
from agentforge.providers.base import EmbeddingProvider, Message
from agentforge.services.memory import MemoryService
from agentforge.services.retrieval import RetrievalService


@dataclass(slots=True)
class BuiltContext:
    messages: list[Message]
    evidence: list[dict[str, Any]] = field(default_factory=list)
    memories: list[dict[str, Any]] = field(default_factory=list)
    token_estimate: int = 0


class ContextBuilder:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider | None,
        memory_service: MemoryService | None = None,
        retrieval_service: RetrievalService | None = None,
        skill_catalog: SkillCatalog | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.memory = memory_service or MemoryService(embedding_provider)
        self.retrieval = retrieval_service or RetrievalService(embedding_provider)
        self.skills = skill_catalog or SkillCatalog()

    async def build(
        self,
        *,
        workspace_id: uuid.UUID,
        agent: AgentDefinition,
        node_input: dict[str, Any],
        upstream_outputs: dict[str, dict[str, Any]],
        knowledge_base_ids: list[uuid.UUID],
        max_context_tokens: int = 32_000,
        response_schema: dict[str, Any] | None = None,
    ) -> BuiltContext:
        query = _query_from_input(node_input)
        evidence = []
        memories = []
        if knowledge_base_ids and query:
            evidence = [
                item.as_dict()
                for item in await self.retrieval.search(query=query, knowledge_base_ids=knowledge_base_ids, limit=8)
            ]
        if query:
            factory = get_session_factory()
            async with factory() as session:
                memory_records = await self.memory.search(
                    session,
                    workspace_id=workspace_id,
                    query=query,
                    agent_name=agent.name,
                    limit=8,
                )
            memories = [
                {
                    "id": str(record.id),
                    "content": record.content,
                    "kind": record.kind,
                    "confidence": record.confidence,
                }
                for record in memory_records
            ]

        skill_names = list((agent.spec.get("spec", {}).get("context", {}) or {}).get("skills", []))
        skill_instructions = self.skills.instructions_for(skill_names)
        system_parts = [
            "You are an AgentForge worker. Use only the provided capabilities.",
            agent.instructions,
            "Return a concise final answer. If a JSON output schema is provided, obey it.",
        ]
        if skill_instructions:
            system_parts.append(
                "\n\n".join(f"Skill: {name}\n{instructions}" for name, instructions in skill_instructions)
            )
        if response_schema:
            system_parts.append("Output schema:\n" + json.dumps(response_schema, ensure_ascii=False))
        if evidence:
            system_parts.append(
                "Evidence (cite its source marker when used):\n"
                + "\n".join(f"{item['citation']} {item['content']}" for item in evidence)
            )
        if memories:
            system_parts.append("Relevant durable memory:\n" + "\n".join(f"- {item['content']}" for item in memories))

        user_payload = {
            "node_input": node_input,
            "upstream_outputs": upstream_outputs,
        }
        messages = [
            Message(role="system", content="\n\n".join(part for part in system_parts if part)),
            Message(role="user", content=json.dumps(user_payload, ensure_ascii=False, indent=2)),
        ]
        messages = _fit_messages(messages, max_context_tokens)
        token_estimate = sum(max(1, len(message.content or "") // 4) for message in messages)
        return BuiltContext(
            messages=messages,
            evidence=evidence,
            memories=memories,
            token_estimate=token_estimate,
        )


def _query_from_input(node_input: dict[str, Any]) -> str:
    for key in ("query", "topic", "question", "prompt", "task"):
        value = node_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(node_input, ensure_ascii=False)[:1000]


def _fit_messages(messages: list[Message], max_tokens: int) -> list[Message]:
    budget = max_tokens * 4
    system = messages[0]
    user = messages[1]
    available = max(2000, budget - len(system.content or ""))
    if len(user.content or "") > available:
        user.content = (user.content or "")[:available] + "\n[context truncated]"
    return [system, user]
