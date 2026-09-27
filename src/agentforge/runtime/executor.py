from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from agentforge.capabilities import (
    CapabilityContext,
    CapabilityRegistry,
    CapabilityResult,
    register_builtin_capabilities,
)
from agentforge.config import get_settings
from agentforge.db import get_session_factory
from agentforge.models import (
    AgentDefinition,
    Artifact,
    KnowledgeBase,
    ModelProfile,
    NodeRun,
    Run,
)
from agentforge.plugins.skills import SkillCatalog
from agentforge.providers.factory import create_model_provider
from agentforge.runtime.agent import LangGraphAgentRuntime
from agentforge.runtime.context import ContextBuilder
from agentforge.runtime.events import EventService
from agentforge.runtime.queue import RedisStreamQueue
from agentforge.runtime.sandbox import SandboxProvider
from agentforge.runtime.worker import NodeExecutionResult
from agentforge.services.mcp import MCPManager
from agentforge.services.memory import MemoryService
from agentforge.services.models import resolve_embedding_provider
from agentforge.services.retrieval import RetrievalService
from agentforge.services.search import build_search_provider
from agentforge.storage.artifacts import LocalArtifactStore
from agentforge.workflow import WorkflowDocument
from agentforge.workflow.conditions import build_evaluation_context, resolve_mapping
from agentforge.workflow.dsl import NodeSpec


class DefaultNodeExecutor:
    def __init__(
        self,
        *,
        sandbox_provider: SandboxProvider | None = None,
        queue: RedisStreamQueue | None = None,
    ):
        self.sandbox_provider = sandbox_provider or SandboxProvider()
        self.queue = queue or RedisStreamQueue()
        self.events = EventService(self.queue)
        self.artifact_store = LocalArtifactStore()
        self.mcp = MCPManager()
        self.skills = SkillCatalog.load(get_settings().example_dir / "skills")
        self._active_sandboxes: dict[uuid.UUID, Any] = {}

    async def execute(
        self,
        *,
        run: Run,
        node_run: NodeRun,
        node_spec: NodeSpec,
        document: WorkflowDocument,
    ) -> NodeExecutionResult:
        upstream = await self._upstream_outputs(run, document)
        evaluation_context = build_evaluation_context(
            run_input=run.input,
            node_outputs=upstream,
            run_metadata=run.run_metadata,
        )
        node_input = resolve_mapping(node_spec.with_, evaluation_context)
        sandbox = await self.sandbox_provider.create(run_id=run.id, node_run_id=node_run.id, attempt=node_run.attempt)
        self._active_sandboxes[node_run.id] = sandbox
        try:
            registry = CapabilityRegistry()
            register_builtin_capabilities(registry)
            await self.mcp.register_workspace_capabilities(registry, run.workspace_id)
            self.skills.register_code_skills(registry)
            embedding_provider = await self._embedding_provider(run.workspace_id)
            memory = MemoryService(embedding_provider)
            retrieval = RetrievalService(embedding_provider)
            capability_context = CapabilityContext(
                workspace_id=run.workspace_id,
                run_id=run.id,
                node_run_id=node_run.id,
                actor_id=run.created_by,
                artifact_store=self.artifact_store,
                idempotency_key=f"{run.id}:{node_run.node_id}:{node_run.attempt}",
                services={
                    "sandbox": sandbox,
                    "search": build_search_provider(),
                    "memory": memory,
                    "retrieval": retrieval,
                    "knowledge_base_ids": await self._knowledge_base_ids(run.workspace_id, node_input),
                    "events": self.events,
                },
            )
            if node_spec.type == "capability":
                result = await registry.invoke(node_spec.uses, capability_context, node_input)
            else:
                result = await self._execute_agent(
                    run=run,
                    node_input=node_input,
                    node_spec=node_spec,
                    upstream=upstream,
                    registry=registry,
                    capability_context=capability_context,
                    embedding_provider=embedding_provider,
                    memory=memory,
                    retrieval=retrieval,
                )
            persisted = await self._persist_artifacts(run, node_run, result.artifacts)
            output = dict(result.output)
            if persisted:
                output["artifacts"] = persisted
            return NodeExecutionResult(output=output, metrics=result.metrics)
        finally:
            self._active_sandboxes.pop(node_run.id, None)
            await sandbox.close()

    async def cancel(self, node_run_id: uuid.UUID) -> None:
        sandbox = self._active_sandboxes.pop(node_run_id, None)
        if sandbox is not None:
            await sandbox.close()

    async def _execute_agent(
        self,
        *,
        run: Run,
        node_input: dict[str, Any],
        node_spec: NodeSpec,
        upstream: dict[str, dict[str, Any]],
        registry: CapabilityRegistry,
        capability_context: CapabilityContext,
        embedding_provider,
        memory: MemoryService,
        retrieval: RetrievalService,
    ) -> CapabilityResult:
        factory = get_session_factory()
        async with factory() as session:
            agent = await session.scalar(
                select(AgentDefinition).where(
                    AgentDefinition.workspace_id == run.workspace_id,
                    AgentDefinition.name == node_spec.uses,
                )
            )
            if agent is None:
                raise ValueError(f"agent {node_spec.uses!r} is not registered")
            profile = (
                await session.scalar(
                    select(ModelProfile)
                    .options(selectinload(ModelProfile.credential))
                    .where(ModelProfile.id == agent.model_profile_id)
                )
                if agent.model_profile_id
                else None
            )
        direct_result: CapabilityResult | None = None
        direct_tool = node_input.get("tool_hint")
        if isinstance(direct_tool, str) and direct_tool not in {"", "none"}:
            allowed_capabilities = node_spec.capabilities if node_spec.capabilities is not None else agent.capabilities
            if direct_tool not in allowed_capabilities:
                raise ValueError(f"tool_hint {direct_tool!r} is not allowed for agent {agent.name!r}")
            await capability_context.service("events").emit(
                capability_context.run_id,
                "tool.started",
                {"tool": direct_tool, "arguments": node_input},
                node_run_id=capability_context.node_run_id,
            )
            direct_result = await registry.invoke(direct_tool, capability_context, node_input)
            await capability_context.service("events").emit(
                capability_context.run_id,
                "tool.succeeded",
                {
                    "tool": direct_tool,
                    "result": {
                        "output": direct_result.output,
                        "metrics": direct_result.metrics,
                        "artifacts": direct_result.artifacts,
                    },
                },
                node_run_id=capability_context.node_run_id,
            )
            node_input = {
                **node_input,
                "tool_result": {
                    "capability": direct_tool,
                    "output": direct_result.output,
                    "metrics": direct_result.metrics,
                },
            }

        provider = create_model_provider(profile)
        context_builder = ContextBuilder(
            embedding_provider=embedding_provider,
            memory_service=memory,
            retrieval_service=retrieval,
            skill_catalog=self.skills,
        )
        built = await context_builder.build(
            workspace_id=run.workspace_id,
            agent=agent,
            node_input=node_input,
            upstream_outputs=upstream,
            knowledge_base_ids=capability_context.services["knowledge_base_ids"],
            max_context_tokens=profile.max_context_tokens if profile else 32_000,
            response_schema=node_input.get("response_schema"),
        )
        allowed = node_spec.capabilities if node_spec.capabilities is not None else agent.capabilities
        filtered = CapabilityRegistry()
        for name in allowed:
            try:
                filtered.register(registry.get(name))
            except Exception:
                continue
        if direct_result is not None:
            direct_evidence = _direct_tool_evidence(direct_tool or "", direct_result.output)
            if direct_evidence:
                built.evidence.extend(direct_evidence)
                built.messages[0].content = (
                    (built.messages[0].content or "")
                    + "\n\nDirect tool evidence:\n"
                    + "\n".join(
                        f"{item['citation']} {item.get('snippet') or item.get('content', '')}"
                        for item in direct_evidence
                    )
                )

        runtime = LangGraphAgentRuntime(
            provider=provider,
            registry=filtered,
            capability_context=capability_context,
            max_steps=int((agent.limits or {}).get("max_steps", 12)),
            max_tool_steps=int((agent.limits or {}).get("max_tool_steps", 1)),
        )
        result = await runtime.run(built.messages)
        if built.evidence:
            result.output["citations"] = built.evidence
        else:
            result.output.setdefault("citations", [])
        result.output.setdefault("memory_used", built.memories)
        return CapabilityResult(
            output=result.output,
            metrics={
                "usage": result.usage,
                "steps": result.steps,
                "context_tokens": built.token_estimate,
                "direct_tool": direct_tool if direct_result is not None else None,
            },
            artifacts=direct_result.artifacts if direct_result is not None else [],
        )

    async def _upstream_outputs(self, run: Run, document: WorkflowDocument) -> dict[str, dict[str, Any]]:
        factory = get_session_factory()
        async with factory() as session:
            nodes = list(
                (
                    await session.scalars(
                        select(NodeRun).where(NodeRun.run_id == run.id).order_by(NodeRun.node_id, NodeRun.attempt)
                    )
                ).all()
            )
        latest: dict[str, NodeRun] = {}
        for node in nodes:
            current = latest.get(node.node_id)
            if current is None or node.attempt > current.attempt:
                latest[node.node_id] = node
        return {
            node_id: node.output or {}
            for node_id, node in latest.items()
            if node_id in {spec.id for spec in document.spec.nodes} and node.output is not None
        }

    async def _embedding_provider(self, workspace_id: uuid.UUID):
        factory = get_session_factory()
        async with factory() as session:
            return await resolve_embedding_provider(session, workspace_id)

    async def _knowledge_base_ids(self, workspace_id: uuid.UUID, node_input: dict[str, Any]) -> list[uuid.UUID]:
        raw = node_input.get("knowledge_base_ids")
        if raw is not None:
            items = [raw] if isinstance(raw, str) else raw
            result = []
            for item in items:
                try:
                    result.append(uuid.UUID(str(item)))
                except ValueError:
                    continue
            return result
        factory = get_session_factory()
        async with factory() as session:
            return list(
                (
                    await session.scalars(select(KnowledgeBase.id).where(KnowledgeBase.workspace_id == workspace_id))
                ).all()
            )

    async def _persist_artifacts(
        self,
        run: Run,
        node_run: NodeRun,
        artifacts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not artifacts:
            return []
        factory = get_session_factory()
        stored_items: list[Artifact] = []
        async with factory() as session:
            async with session.begin():
                for item in artifacts:
                    content = item.get("content", "")
                    if isinstance(content, str):
                        raw = content.encode("utf-8")
                    else:
                        raw = json.dumps(content, ensure_ascii=False).encode("utf-8")
                    stored = await self.artifact_store.put(
                        raw,
                        workspace_id=run.workspace_id,
                        run_id=run.id,
                        filename=item["name"],
                        content_type=item.get("content_type"),
                    )
                    artifact = Artifact(
                        workspace_id=run.workspace_id,
                        run_id=run.id,
                        node_run_id=node_run.id,
                        name=item["name"],
                        content_type=stored.content_type,
                        size=stored.size,
                        checksum=stored.checksum,
                        storage_key=stored.key,
                    )
                    session.add(artifact)
                    stored_items.append(artifact)
                await session.flush()
                return [
                    {
                        "id": str(item.id),
                        "name": item.name,
                        "content_type": item.content_type,
                        "size": item.size,
                    }
                    for item in stored_items
                ]


def _direct_tool_evidence(capability: str, output: dict[str, Any]) -> list[dict[str, Any]]:
    results = output.get("results") if isinstance(output, dict) else None
    if not isinstance(results, list):
        return []
    evidence: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        citation = item.get("citation")
        source = item.get("url") or item.get("filename") or item.get("title") or capability
        if not citation:
            citation = f"[source:{source}]"
        evidence.append({**item, "citation": citation, "capability": capability})
    return evidence
