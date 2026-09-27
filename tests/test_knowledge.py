from __future__ import annotations

import pytest
from sqlalchemy import select

from agentforge.config import get_settings
from agentforge.db import get_session_factory
from agentforge.models import KnowledgeBase, Workspace
from agentforge.providers.fake import FakeEmbeddingProvider
from agentforge.services.knowledge import ingest_document, search_knowledge
from agentforge.storage.artifacts import LocalArtifactStore


@pytest.mark.asyncio
async def test_knowledge_ingestion_and_hybrid_retrieval():
    factory = get_session_factory()
    async with factory() as session:
        workspace = await session.scalar(select(Workspace).limit(1))
        knowledge_base = KnowledgeBase(
            workspace_id=workspace.id,
            name="runtime-notes",
            chunk_size=200,
            chunk_overlap=20,
        )
        session.add(knowledge_base)
        await session.flush()
        document = await ingest_document(
            session,
            knowledge_base=knowledge_base,
            filename="runtime.md",
            content=(
                b"AgentForge uses durable leases and checkpoints for workflow recovery. "
                b"Capabilities provide a unified tool runtime for MCP, RAG and sandbox execution."
            ),
            content_type="text/markdown",
            artifact_store=LocalArtifactStore(get_settings().artifact_root),
            embedding_provider=FakeEmbeddingProvider(),
        )
        await session.commit()
        assert document.status == "ready"

    async with factory() as session:
        results = await search_knowledge(
            session,
            knowledge_base_ids=[knowledge_base.id],
            query="How does workflow recovery use leases?",
            embedding_provider=FakeEmbeddingProvider(),
            limit=3,
        )
    assert results
    assert results[0].filename == "runtime.md"
    assert "leases" in results[0].content
