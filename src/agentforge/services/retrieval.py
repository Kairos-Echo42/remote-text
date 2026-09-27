from __future__ import annotations

import uuid

from agentforge.db import get_session_factory
from agentforge.providers.base import EmbeddingProvider
from agentforge.services.knowledge import RetrievedChunk, search_knowledge


class RetrievalService:
    def __init__(self, embedding_provider: EmbeddingProvider | None):
        self.embedding_provider = embedding_provider

    async def search(
        self,
        *,
        query: str,
        knowledge_base_ids: list[uuid.UUID],
        limit: int = 8,
    ) -> list[RetrievedChunk]:
        factory = get_session_factory()
        async with factory() as session:
            return await search_knowledge(
                session,
                knowledge_base_ids=knowledge_base_ids,
                query=query,
                embedding_provider=self.embedding_provider,
                limit=limit,
            )
