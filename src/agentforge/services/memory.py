from __future__ import annotations

import math
import re
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentforge.models import MemoryRecord, Run
from agentforge.providers.base import EmbeddingProvider


class MemoryService:
    def __init__(self, embedding_provider: EmbeddingProvider | None):
        self.embedding_provider = embedding_provider

    async def create(
        self,
        session: AsyncSession,
        *,
        workspace_id: uuid.UUID,
        content: str,
        agent_name: str | None = None,
        scope: str = "workspace",
        kind: str = "fact",
        confidence: float = 0.8,
        source_run_id: uuid.UUID | None = None,
    ) -> MemoryRecord:
        embedding = (await self.embedding_provider.embed([content]))[0] if self.embedding_provider else None
        record = MemoryRecord(
            workspace_id=workspace_id,
            agent_name=agent_name,
            scope=scope,
            kind=kind,
            content=content.strip(),
            confidence=confidence,
            source_run_id=source_run_id,
            embedding=embedding,
        )
        session.add(record)
        await session.flush()
        return record

    async def search(
        self,
        session: AsyncSession,
        *,
        workspace_id: uuid.UUID,
        query: str,
        agent_name: str | None = None,
        limit: int = 8,
    ) -> list[MemoryRecord]:
        records = list(
            (
                await session.scalars(
                    select(MemoryRecord)
                    .where(
                        MemoryRecord.workspace_id == workspace_id,
                        MemoryRecord.deleted_at.is_(None),
                        (MemoryRecord.agent_name == agent_name) | (MemoryRecord.agent_name.is_(None)),
                    )
                    .limit(300)
                )
            ).all()
        )
        query_terms = set(re.findall(r"\w+", query.lower()))
        query_embedding = (await self.embedding_provider.embed([query]))[0] if self.embedding_provider else None
        scored: list[tuple[float, MemoryRecord]] = []
        for record in records:
            lexical = len(query_terms & set(re.findall(r"\w+", record.content.lower())))
            semantic = _cosine(query_embedding, record.embedding or []) if query_embedding is not None else 0.0
            score = lexical + semantic + record.confidence * 0.2
            scored.append((score, record))
        return [record for _, record in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]

    async def extract_from_run(self, session: AsyncSession, run: Run, *, limit: int = 5) -> list[MemoryRecord]:
        output = run.output or {}
        text = str(output.get("report") or output.get("content") or output)
        candidates = [
            sentence.strip() for sentence in re.split(r"(?<=[。！？.!?])\s+|\n+", text) if len(sentence.strip()) >= 24
        ]
        records: list[MemoryRecord] = []
        for candidate in candidates[:limit]:
            if await self._duplicate(session, run.workspace_id, candidate):
                continue
            records.append(
                await self.create(
                    session,
                    workspace_id=run.workspace_id,
                    content=candidate,
                    scope="workspace",
                    kind="summary",
                    confidence=0.7,
                    source_run_id=run.id,
                )
            )
        return records

    async def _duplicate(self, session: AsyncSession, workspace_id: uuid.UUID, content: str) -> bool:
        normalized = content.strip().lower()
        records = list(
            (
                await session.scalars(
                    select(MemoryRecord)
                    .where(
                        MemoryRecord.workspace_id == workspace_id,
                        MemoryRecord.deleted_at.is_(None),
                    )
                    .limit(300)
                )
            ).all()
        )
        return any(record.content.strip().lower() == normalized for record in records)


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    ln = math.sqrt(sum(a * a for a in left))
    rn = math.sqrt(sum(b * b for b in right))
    return dot / (ln * rn) if ln and rn else 0.0
