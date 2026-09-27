from __future__ import annotations

import io
import math
import re
import uuid
from dataclasses import dataclass
from typing import Any

from pypdf import PdfReader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentforge.models import Chunk, Document, KnowledgeBase
from agentforge.providers.base import EmbeddingProvider
from agentforge.storage.artifacts import ArtifactStore


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    content: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "filename": self.filename,
            "content": self.content,
            "score": self.score,
            "citation": f"[source:{self.filename}#{str(self.chunk_id)[:8]}]",
        }


def extract_text(filename: str, content: bytes) -> str:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix == "pdf":
        reader = PdfReader(io.BytesIO(content))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {"json", "jsonl"}:
        return content.decode("utf-8", errors="replace")
    return content.decode("utf-8", errors="replace")


def chunk_text(text: str, *, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    normalized = re.sub(r"\r\n?", "\n", text).strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = max(normalized.rfind("\n\n", start, end), normalized.rfind(" ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary
        chunks.append(normalized[start:end].strip())
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


async def ingest_document(
    session: AsyncSession,
    *,
    knowledge_base: KnowledgeBase,
    filename: str,
    content: bytes,
    content_type: str | None,
    artifact_store: ArtifactStore,
    embedding_provider: EmbeddingProvider | None,
) -> Document:
    stored = await artifact_store.put(
        content,
        workspace_id=knowledge_base.workspace_id,
        run_id=None,
        filename=filename,
        content_type=content_type,
    )
    document = Document(
        knowledge_base_id=knowledge_base.id,
        filename=filename,
        content_type=content_type,
        storage_key=stored.key,
        checksum=stored.checksum,
        status="processing",
    )
    session.add(document)
    await session.flush()
    try:
        text = extract_text(filename, content)
        chunks = chunk_text(
            text,
            chunk_size=knowledge_base.chunk_size,
            overlap=knowledge_base.chunk_overlap,
        )
        if not chunks:
            raise ValueError("document did not contain extractable text")
        embeddings = await embedding_provider.embed(chunks) if embedding_provider else [None] * len(chunks)
        for ordinal, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            session.add(
                Chunk(
                    document_id=document.id,
                    ordinal=ordinal,
                    content=chunk,
                    token_count=estimate_tokens(chunk),
                    embedding=embedding,
                    metadata_={"filename": filename},
                )
            )
        document.status = "ready"
    except Exception as exc:
        document.status = "failed"
        document.error = str(exc)
    return document


async def search_knowledge(
    session: AsyncSession,
    *,
    knowledge_base_ids: list[uuid.UUID],
    query: str,
    embedding_provider: EmbeddingProvider | None,
    limit: int = 8,
) -> list[RetrievedChunk]:
    if not knowledge_base_ids:
        return []
    vector_ranked: list[uuid.UUID] = []
    query_embedding = (await embedding_provider.embed([query]))[0] if embedding_provider else None
    if query_embedding is not None:
        dialect = session.bind.dialect.name if session.bind else ""
        if dialect == "postgresql":
            distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
            rows = (
                await session.execute(
                    select(Chunk, Document, distance)
                    .join(Document, Chunk.document_id == Document.id)
                    .where(
                        Document.knowledge_base_id.in_(knowledge_base_ids),
                        Chunk.embedding.is_not(None),
                    )
                    .order_by(distance)
                    .limit(limit * 3)
                )
            ).all()
            vector_ranked = [row[0].id for row in rows]
        else:
            rows = (
                await session.execute(
                    select(Chunk, Document)
                    .join(Document, Chunk.document_id == Document.id)
                    .where(
                        Document.knowledge_base_id.in_(knowledge_base_ids),
                        Chunk.embedding.is_not(None),
                    )
                    .limit(500)
                )
            ).all()
            scored = [(chunk.id, cosine_similarity(query_embedding, chunk.embedding or [])) for chunk, _ in rows]
            vector_ranked = [item[0] for item in sorted(scored, key=lambda item: item[1], reverse=True)[: limit * 3]]

    terms = [term.lower() for term in re.findall(r"\w+", query) if len(term) > 1][:8]
    lexical_rows = (
        await session.execute(
            select(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.knowledge_base_id.in_(knowledge_base_ids))
            .limit(1000)
        )
    ).all()
    lexical_scored: list[tuple[uuid.UUID, float]] = []
    mapping: dict[uuid.UUID, tuple[Chunk, Document]] = {}
    for chunk, document in lexical_rows:
        mapping[chunk.id] = (chunk, document)
        haystack = chunk.content.lower()
        score = sum(haystack.count(term) for term in terms)
        if score:
            lexical_scored.append((chunk.id, float(score)))
    lexical_ranked = [item[0] for item in sorted(lexical_scored, key=lambda item: item[1], reverse=True)]

    ranked_ids = rrf(vector_ranked, lexical_ranked, limit=limit)
    results: list[RetrievedChunk] = []
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        chunk, document = mapping[chunk_id]
        results.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=document.id,
                filename=document.filename,
                content=chunk.content,
                score=round(1.0 / (60 + rank), 6),
            )
        )
    return results


def rrf(*rankings: list[uuid.UUID], limit: int, k: int = 60) -> list[uuid.UUID]:
    scores: dict[uuid.UUID, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return [item for item, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:limit]]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
