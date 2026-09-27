from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agentforge.db import get_session_factory
from agentforge.models import OutboxEvent, Run, RunEvent
from agentforge.runtime.queue import RedisStreamQueue
from agentforge.security import redact


class EventService:
    """Appends authoritative run events and best-effort fans them out through Redis."""

    def __init__(self, queue: RedisStreamQueue | None = None, *, publish: bool = True):
        self.queue = queue or RedisStreamQueue()
        self.publish = publish

    async def append(
        self,
        session: AsyncSession,
        run_id: uuid.UUID,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        node_run_id: uuid.UUID | None = None,
    ) -> RunEvent:
        result = await session.execute(
            update(Run)
            .where(Run.id == run_id)
            .values(last_event_seq=Run.last_event_seq + 1)
            .returning(Run.last_event_seq)
        )
        seq = int(result.scalar_one())
        event = RunEvent(
            run_id=run_id,
            node_run_id=node_run_id,
            seq=seq,
            event_type=event_type,
            payload=redact(payload or {}),
        )
        session.add(event)
        await session.flush()
        return event

    async def emit(
        self,
        run_id: uuid.UUID,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        node_run_id: uuid.UUID | None = None,
    ) -> RunEvent:
        factory = get_session_factory()
        async with factory() as session:
            async with session.begin():
                event = await self.append(session, run_id, event_type, payload, node_run_id=node_run_id)
        if self.publish:
            try:
                await self.queue.publish_event(
                    str(run_id),
                    {
                        "id": str(event.id),
                        "seq": event.seq,
                        "type": event.event_type,
                        "payload": json.dumps(event.payload, ensure_ascii=False),
                        "created_at": event.created_at.isoformat(),
                    },
                )
            except Exception:
                # Persisted events remain replayable even if Redis fan-out is unavailable.
                pass
        return event


class OutboxPublisher:
    """Publishes transactional outbox records to Redis."""

    def __init__(self, queue: RedisStreamQueue | None = None):
        self.queue = queue or RedisStreamQueue()

    async def publish_pending(self, *, limit: int = 100) -> int:
        published = 0
        factory = get_session_factory()
        async with factory() as session:
            async with session.begin():
                result = await session.execute(
                    select(OutboxEvent)
                    .where(OutboxEvent.published_at.is_(None))
                    .order_by(OutboxEvent.created_at)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
                events = list(result.scalars())
                for event in events:
                    try:
                        if event.topic == "node.ready":
                            await self.queue.publish_node(
                                str(event.payload["node_run_id"]),
                                {"run_id": str(event.aggregate_id)},
                            )
                        else:
                            await self.queue.publish_event(
                                str(event.aggregate_id), {"type": event.topic, **event.payload}
                            )
                        event.published_at = datetime.now(UTC)
                        event.attempts += 1
                        published += 1
                    except Exception as exc:
                        event.attempts += 1
                        event.last_error = str(exc)[:2000]
        return published
