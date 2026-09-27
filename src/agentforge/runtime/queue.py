from __future__ import annotations

from typing import Any

import redis.asyncio as redis
from redis.exceptions import ResponseError

from agentforge.config import get_settings

_redis: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(get_settings().redis_url, decode_responses=True)
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


class RedisStreamQueue:
    NODE_STREAM = "agentforge:nodes"
    EVENT_STREAM_PREFIX = "agentforge:events"
    WORKER_GROUP = "agentforge-workers"

    def __init__(self, client: redis.Redis | None = None):
        self.client = client or get_redis()

    async def ensure_worker_group(self) -> None:
        try:
            await self.client.xgroup_create(self.NODE_STREAM, self.WORKER_GROUP, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def publish_node(self, node_run_id: str, payload: dict[str, Any] | None = None) -> str:
        fields = {"node_run_id": node_run_id}
        if payload:
            fields.update({key: str(value) for key, value in payload.items()})
        return await self.client.xadd(self.NODE_STREAM, fields)  # type: ignore[arg-type]

    async def consume_node(
        self, consumer: str, *, block_ms: int = 5_000, count: int = 1
    ) -> list[tuple[str, dict[str, str]]]:
        await self.ensure_worker_group()
        response = await self.client.xreadgroup(
            self.WORKER_GROUP,
            consumer,
            {self.NODE_STREAM: ">"},
            count=count,
            block=block_ms,
        )
        messages: list[tuple[str, dict[str, str]]] = []
        for _, items in response or []:
            messages.extend((message_id, fields) for message_id, fields in items)
        return messages

    async def acknowledge_node(self, message_id: str) -> int:
        return await self.client.xack(self.NODE_STREAM, self.WORKER_GROUP, message_id)

    async def claim_stale(
        self, consumer: str, *, min_idle_ms: int, count: int = 20
    ) -> list[tuple[str, dict[str, str]]]:
        response = await self.client.xautoclaim(
            self.NODE_STREAM,
            self.WORKER_GROUP,
            consumer,
            min_idle_time=min_idle_ms,
            start_id="0-0",
            count=count,
        )
        _, items, _ = response
        return [(message_id, fields) for message_id, fields in items]

    async def publish_event(self, run_id: str, fields: dict[str, Any], *, maxlen: int = 10_000) -> str:
        return await self.client.xadd(
            f"{self.EVENT_STREAM_PREFIX}:{run_id}",
            {key: str(value) for key, value in fields.items()},
            maxlen=maxlen,
            approximate=True,
        )

    async def read_events(self, run_id: str, after_id: str = "0-0", *, count: int = 200):
        return await self.client.xread(
            {f"{self.EVENT_STREAM_PREFIX}:{run_id}": after_id},
            count=count,
            block=None,
        )
