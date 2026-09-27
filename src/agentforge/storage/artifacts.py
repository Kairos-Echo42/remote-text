from __future__ import annotations

import hashlib
import mimetypes
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import aiofiles

from agentforge.config import get_settings


@dataclass(slots=True)
class StoredObject:
    key: str
    size: int
    checksum: str
    content_type: str


class ArtifactStore(Protocol):
    async def put(
        self,
        content: bytes,
        *,
        workspace_id: uuid.UUID,
        run_id: uuid.UUID | None,
        filename: str,
        content_type: str | None = None,
    ) -> StoredObject: ...

    async def get(self, key: str) -> bytes: ...

    async def delete(self, key: str) -> None: ...


class LocalArtifactStore:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or get_settings().artifact_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        target = (self.root / key).resolve()
        if self.root not in target.parents and target != self.root:
            raise ValueError("artifact key escapes the configured root")
        return target

    async def put(
        self,
        content: bytes,
        *,
        workspace_id: uuid.UUID,
        run_id: uuid.UUID | None,
        filename: str,
        content_type: str | None = None,
    ) -> StoredObject:
        checksum = hashlib.sha256(content).hexdigest()
        safe_name = Path(filename).name
        prefix = f"{workspace_id}/{run_id or 'unassigned'}/{checksum[:2]}"
        key = f"{prefix}/{uuid.uuid4().hex}-{safe_name}"
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "wb") as handle:
            await handle.write(content)
        detected = content_type or mimetypes.guess_type(safe_name)[0]
        return StoredObject(
            key=key,
            size=len(content),
            checksum=checksum,
            content_type=detected or "application/octet-stream",
        )

    async def get(self, key: str) -> bytes:
        async with aiofiles.open(self._resolve(key), "rb") as handle:
            return await handle.read()

    async def delete(self, key: str) -> None:
        target = self._resolve(key)
        if target.exists():
            target.unlink()


class S3ArtifactStore:
    """Reserved S3-compatible adapter; v0.1 ships the local implementation only."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("S3 ArtifactStore will be provided in a future release")
