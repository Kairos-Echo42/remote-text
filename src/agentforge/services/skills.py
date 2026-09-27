from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentforge.models import SkillRecord
from agentforge.plugins.skills import SkillCatalog


async def sync_skill_catalog(
    session: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    root: str | Path,
) -> int:
    catalog = SkillCatalog.load(root)
    count = 0
    for skill in catalog.skills.values():
        record = await session.scalar(
            select(SkillRecord).where(
                SkillRecord.workspace_id == workspace_id,
                SkillRecord.name == skill.manifest.name,
            )
        )
        if record is None:
            record = SkillRecord(
                workspace_id=workspace_id,
                name=skill.manifest.name,
                version=skill.manifest.version,
            )
            session.add(record)
        record.version = skill.manifest.version
        record.description = skill.manifest.description
        record.manifest = skill.manifest.model_dump(mode="json")
        record.entry_point = skill.manifest.entry_point
        record.trusted = skill.manifest.entry_point is None
        record.enabled = True
        count += 1
    await session.flush()
    return count
