from __future__ import annotations

import pytest
from sqlalchemy import select

from agentforge.config import get_settings
from agentforge.db import get_session_factory
from agentforge.models import ModelProfile, Workspace
from agentforge.services.models import bootstrap_model_profiles


@pytest.mark.asyncio
async def test_bootstrap_creates_chat_and_embedding_profiles(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "deepseek_api_key", "test-deepseek-key")
    monkeypatch.setattr(settings, "dashscope_api_key", "test-dashscope-key")
    async with get_session_factory()() as session:
        workspace = await session.scalar(select(Workspace).limit(1))
        await bootstrap_model_profiles(session, workspace_id=workspace.id)
        await session.commit()
        profiles = list(
            (
                await session.scalars(
                    select(ModelProfile).order_by(ModelProfile.kind, ModelProfile.name)
                )
            ).all()
        )
    assert [(profile.name, profile.kind) for profile in profiles] == [
        ("deepseek-chat", "chat"),
        ("dashscope-embedding", "embedding"),
    ]
    assert all(profile.is_default for profile in profiles)