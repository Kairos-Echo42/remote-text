from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from agentforge.config import get_settings
from agentforge.models import ModelProfile, Secret
from agentforge.providers.base import EmbeddingProvider
from agentforge.providers.factory import create_embedding_provider
from agentforge.security import encrypt_secret


async def bootstrap_model_profiles(
    session: AsyncSession,
    *,
    workspace_id: uuid.UUID,
) -> list[ModelProfile]:
    """Create/update environment-backed chat and embedding profiles."""
    settings = get_settings()
    profiles: list[ModelProfile] = []
    if settings.deepseek_api_key:
        credential_id = await _upsert_secret(
            session,
            workspace_id=workspace_id,
            name="provider:deepseek",
            value=settings.deepseek_api_key,
        )
        profiles.append(
            await _upsert_profile(
                session,
                workspace_id=workspace_id,
                name="deepseek-chat",
                kind="chat",
                provider="openai_compatible",
                model=settings.deepseek_chat_model,
                base_url=settings.deepseek_base_url,
                credential_id=credential_id,
                capabilities={"chat": True, "tools": True},
            )
        )
    if settings.dashscope_api_key:
        credential_id = await _upsert_secret(
            session,
            workspace_id=workspace_id,
            name="provider:dashscope",
            value=settings.dashscope_api_key,
        )
        profiles.append(
            await _upsert_profile(
                session,
                workspace_id=workspace_id,
                name="dashscope-embedding",
                kind="embedding",
                provider="openai_compatible",
                model=settings.dashscope_embedding_model,
                base_url=settings.dashscope_base_url,
                credential_id=credential_id,
                capabilities={"embeddings": True},
            )
        )
    return profiles


async def resolve_embedding_provider(
    session: AsyncSession,
    workspace_id: uuid.UUID,
) -> EmbeddingProvider:
    profile = await session.scalar(
        select(ModelProfile)
        .options(selectinload(ModelProfile.credential))
        .where(
            ModelProfile.workspace_id == workspace_id,
            ModelProfile.kind == "embedding",
            ModelProfile.is_default.is_(True),
        )
        .order_by(ModelProfile.updated_at.desc())
        .limit(1)
    )
    return create_embedding_provider(profile)


async def _upsert_secret(
    session: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    name: str,
    value: str,
) -> uuid.UUID:
    secret = await session.scalar(select(Secret).where(Secret.workspace_id == workspace_id, Secret.name == name))
    encrypted_value = encrypt_secret(value)
    if secret is None:
        secret = Secret(
            workspace_id=workspace_id,
            name=name,
            encrypted_value=encrypted_value,
        )
        session.add(secret)
        await session.flush()
    else:
        secret.encrypted_value = encrypted_value
    return secret.id


async def _upsert_profile(
    session: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    name: str,
    kind: str,
    provider: str,
    model: str,
    base_url: str,
    credential_id: uuid.UUID,
    capabilities: dict,
) -> ModelProfile:
    profile = await session.scalar(
        select(ModelProfile).where(
            ModelProfile.workspace_id == workspace_id,
            ModelProfile.name == name,
        )
    )
    if profile is None:
        profile = ModelProfile(
            workspace_id=workspace_id,
            name=name,
            kind=kind,
        )
        session.add(profile)
    profile.kind = kind
    profile.provider = provider
    profile.model = model
    profile.base_url = base_url
    profile.credential_id = credential_id
    profile.capabilities = capabilities
    profile.default_params = {}
    profile.is_default = True
    await session.flush()
    return profile
