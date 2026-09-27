from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentforge.config import get_settings
from agentforge.logging import get_logger
from agentforge.models import ApiKey, Membership, User, Workspace
from agentforge.security import (
    create_access_token,
    generate_api_key,
    hash_password,
    verify_api_key,
    verify_password,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class Principal:
    user_id: uuid.UUID
    workspace_id: uuid.UUID
    role: str
    auth_type: str


async def bootstrap_default_admin() -> None:
    from agentforge.db import get_session_factory

    settings = get_settings()
    factory = get_session_factory()
    async with factory() as session:
        async with session.begin():
            user = await session.scalar(select(User).where(User.email == settings.admin_email))
            if user is None:
                user = User(
                    email=settings.admin_email,
                    password_hash=hash_password(settings.admin_password),
                    display_name="Administrator",
                    is_superuser=True,
                )
                session.add(user)
                await session.flush()
            workspace = await session.scalar(select(Workspace).order_by(Workspace.created_at).limit(1))
            if workspace is None:
                workspace = Workspace(
                    name="默认工作区",
                    slug="default",
                    description="AgentForge 默认工作区",
                )
                session.add(workspace)
                await session.flush()
            membership = await session.scalar(
                select(Membership).where(Membership.user_id == user.id, Membership.workspace_id == workspace.id)
            )
            if membership is None:
                session.add(Membership(user_id=user.id, workspace_id=workspace.id, role="owner"))
            from agentforge.services.models import bootstrap_model_profiles

            await bootstrap_model_profiles(session, workspace_id=workspace.id)
    logger.info("default_admin_ready", email=settings.admin_email)


async def authenticate(session: AsyncSession, email: str, password: str) -> User | None:
    user = await session.scalar(select(User).where(User.email == email.lower()))
    if user is None or not user.is_active or not verify_password(password, user.password_hash):
        return None
    return user


def session_token(user: User) -> str:
    return create_access_token(str(user.id), extra={"email": user.email})


async def principal_from_api_key(session: AsyncSession, raw_key: str) -> Principal | None:
    prefix = raw_key[:12]
    candidates = list(
        (await session.scalars(select(ApiKey).where(ApiKey.key_prefix == prefix, ApiKey.revoked_at.is_(None)))).all()
    )
    for candidate in candidates:
        if candidate.expires_at and candidate.expires_at < datetime.now(UTC):
            continue
        if verify_api_key(raw_key, candidate.key_hash):
            candidate.last_used_at = datetime.now(UTC)
            owner_id = candidate.created_by
            if owner_id is None:
                owner_id = await session.scalar(
                    select(Membership.user_id)
                    .where(
                        Membership.workspace_id == candidate.workspace_id,
                        Membership.role.in_(["owner", "admin"]),
                    )
                    .order_by(Membership.created_at)
                    .limit(1)
                )
            if owner_id is None:
                return None
            return Principal(
                user_id=owner_id,
                workspace_id=candidate.workspace_id,
                role="admin",
                auth_type="api_key",
            )
    return None


async def principal_from_user(
    session: AsyncSession, user_id: uuid.UUID, workspace_id: uuid.UUID | None = None
) -> Principal | None:
    query = select(Membership).where(Membership.user_id == user_id).order_by(Membership.created_at).limit(1)
    if workspace_id is not None:
        query = (
            select(Membership).where(Membership.user_id == user_id, Membership.workspace_id == workspace_id).limit(1)
        )
    membership = await session.scalar(query)
    if membership is None:
        return None
    return Principal(
        user_id=user_id,
        workspace_id=membership.workspace_id,
        role=membership.role,
        auth_type="session",
    )


async def create_workspace_key(
    session: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    name: str,
    scopes: list[str],
    expires_days: int | None = None,
) -> tuple[ApiKey, str]:
    raw, prefix, key_hash = generate_api_key()
    key = ApiKey(
        workspace_id=workspace_id,
        created_by=user_id,
        name=name,
        key_prefix=prefix,
        key_hash=key_hash,
        scopes=scopes,
        expires_at=datetime.now(UTC) + timedelta(days=expires_days) if expires_days else None,
    )
    session.add(key)
    await session.flush()
    return key, raw
