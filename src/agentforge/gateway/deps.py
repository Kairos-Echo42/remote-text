from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentforge.db import get_session
from agentforge.models import Membership, Workspace
from agentforge.security import AuthenticationError, decode_access_token
from agentforge.services.auth import Principal, principal_from_api_key, principal_from_user


async def database_session() -> AsyncIterator[AsyncSession]:
    async for session in get_session():
        yield session


async def optional_principal(request: Request, session: AsyncSession = Depends(database_session)) -> Principal | None:
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        if token.startswith("af_"):
            return await principal_from_api_key(session, token)
        try:
            payload = decode_access_token(token)
            return await principal_from_user(session, uuid.UUID(payload["sub"]))
        except (AuthenticationError, ValueError):
            return None
    cookie = request.cookies.get("agentforge_session")
    if cookie:
        try:
            payload = decode_access_token(cookie)
            return await principal_from_user(session, uuid.UUID(payload["sub"]))
        except (AuthenticationError, ValueError):
            return None
    return None


async def require_principal(
    principal: Principal | None = Depends(optional_principal),
) -> Principal:
    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required")
    return principal


async def require_workspace(
    workspace_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
) -> Principal:
    if principal.workspace_id == workspace_id:
        return principal
    membership = await session.scalar(
        select(Membership).where(
            Membership.user_id == principal.user_id,
            Membership.workspace_id == workspace_id,
        )
    )
    if membership is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    return Principal(
        user_id=principal.user_id,
        workspace_id=workspace_id,
        role=membership.role,
        auth_type=principal.auth_type,
    )


def require_role(principal: Principal, *roles: str) -> None:
    if principal.role not in set(roles):
        raise HTTPException(status_code=403, detail="insufficient permissions")


def validate_csrf(request: Request, provided: str | None = None) -> None:
    cookie = request.cookies.get("agentforge_csrf")
    candidate = provided or request.headers.get("X-CSRF-Token")
    if not cookie or not candidate or cookie != candidate:
        raise HTTPException(status_code=403, detail="invalid CSRF token")


async def default_workspace(
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
) -> Workspace:
    workspace = await session.get(Workspace, principal.workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    return workspace
