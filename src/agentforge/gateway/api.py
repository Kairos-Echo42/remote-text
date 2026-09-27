from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from agentforge.config import get_settings
from agentforge.db import get_session_factory
from agentforge.gateway.deps import (
    database_session,
    require_principal,
    require_role,
    require_workspace,
)
from agentforge.models import (
    AgentDefinition,
    Artifact,
    KnowledgeBase,
    MCPServer,
    Membership,
    MemoryRecord,
    ModelProfile,
    NodeRun,
    Run,
    RunEvent,
    SkillRecord,
    WorkflowDefinition,
    WorkflowVersion,
    Workspace,
)
from agentforge.runtime.events import EventService
from agentforge.runtime.state import RUN_TERMINAL_STATES
from agentforge.schemas import (
    ApiKeyCreate,
    ApiKeyCreated,
    ArtifactRead,
    KnowledgeBaseCreate,
    KnowledgeBaseRead,
    MCPServerCreate,
    MCPServerRead,
    MemoryCreate,
    MemoryRead,
    MemoryUpdate,
    ModelCreate,
    ModelRead,
    NodeRunRead,
    RunAccepted,
    RunCreate,
    RunRead,
    WorkflowRead,
    WorkflowVersionRead,
    WorkspaceCreate,
    WorkspaceRead,
)
from agentforge.security import encrypt_secret
from agentforge.services.auth import Principal, create_workspace_key
from agentforge.services.knowledge import ingest_document
from agentforge.services.models import resolve_embedding_provider
from agentforge.services.runs import (
    InvalidRunState,
    RunNotFoundError,
    create_run,
    request_cancel,
    request_pause,
    request_resume,
    request_retry,
)
from agentforge.services.workflows import WorkflowValidationError, sync_workflow
from agentforge.storage.artifacts import LocalArtifactStore
from agentforge.workflow import WorkflowDocument

router = APIRouter(prefix="/api/v1")
events = EventService()


@router.get("/me")
async def me(principal: Principal = Depends(require_principal)) -> dict[str, Any]:
    return {
        "user_id": str(principal.user_id),
        "workspace_id": str(principal.workspace_id),
        "role": principal.role,
        "auth_type": principal.auth_type,
    }


@router.get("/workspaces", response_model=list[WorkspaceRead])
async def list_workspaces(
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    result = await session.scalars(
        select(Workspace)
        .join(Membership, Membership.workspace_id == Workspace.id)
        .where(Membership.user_id == principal.user_id)
        .order_by(Workspace.created_at)
    )
    return list(result.all())


@router.post("/workspaces", response_model=WorkspaceRead, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    payload: WorkspaceCreate,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    workspace = Workspace(**payload.model_dump())
    session.add(workspace)
    await session.flush()
    session.add(Membership(user_id=principal.user_id, workspace_id=workspace.id, role="owner"))
    await session.commit()
    await session.refresh(workspace)
    return workspace


@router.get("/workspaces/{workspace_id}/workflows", response_model=list[WorkflowRead])
async def list_workflows(
    workspace_id: uuid.UUID,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    result = await session.scalars(
        select(WorkflowDefinition)
        .where(WorkflowDefinition.workspace_id == workspace_id)
        .order_by(WorkflowDefinition.name)
    )
    return list(result.all())


@router.get(
    "/workspaces/{workspace_id}/workflows/{workflow_id}/versions",
    response_model=list[WorkflowVersionRead],
)
async def list_workflow_versions(
    workspace_id: uuid.UUID,
    workflow_id: uuid.UUID,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    result = await session.scalars(
        select(WorkflowVersion)
        .join(WorkflowDefinition, WorkflowVersion.workflow_id == WorkflowDefinition.id)
        .where(
            WorkflowVersion.workflow_id == workflow_id,
            WorkflowDefinition.workspace_id == workspace_id,
        )
        .order_by(WorkflowVersion.created_at.desc())
    )
    return list(result.all())


@router.post("/workspaces/{workspace_id}/workflows/sync", response_model=WorkflowVersionRead)
async def sync_workflow_document(
    workspace_id: uuid.UUID,
    payload: dict[str, Any],
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    require_role(principal, "owner", "admin")
    try:
        document = WorkflowDocument.model_validate(payload)
        _, version = await sync_workflow(session, workspace_id, document, published_by=principal.user_id)
        await session.commit()
        await session.refresh(version)
        return version
    except (WorkflowValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/workspaces/{workspace_id}/agents")
async def list_agents(
    workspace_id: uuid.UUID,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    result = await session.scalars(
        select(AgentDefinition).where(AgentDefinition.workspace_id == workspace_id).order_by(AgentDefinition.name)
    )
    return [
        {
            "id": str(agent.id),
            "name": agent.name,
            "version": agent.version,
            "description": agent.description,
            "capabilities": agent.capabilities,
            "limits": agent.limits,
            "updated_at": agent.updated_at,
        }
        for agent in result.all()
    ]


@router.get("/workspaces/{workspace_id}/capabilities")
async def list_capabilities(
    workspace_id: uuid.UUID,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    servers = list((await session.scalars(select(MCPServer).where(MCPServer.workspace_id == workspace_id))).all())
    builtins = [
        "fixture.echo",
        "web.search",
        "knowledge.search",
        "memory.search",
        "memory.write",
        "workspace.read",
        "workspace.write",
        "workspace.list",
        "sandbox.execute",
        "artifact.write",
    ]
    return {
        "builtins": builtins,
        "mcp": [
            {
                "server": server.name,
                "enabled": server.enabled,
                "tools": server.tool_cache,
            }
            for server in servers
        ],
    }


@router.post(
    "/workspaces/{workspace_id}/runs",
    response_model=RunAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_agent_run(
    workspace_id: uuid.UUID,
    payload: RunCreate,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    try:
        run = await create_run(
            session,
            workspace_id=workspace_id,
            workflow_version_id=payload.workflow_version_id,
            input_data=payload.input,
            metadata=payload.metadata,
            created_by=principal.user_id,
        )
        await events.append(session, run.id, "run.queued", {"workflow_version_id": str(payload.workflow_version_id)})
        await session.commit()
        return RunAccepted(
            run_id=run.id,
            status=run.status,
            event_url=f"/api/v1/runs/{run.id}/events",
        )
    except RunNotFoundError as exc:
        await session.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}", response_model=RunRead)
async def read_run(
    run_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    run = await _run_for_principal(session, run_id, principal)
    return run


@router.get("/runs/{run_id}/nodes", response_model=list[NodeRunRead])
async def read_run_nodes(
    run_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    await _run_for_principal(session, run_id, principal)
    nodes = list(
        (await session.scalars(select(NodeRun).where(NodeRun.run_id == run_id).order_by(NodeRun.attempt))).all()
    )
    latest: dict[str, NodeRun] = {}
    for node in nodes:
        current = latest.get(node.node_id)
        if current is None or node.attempt > current.attempt:
            latest[node.node_id] = node
    return list(latest.values())


@router.post("/runs/{run_id}/pause", response_model=RunRead)
async def pause_run(
    run_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    return await _run_action(session, run_id, principal, request_pause, "run.paused")


@router.post("/runs/{run_id}/resume", response_model=RunRead)
async def resume_run(
    run_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    return await _run_action(session, run_id, principal, request_resume, "run.resumed")


@router.post("/runs/{run_id}/cancel", response_model=RunRead)
async def cancel_run(
    run_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    run = await _run_action(session, run_id, principal, request_cancel, "run.cancel_requested")
    try:
        from agentforge.runtime.queue import get_redis

        await get_redis().set(f"agentforge:cancel:{run.id}", "1", ex=86_400)
    except Exception:
        pass
    return run


@router.post("/runs/{run_id}/retry", response_model=RunRead)
async def retry_run(
    run_id: uuid.UUID,
    node_id: str | None = Query(default=None),
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    run = await _run_for_principal(session, run_id, principal)
    try:
        await request_retry(session, run.id, run.workspace_id, node_id=node_id)
        await events.append(session, run.id, "run.retry_requested", {"node_id": node_id})
        await session.commit()
        await session.refresh(run)
        return run
    except InvalidRunState as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: uuid.UUID,
    request: Request,
    after: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(database_session),
):
    token = request.query_params.get("token")
    principal = await _event_principal(request, session, token)
    if principal is None:
        raise HTTPException(status_code=401, detail="authentication required")
    await _run_for_principal(session, run_id, principal)

    async def event_generator():
        last_seq = max(after, _last_event_id(request))
        while True:
            if await request.is_disconnected():
                break
            factory = get_session_factory()
            async with factory() as stream_session:
                rows = list(
                    (
                        await stream_session.scalars(
                            select(RunEvent)
                            .where(RunEvent.run_id == run_id, RunEvent.seq > last_seq)
                            .order_by(RunEvent.seq)
                            .limit(200)
                        )
                    ).all()
                )
                for event in rows:
                    last_seq = event.seq
                    yield {
                        "id": str(event.seq),
                        "event": event.event_type,
                        "data": json.dumps(
                            {
                                "id": str(event.id),
                                "run_id": str(event.run_id),
                                "node_run_id": str(event.node_run_id) if event.node_run_id else None,
                                "seq": event.seq,
                                "type": event.event_type,
                                "payload": event.payload,
                                "created_at": event.created_at.isoformat(),
                            },
                            ensure_ascii=False,
                        ),
                    }
                current_run = await stream_session.get(Run, run_id)
            if current_run and current_run.status in RUN_TERMINAL_STATES and not rows:
                break
            await asyncio.sleep(1)

    return EventSourceResponse(event_generator(), ping=get_settings().sse_heartbeat_seconds)


@router.get("/workspaces/{workspace_id}/models", response_model=list[ModelRead])
async def list_models(
    workspace_id: uuid.UUID,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    result = await session.scalars(
        select(ModelProfile).where(ModelProfile.workspace_id == workspace_id).order_by(ModelProfile.name)
    )
    return list(result.all())


@router.post("/workspaces/{workspace_id}/models", response_model=ModelRead, status_code=201)
async def create_model(
    workspace_id: uuid.UUID,
    payload: ModelCreate,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    require_role(principal, "owner", "admin")
    if payload.api_key:
        from agentforge.models import Secret

        secret = Secret(
            workspace_id=workspace_id,
            name=f"model:{payload.name}",
            encrypted_value=encrypt_secret(payload.api_key),
        )
        session.add(secret)
        await session.flush()
        credential_id = secret.id
    else:
        credential_id = None
    if payload.is_default:
        for existing in (
            await session.scalars(
                select(ModelProfile).where(
                    ModelProfile.workspace_id == workspace_id,
                    ModelProfile.kind == payload.kind,
                )
            )
        ).all():
            existing.is_default = False
    profile = ModelProfile(
        workspace_id=workspace_id,
        name=payload.name,
        kind=payload.kind,
        provider=payload.provider,
        model=payload.model,
        base_url=payload.base_url,
        credential_id=credential_id,
        capabilities=payload.capabilities,
        default_params=payload.default_params,
        max_context_tokens=payload.max_context_tokens,
        is_default=payload.is_default,
    )
    session.add(profile)
    await session.commit()
    await session.refresh(profile)
    return profile


@router.get("/workspaces/{workspace_id}/mcp-servers", response_model=list[MCPServerRead])
async def list_mcp_servers(
    workspace_id: uuid.UUID,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    result = await session.scalars(
        select(MCPServer).where(MCPServer.workspace_id == workspace_id).order_by(MCPServer.name)
    )
    return list(result.all())


@router.post("/workspaces/{workspace_id}/mcp-servers", response_model=MCPServerRead, status_code=201)
async def create_mcp_server(
    workspace_id: uuid.UUID,
    payload: MCPServerCreate,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    require_role(principal, "owner", "admin")
    server = MCPServer(
        workspace_id=workspace_id,
        name=payload.name,
        transport=payload.transport,
        endpoint=payload.endpoint,
        command=payload.command,
        enabled=payload.enabled,
    )
    session.add(server)
    await session.commit()
    await session.refresh(server)
    return server


@router.get("/workspaces/{workspace_id}/knowledge-bases", response_model=list[KnowledgeBaseRead])
async def list_knowledge_bases(
    workspace_id: uuid.UUID,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    result = await session.scalars(
        select(KnowledgeBase).where(KnowledgeBase.workspace_id == workspace_id).order_by(KnowledgeBase.name)
    )
    return list(result.all())


@router.post("/workspaces/{workspace_id}/knowledge-bases", response_model=KnowledgeBaseRead, status_code=201)
async def create_knowledge_base(
    workspace_id: uuid.UUID,
    payload: KnowledgeBaseCreate,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    knowledge_base = KnowledgeBase(
        workspace_id=workspace_id,
        name=payload.name,
        description=payload.description,
        embedding_model_id=payload.embedding_model_id,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
    )
    session.add(knowledge_base)
    await session.commit()
    await session.refresh(knowledge_base)
    return knowledge_base


@router.post("/knowledge-bases/{knowledge_base_id}/documents")
async def upload_document(
    knowledge_base_id: uuid.UUID,
    file: UploadFile = File(...),
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
    if knowledge_base is None:
        raise HTTPException(status_code=404, detail="knowledge base not found")
    await _workspace_access(session, principal, knowledge_base.workspace_id)
    embedding_provider = await resolve_embedding_provider(session, knowledge_base.workspace_id)
    document = await ingest_document(
        session,
        knowledge_base=knowledge_base,
        filename=file.filename or "upload.txt",
        content=await file.read(),
        content_type=file.content_type,
        artifact_store=LocalArtifactStore(),
        embedding_provider=embedding_provider,
    )
    await session.commit()
    await session.refresh(document)
    return {
        "id": str(document.id),
        "filename": document.filename,
        "status": document.status,
        "error": document.error,
    }


@router.get("/workspaces/{workspace_id}/memories", response_model=list[MemoryRead])
async def list_memories(
    workspace_id: uuid.UUID,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    result = await session.scalars(
        select(MemoryRecord)
        .where(MemoryRecord.workspace_id == workspace_id, MemoryRecord.deleted_at.is_(None))
        .order_by(MemoryRecord.updated_at.desc())
    )
    return list(result.all())


@router.post("/workspaces/{workspace_id}/memories", response_model=MemoryRead, status_code=201)
async def create_memory(
    workspace_id: uuid.UUID,
    payload: MemoryCreate,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    from agentforge.services.memory import MemoryService

    embedding_provider = await resolve_embedding_provider(session, workspace_id)
    record = await MemoryService(embedding_provider).create(
        session,
        workspace_id=workspace_id,
        content=payload.content,
        agent_name=payload.agent_name,
        scope=payload.scope,
        kind=payload.kind,
        confidence=payload.confidence,
    )
    await session.commit()
    await session.refresh(record)
    return record


@router.patch("/memories/{memory_id}", response_model=MemoryRead)
async def update_memory(
    memory_id: uuid.UUID,
    payload: MemoryUpdate,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    record = await session.get(MemoryRecord, memory_id)
    if record is None:
        raise HTTPException(status_code=404, detail="memory not found")
    await _workspace_access(session, principal, record.workspace_id)
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(record, key, value)
    await session.commit()
    await session.refresh(record)
    return record


@router.delete("/memories/{memory_id}", status_code=204)
async def delete_memory(
    memory_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    record = await session.get(MemoryRecord, memory_id)
    if record is None:
        raise HTTPException(status_code=404, detail="memory not found")
    await _workspace_access(session, principal, record.workspace_id)
    record.deleted_at = datetime.now(UTC)
    await session.commit()
    return Response(status_code=204)


@router.get("/workspaces/{workspace_id}/artifacts", response_model=list[ArtifactRead])
async def list_artifacts(
    workspace_id: uuid.UUID,
    run_id: uuid.UUID | None = None,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    query = select(Artifact).where(Artifact.workspace_id == workspace_id)
    if run_id:
        query = query.where(Artifact.run_id == run_id)
    result = await session.scalars(query.order_by(Artifact.created_at.desc()))
    return list(result.all())


@router.get("/artifacts/{artifact_id}")
async def download_artifact(
    artifact_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    artifact = await session.get(Artifact, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    await _workspace_access(session, principal, artifact.workspace_id)
    content = await LocalArtifactStore().get(artifact.storage_key)
    return Response(
        content,
        media_type=artifact.content_type,
        headers={"Content-Disposition": f'attachment; filename="{artifact.name}"'},
    )


@router.get("/workspaces/{workspace_id}/api-keys")
async def list_api_keys(
    workspace_id: uuid.UUID,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    require_role(principal, "owner", "admin")
    from agentforge.models import ApiKey

    rows = list(
        (
            await session.scalars(
                select(ApiKey).where(ApiKey.workspace_id == workspace_id, ApiKey.revoked_at.is_(None))
            )
        ).all()
    )
    return [
        {
            "id": str(key.id),
            "name": key.name,
            "key_prefix": key.key_prefix,
            "scopes": key.scopes,
            "expires_at": key.expires_at,
            "created_at": key.created_at,
        }
        for key in rows
    ]


@router.post(
    "/workspaces/{workspace_id}/api-keys",
    response_model=ApiKeyCreated,
    status_code=201,
)
async def create_api_key(
    workspace_id: uuid.UUID,
    payload: ApiKeyCreate,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    require_role(principal, "owner", "admin")
    key, raw = await create_workspace_key(
        session,
        workspace_id=workspace_id,
        user_id=principal.user_id,
        name=payload.name,
        scopes=payload.scopes,
    )
    await session.commit()
    return ApiKeyCreated(
        id=key.id,
        name=key.name,
        key=raw,
        key_prefix=key.key_prefix,
        scopes=key.scopes,
    )


@router.get("/workspaces/{workspace_id}/skills")
async def list_skills(
    workspace_id: uuid.UUID,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    rows = list(
        (
            await session.scalars(
                select(SkillRecord).where(SkillRecord.workspace_id == workspace_id).order_by(SkillRecord.name)
            )
        ).all()
    )
    return [
        {
            "id": str(skill.id),
            "name": skill.name,
            "version": skill.version,
            "description": skill.description,
            "enabled": skill.enabled,
            "trusted": skill.trusted,
        }
        for skill in rows
    ]


async def _run_action(session, run_id, principal, action, event_type: str):
    run = await _run_for_principal(session, run_id, principal)
    try:
        result = await action(session, run.id, run.workspace_id)
        await events.append(session, run.id, event_type, {})
        await session.commit()
        await session.refresh(result)
        return result
    except InvalidRunState as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc


async def _run_for_principal(session: AsyncSession, run_id: uuid.UUID, principal: Principal) -> Run:
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    await _workspace_access(session, principal, run.workspace_id)
    return run


async def _workspace_access(session: AsyncSession, principal: Principal, workspace_id: uuid.UUID) -> None:
    if principal.workspace_id == workspace_id:
        return
    membership = await session.scalar(
        select(Membership).where(
            Membership.user_id == principal.user_id,
            Membership.workspace_id == workspace_id,
        )
    )
    if membership is None:
        raise HTTPException(status_code=404, detail="resource not found")


async def _event_principal(request: Request, session: AsyncSession, token: str | None) -> Principal | None:
    from agentforge.security import AuthenticationError, decode_access_token
    from agentforge.services.auth import principal_from_api_key, principal_from_user

    if token:
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


def _last_event_id(request: Request) -> int:
    raw = request.headers.get("Last-Event-ID") or "0"
    try:
        return int(raw)
    except ValueError:
        return 0
