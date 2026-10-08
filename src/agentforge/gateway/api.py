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
    BudgetReservation,
    Checkpoint,
    Dataset,
    DatasetVersion,
    DecisionRecord,
    Experiment,
    KnowledgeBase,
    LineageEdge,
    LineageNode,
    MCPServer,
    Membership,
    MemoryRecord,
    MetricPoint,
    ModelProfile,
    ModelVersion,
    NodeRun,
    Run,
    RunEvent,
    SkillRecord,
    TrainingAttempt,
    TrainingEvent,
    TrainingJob,
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
    BaselineStrategyRead,
    BudgetReservationRead,
    CheckpointRead,
    DatasetCreate,
    DatasetRead,
    DatasetVersionRead,
    DecisionRecordRead,
    ExperimentCreate,
    ExperimentLeaderboardItem,
    ExperimentRead,
    ExperimentReport,
    KnowledgeBaseCreate,
    KnowledgeBaseRead,
    MCPServerCreate,
    MCPServerRead,
    MemoryCreate,
    MemoryRead,
    MemoryUpdate,
    MetricPointRead,
    ModelCreate,
    ModelRead,
    ModelVersionRead,
    NodeRunRead,
    RunAccepted,
    RunCreate,
    RunRead,
    TrainingEventRead,
    TrainingJobCreate,
    TrainingJobRead,
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
from agentforge.training.audit import DecisionRecordData, DecisionResult, record_decision, record_rejection
from agentforge.training.baselines import build_baseline_registry
from agentforge.training.datasets import create_dataset, upload_dataset_version
from agentforge.training.recipes import get_recipe_registry
from agentforge.training.service import TrainingService
from agentforge.training.types import (
    DatasetKind,
    ExperimentBudget,
    ModelStage,
    SelectionPolicy,
    TaskType,
    TrainingJobSpec,
)
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
        "ml.dataset.inspect",
        "ml.recipe.list",
        "ml.experiment.create",
        "training.submit_batch",
        "training.await_batch",
        "training.results",
        "training.await_experiment",
        "training.test_metrics",
        "training.cancel",
        "model.select_best",
        "model.register_candidate",
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


@router.get("/workspaces/{workspace_id}/datasets", response_model=list[DatasetRead])
async def list_datasets(
    workspace_id: uuid.UUID,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    rows = await session.scalars(
        select(Dataset).where(Dataset.workspace_id == workspace_id).order_by(Dataset.created_at.desc())
    )
    return list(rows.all())


@router.post("/workspaces/{workspace_id}/datasets", response_model=DatasetRead, status_code=201)
async def create_training_dataset(
    workspace_id: uuid.UUID,
    payload: DatasetCreate,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    dataset = await create_dataset(
        session,
        workspace_id=workspace_id,
        name=payload.name,
        kind=DatasetKind(payload.kind),
        task_type=TaskType(payload.task_type),
        description=payload.description,
        created_by=principal.user_id,
    )
    await session.commit()
    await session.refresh(dataset)
    return dataset


@router.get("/datasets/{dataset_id}/versions", response_model=list[DatasetVersionRead])
async def list_dataset_versions(
    dataset_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    dataset = await session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    await _workspace_access(session, principal, dataset.workspace_id)
    rows = await session.scalars(
        select(DatasetVersion).where(DatasetVersion.dataset_id == dataset.id).order_by(DatasetVersion.created_at.desc())
    )
    return list(rows.all())


@router.post("/datasets/{dataset_id}/versions", response_model=DatasetVersionRead, status_code=201)
async def upload_training_dataset_version(
    dataset_id: uuid.UUID,
    file: UploadFile = File(...),
    target_column: str | None = Query(default=None),
    split_seed: int = Query(default=42, ge=0),
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    dataset = await session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    await _workspace_access(session, principal, dataset.workspace_id)
    content = await file.read()
    if len(content) > get_settings().training_max_dataset_bytes:
        raise HTTPException(status_code=413, detail="dataset exceeds configured size limit")
    version = await upload_dataset_version(
        session,
        dataset=dataset,
        filename=file.filename or "dataset.bin",
        content=content,
        artifact_store=LocalArtifactStore(),
        target_column=target_column,
        split_seed=split_seed,
    )
    await session.commit()
    await session.refresh(version)
    return version


@router.get("/workspaces/{workspace_id}/training/recipes")
async def list_training_recipes(
    workspace_id: uuid.UUID,
    principal: Principal = Depends(require_workspace),
):
    del workspace_id, principal
    return {
        "recipes": [
            {
                "recipe_id": recipe.qualified_id,
                "display_name": recipe.display_name,
                "backend": recipe.backend,
                "task_types": [item.value for item in recipe.task_types],
                "dataset_kinds": [item.value for item in recipe.dataset_kinds],
                "default_metric": recipe.default_metric,
                "default_direction": recipe.default_direction.value,
                "checksum": recipe.checksum,
                "hyperparameters": [item.model_dump(mode="json") for item in recipe.hyperparameters],
                "resources": recipe.resources.model_dump(mode="json"),
            }
            for recipe in get_recipe_registry().list()
        ]
    }


@router.get(
    "/workspaces/{workspace_id}/training/baselines",
    response_model=list[BaselineStrategyRead],
)
async def list_baseline_strategies(
    workspace_id: uuid.UUID,
    principal: Principal = Depends(require_workspace),
):
    del workspace_id, principal
    return [
        {
            **strategy.spec.model_dump(mode="json"),
            "checksum": strategy.spec.checksum,
        }
        for strategy in build_baseline_registry().list()
    ]


@router.get("/workspaces/{workspace_id}/experiments", response_model=list[ExperimentRead])
async def list_experiments(
    workspace_id: uuid.UUID,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    rows = await session.scalars(
        select(Experiment).where(Experiment.workspace_id == workspace_id).order_by(Experiment.created_at.desc())
    )
    experiments = list(rows.all())
    if principal.auth_type == "session":
        return experiments
    return [
        ExperimentRead.model_validate(experiment).model_copy(update={"final_test_metrics": None})
        for experiment in experiments
    ]


@router.post("/workspaces/{workspace_id}/experiments", response_model=ExperimentRead, status_code=201)
async def create_training_experiment(
    workspace_id: uuid.UUID,
    payload: ExperimentCreate,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    service = TrainingService()
    budget = ExperimentBudget.model_validate(payload.budget)
    policy = SelectionPolicy.model_validate(payload.selection_policy) if payload.selection_policy else None
    experiment = await service.create_experiment(
        session,
        workspace_id=workspace_id,
        dataset_version_id=payload.dataset_version_id,
        name=payload.name,
        baseline_strategy_id=payload.baseline_strategy_id,
        budget=budget,
        search_strategy=payload.search_strategy,
        selection_policy=policy,
        created_by=principal.user_id,
    )
    await session.commit()
    await session.refresh(experiment)
    return experiment


@router.get("/experiments/{experiment_id}", response_model=ExperimentRead)
async def get_experiment(
    experiment_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    experiment = await session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    await _workspace_access(session, principal, experiment.workspace_id)
    if principal.auth_type != "session":
        return ExperimentRead.model_validate(experiment).model_copy(update={"final_test_metrics": None})
    return experiment


@router.get(
    "/experiments/{experiment_id}/leaderboard",
    response_model=list[ExperimentLeaderboardItem],
)
async def get_experiment_leaderboard(
    experiment_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    experiment = await session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    await _workspace_access(session, principal, experiment.workspace_id)
    return await _experiment_leaderboard(session, experiment)


@router.get("/experiments/{experiment_id}/budget-reservations", response_model=list[BudgetReservationRead])
async def get_experiment_budget_reservations(
    experiment_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    experiment = await session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    await _workspace_access(session, principal, experiment.workspace_id)
    rows = await session.scalars(
        select(BudgetReservation)
        .where(BudgetReservation.experiment_id == experiment.id)
        .order_by(BudgetReservation.created_at)
    )
    return list(rows.all())


@router.get("/experiments/{experiment_id}/report", response_model=ExperimentReport)
async def get_experiment_report(
    experiment_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    experiment = await session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    await _workspace_access(session, principal, experiment.workspace_id)
    baseline = await session.get(TrainingJob, experiment.baseline_job_id) if experiment.baseline_job_id else None
    experiment_payload = ExperimentRead.model_validate(experiment)
    if principal.auth_type != "session":
        experiment_payload = experiment_payload.model_copy(update={"final_test_metrics": None})
    return {
        "experiment": experiment_payload.model_dump(mode="json"),
        "baseline_status": baseline.status if baseline else None,
        "budget_ledger": {
            "reserved_total_seconds": experiment.reserved_total_seconds,
            "consumed_total_seconds": experiment.consumed_total_seconds,
            "reserved_gpu_seconds": experiment.reserved_gpu_seconds,
            "consumed_gpu_seconds": experiment.consumed_gpu_seconds,
            "jobs_started": experiment.jobs_started,
        },
        "validation_leaderboard": await _experiment_leaderboard(session, experiment),
        "selection_reason": experiment.selection_reason,
        "selection_policy": experiment.selection_policy,
        "final_test_metrics": experiment.final_test_metrics if principal.auth_type == "session" else None,
        "final_test_visibility": (
            "human_review_only" if principal.auth_type == "session" else "hidden_from_agent_api_key"
        ),
    }


@router.get("/experiments/{experiment_id}/jobs", response_model=list[TrainingJobRead])
async def list_experiment_jobs(
    experiment_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    experiment = await session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    await _workspace_access(session, principal, experiment.workspace_id)
    rows = await session.scalars(
        select(TrainingJob).where(TrainingJob.experiment_id == experiment.id).order_by(TrainingJob.created_at)
    )
    return list(rows.all())


@router.post("/experiments/{experiment_id}/jobs", response_model=list[TrainingJobRead], status_code=201)
async def submit_training_jobs(
    experiment_id: uuid.UUID,
    payload: TrainingJobCreate,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    service = TrainingService()
    experiment = await session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    await _workspace_access(session, principal, experiment.workspace_id)
    jobs = [TrainingJobSpec.model_validate(item) for item in payload.jobs]
    created = await service.submit_batch(
        session,
        workspace_id=experiment.workspace_id,
        experiment_id=experiment.id,
        jobs=jobs,
        round_number=payload.round_number,
        submitted_by=principal.user_id,
    )
    await session.commit()
    return created


@router.get("/training/jobs/{job_id}", response_model=TrainingJobRead)
async def get_training_job(
    job_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    job = await _training_job_for_principal(session, job_id, principal)
    return job


@router.get("/training/jobs/{job_id}/metrics", response_model=list[MetricPointRead])
async def get_training_metrics(
    job_id: uuid.UUID,
    split: str | None = None,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    await _training_job_for_principal(session, job_id, principal)
    query = select(MetricPoint).where(MetricPoint.job_id == job_id)
    if split:
        if split == "test" and principal.auth_type != "session":
            raise HTTPException(status_code=403, detail="test metrics are hidden from agent API keys")
        query = query.where(MetricPoint.split == split)
    elif principal.auth_type != "session":
        query = query.where(MetricPoint.split != "test")
    rows = await session.scalars(query.order_by(MetricPoint.created_at))
    return list(rows.all())


@router.get("/training/jobs/{job_id}/events", response_model=list[TrainingEventRead])
async def get_training_events(
    job_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    await _training_job_for_principal(session, job_id, principal)
    rows = await session.scalars(
        select(TrainingEvent).where(TrainingEvent.job_id == job_id).order_by(TrainingEvent.seq)
    )
    events = list(rows.all())
    if principal.auth_type == "session":
        return events
    return [
        event
        for event in events
        if not (
            event.event_type == "training.metric"
            and isinstance(event.payload, dict)
            and event.payload.get("split") == "test"
        )
    ]


@router.get("/training/jobs/{job_id}/manifest")
async def get_training_manifest(
    job_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    await _training_job_for_principal(session, job_id, principal)
    attempt = await session.scalar(
        select(TrainingAttempt)
        .where(TrainingAttempt.job_id == job_id)
        .order_by(TrainingAttempt.attempt.desc())
        .limit(1)
    )
    return attempt.manifest if attempt else {}


@router.get("/training/jobs/{job_id}/checkpoints", response_model=list[CheckpointRead])
async def get_training_checkpoints(
    job_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    await _training_job_for_principal(session, job_id, principal)
    rows = await session.scalars(
        select(Checkpoint).where(Checkpoint.job_id == job_id).order_by(Checkpoint.created_at)
    )
    return list(rows.all())


@router.post("/training/jobs/{job_id}/cancel", response_model=TrainingJobRead)
async def cancel_training_job(
    job_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    job = await _training_job_for_principal(session, job_id, principal)
    if job.status not in {"succeeded", "failed", "cancelled"}:
        job.status = "cancelling"
        await TrainingService().record_cancel_request(
            session,
            job=job,
            actor=f"{principal.auth_type}:{principal.user_id}",
        )
        await session.commit()
        try:
            from agentforge.runtime.queue import get_redis

            await get_redis().set(f"agentforge:training:cancel:{job.id}", "1", ex=86_400)
        except Exception:
            pass
    await session.refresh(job)
    return job


@router.post("/experiments/{experiment_id}/select-best", response_model=TrainingJobRead)
async def select_experiment_best(
    experiment_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    experiment = await session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    await _workspace_access(session, principal, experiment.workspace_id)
    job = await TrainingService().select_best(
        session,
        experiment_id=experiment.id,
        workspace_id=experiment.workspace_id,
        actor=f"user:{principal.user_id}",
    )
    await session.commit()
    return job


@router.post("/experiments/{experiment_id}/finalize")
async def finalize_experiment(
    experiment_id: uuid.UUID,
    timeout_seconds: int = Query(default=900, ge=1),
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    experiment = await session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    await _workspace_access(session, principal, experiment.workspace_id)
    return await TrainingService().await_experiment_finalization(
        experiment_id=experiment.id, timeout_seconds=timeout_seconds
    )


@router.get("/experiments/{experiment_id}/decisions", response_model=list[DecisionRecordRead])
async def list_experiment_decisions(
    experiment_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    experiment = await session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    await _workspace_access(session, principal, experiment.workspace_id)
    rows = await session.scalars(
        select(DecisionRecord).where(DecisionRecord.experiment_id == experiment.id).order_by(DecisionRecord.created_at)
    )
    return list(rows.all())


@router.get("/experiments/{experiment_id}/lineage")
async def get_experiment_lineage(
    experiment_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    experiment = await session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    await _workspace_access(session, principal, experiment.workspace_id)
    nodes = list(
        (
            await session.scalars(
                select(LineageNode)
                .where(LineageNode.workspace_id == experiment.workspace_id)
                .order_by(LineageNode.created_at)
            )
        ).all()
    )
    edges = list(
        (
            await session.scalars(
                select(LineageEdge)
                .where(LineageEdge.workspace_id == experiment.workspace_id)
                .order_by(LineageEdge.created_at)
            )
        ).all()
    )
    return {
        "nodes": [
            {
                "id": str(node.id),
                "node_type": node.node_type,
                "ref_id": str(node.ref_id),
                "artifact_id": str(node.artifact_id) if node.artifact_id else None,
                "metadata": node.metadata_,
            }
            for node in nodes
        ],
        "edges": [
            {
                "id": str(edge.id),
                "input_node_id": str(edge.input_node_id),
                "output_node_id": str(edge.output_node_id),
                "operation": edge.operation,
                "actor_type": edge.actor_type,
                "actor_id": edge.actor_id,
                "created_at": edge.created_at.isoformat(),
            }
            for edge in edges
        ],
    }


@router.get("/workspaces/{workspace_id}/model-versions", response_model=list[ModelVersionRead])
async def list_model_versions(
    workspace_id: uuid.UUID,
    principal: Principal = Depends(require_workspace),
    session: AsyncSession = Depends(database_session),
):
    rows = await session.scalars(
        select(ModelVersion).where(ModelVersion.workspace_id == workspace_id).order_by(ModelVersion.created_at.desc())
    )
    models = list(rows.all())
    if principal.auth_type == "session":
        return models
    return [
        ModelVersionRead.model_validate(model).model_copy(update={"final_test_metrics": None})
        for model in models
    ]


@router.post("/model-versions/{model_version_id}/promote", response_model=ModelVersionRead)
async def promote_model_version(
    model_version_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    model_version = await session.get(ModelVersion, model_version_id)
    if model_version is None:
        raise HTTPException(status_code=404, detail="model version not found")
    await _workspace_access(session, principal, model_version.workspace_id)
    require_role(principal, "owner", "admin")
    if principal.auth_type != "session":
        await record_rejection(
            session,
            workspace_id=model_version.workspace_id,
            experiment_id=model_version.experiment_id,
            job_id=model_version.job_id,
            action="model.promote",
            reason_code="human_review_required",
            reason="Production promotion requires a human-authenticated session",
            actor=f"api_key:{principal.user_id}",
        )
        await session.commit()
        raise HTTPException(status_code=403, detail="human review is required for Production promotion")
    previous = await session.scalars(
        select(ModelVersion).where(
            ModelVersion.workspace_id == model_version.workspace_id,
            ModelVersion.name == model_version.name,
            ModelVersion.stage == ModelStage.PRODUCTION.value,
        )
    )
    for item in previous.all():
        item.stage = ModelStage.ARCHIVED.value
        item.archived_at = datetime.now(UTC)
    model_version.stage = ModelStage.PRODUCTION.value
    model_version.promoted_by = principal.user_id
    model_version.promoted_at = datetime.now(UTC)
    await record_decision(
        session,
        workspace_id=model_version.workspace_id,
        experiment_id=model_version.experiment_id,
        job_id=model_version.job_id,
        data=DecisionRecordData(
            action="model.promote",
            result=DecisionResult.ACCEPTED,
            reason_code="human_promotion_approved",
            reason="Human reviewer promoted the candidate to Production",
            actor=f"user:{principal.user_id}",
            phase="promotion",
            validation_metrics=model_version.validation_metrics,
        ),
    )
    await session.commit()
    await session.refresh(model_version)
    return model_version


@router.post("/model-versions/{model_version_id}/archive", response_model=ModelVersionRead)
async def archive_model_version(
    model_version_id: uuid.UUID,
    principal: Principal = Depends(require_principal),
    session: AsyncSession = Depends(database_session),
):
    model_version = await session.get(ModelVersion, model_version_id)
    if model_version is None:
        raise HTTPException(status_code=404, detail="model version not found")
    await _workspace_access(session, principal, model_version.workspace_id)
    require_role(principal, "owner", "admin")
    if principal.auth_type != "session":
        raise HTTPException(status_code=403, detail="human session required")
    model_version.stage = ModelStage.ARCHIVED.value
    model_version.archived_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(model_version)
    return model_version


async def _experiment_leaderboard(
    session: AsyncSession,
    experiment: Experiment,
) -> list[dict[str, Any]]:
    jobs = list(
        (
            await session.scalars(
                select(TrainingJob)
                .where(
                    TrainingJob.experiment_id == experiment.id,
                    TrainingJob.status == "succeeded",
                    TrainingJob.job_kind.in_(["baseline", "candidate", "refinement"]),
                )
                .order_by(TrainingJob.created_at)
            )
        ).all()
    )
    rows: list[dict[str, Any]] = []
    for job in jobs:
        attempt = await session.scalar(
            select(TrainingAttempt)
            .where(TrainingAttempt.job_id == job.id)
            .order_by(TrainingAttempt.attempt.desc())
            .limit(1)
        )
        validation = dict((attempt.metrics or {}).get("validation", {})) if attempt else {}
        objective_value = validation.get(experiment.objective_metric)
        rows.append(
            {
                "job_id": job.id,
                "job_kind": job.job_kind,
                "recipe_id": job.recipe_id,
                "round_number": job.round_number,
                "validation_metrics": validation,
                "objective_value": float(objective_value) if objective_value is not None else None,
                "actual_total_seconds": job.actual_total_seconds,
                "actual_gpu_seconds": job.actual_gpu_seconds,
                "selected": job.id == experiment.best_job_id,
            }
        )
    scored = [row for row in rows if row["objective_value"] is not None]
    unscored = [row for row in rows if row["objective_value"] is None]
    scored.sort(
        key=lambda row: (
            -float(row["objective_value"])
            if experiment.objective_direction == "maximize"
            else float(row["objective_value"]),
            float(row["actual_total_seconds"]),
            str(row["job_id"]),
        )
    )
    return scored + unscored


async def _training_job_for_principal(session: AsyncSession, job_id: uuid.UUID, principal: Principal) -> TrainingJob:
    job = await session.get(TrainingJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="training job not found")
    await _workspace_access(session, principal, job.workspace_id)
    return job


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
