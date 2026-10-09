from __future__ import annotations

import shlex
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from agentforge.config import get_settings
from agentforge.gateway.deps import database_session, optional_principal, validate_csrf
from agentforge.models import (
    AgentDefinition,
    ApiKey,
    Artifact,
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
    SkillRecord,
    TrainingAttempt,
    TrainingEvent,
    TrainingJob,
    User,
    WorkflowDefinition,
    WorkflowVersion,
    Workspace,
)
from agentforge.runtime.events import EventService
from agentforge.security import encrypt_secret, hash_password
from agentforge.services.auth import Principal, authenticate, create_workspace_key, session_token
from agentforge.services.knowledge import ingest_document
from agentforge.services.memory import MemoryService
from agentforge.services.models import resolve_embedding_provider
from agentforge.services.runs import (
    InvalidRunState,
    create_run,
    request_cancel,
    request_pause,
    request_resume,
    request_retry,
)
from agentforge.storage.artifacts import LocalArtifactStore
from agentforge.training.audit import DecisionRecordData, DecisionResult, record_decision
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
)

router = APIRouter()
settings = get_settings()
templates = Jinja2Templates(directory=str(settings.web_dir / "templates"))
events = EventService()


def _context(request: Request, **values: Any) -> dict[str, Any]:
    return {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "current_path": request.url.path,
        **values,
    }


@router.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse("/app", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", _context(request, error=None))


@router.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    session: AsyncSession = Depends(database_session),
):
    user = await authenticate(session, email, password)
    if user is None:
        return templates.TemplateResponse(
            request,
            "login.html",
            _context(request, error="邮箱或密码不正确"),
            status_code=401,
        )
    response = RedirectResponse("/app", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        "agentforge_session",
        session_token(user),
        httponly=True,
        samesite="lax",
        secure=settings.is_production,
        max_age=12 * 60 * 60,
    )
    return response


@router.post("/logout")
async def logout(request: Request):
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    response = RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("agentforge_session")
    return response


@router.get("/app", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    workspace = await session.get(Workspace, principal.workspace_id)
    counts = {
        "runs": await session.scalar(select(func.count(Run.id)).where(Run.workspace_id == principal.workspace_id)) or 0,
        "active": await session.scalar(
            select(func.count(Run.id)).where(
                Run.workspace_id == principal.workspace_id,
                Run.status.in_(["queued", "running", "paused", "cancelling"]),
            )
        )
        or 0,
        "agents": await session.scalar(
            select(func.count(AgentDefinition.id)).where(AgentDefinition.workspace_id == principal.workspace_id)
        )
        or 0,
        "knowledge_bases": await session.scalar(
            select(func.count(KnowledgeBase.id)).where(KnowledgeBase.workspace_id == principal.workspace_id)
        )
        or 0,
    }
    recent_runs = list(
        (
            await session.scalars(
                select(Run).where(Run.workspace_id == principal.workspace_id).order_by(Run.created_at.desc()).limit(8)
            )
        ).all()
    )
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        _context(
            request,
            principal=principal,
            workspace=workspace,
            counts=counts,
            recent_runs=recent_runs,
        ),
    )


@router.get("/app/workbench", response_class=HTMLResponse)
async def workbench(
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    workflows = list(
        (
            await session.scalars(
                select(WorkflowDefinition)
                .where(WorkflowDefinition.workspace_id == principal.workspace_id)
                .order_by(WorkflowDefinition.name)
            )
        ).all()
    )
    versions: list[WorkflowVersion] = []
    if workflows:
        versions = list(
            (
                await session.scalars(
                    select(WorkflowVersion)
                    .where(WorkflowVersion.workflow_id.in_([workflow.id for workflow in workflows]))
                    .order_by(WorkflowVersion.created_at.desc())
                )
            ).all()
        )
    return templates.TemplateResponse(
        request,
        "workbench.html",
        _context(request, principal=principal, workflows=workflows, versions=versions),
    )


@router.post("/app/workbench/run")
async def workbench_run(
    request: Request,
    workflow_version_id: str = Form(...),
    input_json: str = Form("{}"),
    session: AsyncSession = Depends(database_session),
    principal: Principal | None = Depends(optional_principal),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    assert principal is not None
    try:
        import json

        input_data = json.loads(input_json)
        run = await create_run(
            session,
            workspace_id=principal.workspace_id,
            workflow_version_id=uuid.UUID(workflow_version_id),
            input_data=input_data,
            created_by=principal.user_id,
        )
        await events.append(session, run.id, "run.queued", {})
        await session.commit()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RedirectResponse(f"/app/runs/{run.id}", status_code=303)


@router.get("/app/runs", response_class=HTMLResponse)
async def runs_page(
    request: Request,
    status_filter: str | None = None,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    query = select(Run).where(Run.workspace_id == principal.workspace_id)
    if status_filter:
        query = query.where(Run.status == status_filter)
    runs = list((await session.scalars(query.order_by(Run.created_at.desc()).limit(100))).all())
    return templates.TemplateResponse(
        request,
        "runs.html",
        _context(request, principal=principal, runs=runs, status_filter=status_filter),
    )


@router.get("/app/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(
    run_id: uuid.UUID,
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    run = await session.get(Run, run_id)
    if run is None or run.workspace_id != principal.workspace_id:
        raise HTTPException(status_code=404, detail="run not found")
    version = await session.get(WorkflowVersion, run.workflow_version_id)
    nodes = list(
        (await session.scalars(select(NodeRun).where(NodeRun.run_id == run.id).order_by(NodeRun.attempt))).all()
    )
    latest: dict[str, NodeRun] = {}
    for node in nodes:
        current = latest.get(node.node_id)
        if current is None or node.attempt > current.attempt:
            latest[node.node_id] = node
    artifacts = list(
        (await session.scalars(select(Artifact).where(Artifact.run_id == run.id).order_by(Artifact.created_at))).all()
    )
    return templates.TemplateResponse(
        request,
        "run_detail.html",
        _context(
            request,
            principal=principal,
            run=run,
            version=version,
            nodes=list(latest.values()),
            artifacts=artifacts,
        ),
    )


@router.post("/app/runs/{run_id}/{action}")
async def run_action(
    run_id: uuid.UUID,
    action: str,
    request: Request,
    session: AsyncSession = Depends(database_session),
    principal: Principal | None = Depends(optional_principal),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    assert principal is not None
    run = await session.get(Run, run_id)
    if run is None or run.workspace_id != principal.workspace_id:
        raise HTTPException(status_code=404, detail="run not found")
    actions = {
        "pause": (request_pause, "run.paused"),
        "resume": (request_resume, "run.resumed"),
        "cancel": (request_cancel, "run.cancel_requested"),
        "retry": (request_retry, "run.retry_requested"),
    }
    if action not in actions:
        raise HTTPException(status_code=404, detail="unknown action")
    event_type = actions[action][1]
    try:
        if action == "pause":
            await request_pause(session, run.id, run.workspace_id)
        elif action == "resume":
            await request_resume(session, run.id, run.workspace_id)
        elif action == "cancel":
            await request_cancel(session, run.id, run.workspace_id)
            try:
                from agentforge.runtime.queue import get_redis

                await get_redis().set(f"agentforge:cancel:{run.id}", "1", ex=86_400)
            except Exception:
                pass
        else:
            raw_node_id = form.get("node_id")
            node_id = raw_node_id if isinstance(raw_node_id, str) and raw_node_id else None
            await request_retry(session, run.id, run.workspace_id, node_id=node_id)
        await events.append(session, run.id, event_type, {"node_id": form.get("node_id")})
        await session.commit()
    except InvalidRunState as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RedirectResponse(f"/app/runs/{run.id}", status_code=303)


@router.get("/app/workflows", response_class=HTMLResponse)
async def workflows_page(
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    workflows = list(
        (
            await session.scalars(
                select(WorkflowDefinition).where(WorkflowDefinition.workspace_id == principal.workspace_id)
            )
        ).all()
    )
    versions = list(
        (
            await session.scalars(
                select(WorkflowVersion)
                .where(WorkflowVersion.workflow_id.in_([item.id for item in workflows] or [uuid.uuid4()]))
                .order_by(WorkflowVersion.created_at.desc())
            )
        ).all()
    )
    return templates.TemplateResponse(
        request,
        "workflows.html",
        _context(request, principal=principal, workflows=workflows, versions=versions),
    )


@router.get("/app/agents", response_class=HTMLResponse)
async def agents_page(
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    agents = list(
        (
            await session.scalars(
                select(AgentDefinition)
                .where(AgentDefinition.workspace_id == principal.workspace_id)
                .order_by(AgentDefinition.name)
            )
        ).all()
    )
    return templates.TemplateResponse(request, "agents.html", _context(request, principal=principal, agents=agents))


@router.get("/app/knowledge", response_class=HTMLResponse)
async def knowledge_page(
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    bases = list(
        (
            await session.scalars(
                select(KnowledgeBase)
                .where(KnowledgeBase.workspace_id == principal.workspace_id)
                .order_by(KnowledgeBase.name)
            )
        ).all()
    )
    return templates.TemplateResponse(request, "knowledge.html", _context(request, principal=principal, bases=bases))


@router.post("/app/knowledge")
async def create_knowledge_base(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    session: AsyncSession = Depends(database_session),
    principal: Principal | None = Depends(optional_principal),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    assert principal is not None
    base = KnowledgeBase(workspace_id=principal.workspace_id, name=name, description=description or None)
    session.add(base)
    await session.commit()
    return RedirectResponse("/app/knowledge", status_code=303)


@router.post("/app/knowledge/{knowledge_base_id}/upload")
async def upload_knowledge_document(
    knowledge_base_id: uuid.UUID,
    request: Request,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(database_session),
    principal: Principal | None = Depends(optional_principal),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    assert principal is not None
    base = await session.get(KnowledgeBase, knowledge_base_id)
    if base is None or base.workspace_id != principal.workspace_id:
        raise HTTPException(status_code=404, detail="knowledge base not found")
    await ingest_document(
        session,
        knowledge_base=base,
        filename=file.filename or "upload.txt",
        content=await file.read(),
        content_type=file.content_type,
        artifact_store=LocalArtifactStore(),
        embedding_provider=await resolve_embedding_provider(session, base.workspace_id),
    )
    await session.commit()
    return RedirectResponse("/app/knowledge", status_code=303)


@router.get("/app/memory", response_class=HTMLResponse)
async def memory_page(
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    records = list(
        (
            await session.scalars(
                select(MemoryRecord)
                .where(
                    MemoryRecord.workspace_id == principal.workspace_id,
                    MemoryRecord.deleted_at.is_(None),
                )
                .order_by(MemoryRecord.updated_at.desc())
            )
        ).all()
    )
    return templates.TemplateResponse(request, "memory.html", _context(request, principal=principal, records=records))


@router.post("/app/memory")
async def create_memory_page(
    request: Request,
    content: str = Form(...),
    session: AsyncSession = Depends(database_session),
    principal: Principal | None = Depends(optional_principal),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    assert principal is not None
    embedding_provider = await resolve_embedding_provider(session, principal.workspace_id)
    await MemoryService(embedding_provider).create(session, workspace_id=principal.workspace_id, content=content)
    await session.commit()
    return RedirectResponse("/app/memory", status_code=303)


@router.post("/app/memory/{memory_id}/update")
async def update_memory_page(
    memory_id: uuid.UUID,
    request: Request,
    content: str = Form(...),
    session: AsyncSession = Depends(database_session),
    principal: Principal | None = Depends(optional_principal),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    assert principal is not None
    record = await session.get(MemoryRecord, memory_id)
    if record and record.workspace_id == principal.workspace_id:
        record.content = content
        await session.commit()
    return RedirectResponse("/app/memory", status_code=303)


@router.post("/app/memory/{memory_id}/delete")
async def delete_memory_page(
    memory_id: uuid.UUID,
    request: Request,
    session: AsyncSession = Depends(database_session),
    principal: Principal | None = Depends(optional_principal),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    assert principal is not None
    record = await session.get(MemoryRecord, memory_id)
    if record and record.workspace_id == principal.workspace_id:
        record.deleted_at = datetime.now(UTC)
        await session.commit()
    return RedirectResponse("/app/memory", status_code=303)


@router.get("/app/models", response_class=HTMLResponse)
async def models_page(
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    models = list(
        (await session.scalars(select(ModelProfile).where(ModelProfile.workspace_id == principal.workspace_id))).all()
    )
    return templates.TemplateResponse(request, "models.html", _context(request, principal=principal, models=models))


@router.post("/app/models")
async def create_model_page(
    request: Request,
    name: str = Form(...),
    model: str = Form(...),
    base_url: str = Form(""),
    api_key: str = Form(""),
    provider: str = Form("openai_compatible"),
    kind: str = Form("chat"),
    session: AsyncSession = Depends(database_session),
    principal: Principal | None = Depends(optional_principal),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    assert principal is not None
    from agentforge.models import Secret

    credential_id = None
    if api_key:
        secret = Secret(
            workspace_id=principal.workspace_id,
            name=f"model:{name}",
            encrypted_value=encrypt_secret(api_key),
        )
        session.add(secret)
        await session.flush()
        credential_id = secret.id
    profile = ModelProfile(
        workspace_id=principal.workspace_id,
        name=name,
        kind=kind if kind in {"chat", "embedding"} else "chat",
        provider=provider,
        model=model,
        base_url=base_url or None,
        credential_id=credential_id,
    )
    session.add(profile)
    await session.commit()
    return RedirectResponse("/app/models", status_code=303)


@router.get("/app/mcp", response_class=HTMLResponse)
async def mcp_page(
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    servers = list(
        (await session.scalars(select(MCPServer).where(MCPServer.workspace_id == principal.workspace_id))).all()
    )
    return templates.TemplateResponse(request, "mcp.html", _context(request, principal=principal, servers=servers))


@router.post("/app/mcp")
async def create_mcp_server_page(
    request: Request,
    name: str = Form(...),
    transport: str = Form("streamable_http"),
    endpoint: str = Form(""),
    command: str = Form(""),
    session: AsyncSession = Depends(database_session),
    principal: Principal | None = Depends(optional_principal),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    assert principal is not None
    server = MCPServer(
        workspace_id=principal.workspace_id,
        name=name,
        transport=transport,
        endpoint=endpoint or None,
        command=shlex.split(command) if command else [],
        enabled=True,
    )
    session.add(server)
    await session.commit()
    return RedirectResponse("/app/mcp", status_code=303)


@router.get("/app/skills", response_class=HTMLResponse)
async def skills_page(
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    skills = list(
        (await session.scalars(select(SkillRecord).where(SkillRecord.workspace_id == principal.workspace_id))).all()
    )
    return templates.TemplateResponse(request, "skills.html", _context(request, principal=principal, skills=skills))


@router.get("/app/settings", response_class=HTMLResponse)
async def settings_page(
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    keys = list(
        (
            await session.scalars(
                select(ApiKey).where(ApiKey.workspace_id == principal.workspace_id, ApiKey.revoked_at.is_(None))
            )
        ).all()
    )
    members = list(
        (
            await session.execute(
                select(Membership, User)
                .join(User, Membership.user_id == User.id)
                .where(Membership.workspace_id == principal.workspace_id)
                .order_by(Membership.created_at)
            )
        ).all()
    )
    return templates.TemplateResponse(
        request,
        "settings.html",
        _context(
            request,
            principal=principal,
            api_keys=keys,
            members=members,
            new_api_key=None,
        ),
    )


@router.post("/app/settings/api-keys")
async def create_api_key_page(
    request: Request,
    name: str = Form(...),
    session: AsyncSession = Depends(database_session),
    principal: Principal | None = Depends(optional_principal),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    assert principal is not None
    _key, raw = await create_workspace_key(
        session,
        workspace_id=principal.workspace_id,
        user_id=principal.user_id,
        name=name,
        scopes=["runs:read", "runs:write"],
    )
    await session.commit()
    keys = list(
        (
            await session.scalars(
                select(ApiKey).where(ApiKey.workspace_id == principal.workspace_id, ApiKey.revoked_at.is_(None))
            )
        ).all()
    )
    return templates.TemplateResponse(
        request,
        "settings.html",
        _context(request, principal=principal, api_keys=keys, new_api_key=raw),
    )


@router.post("/app/settings/users")
async def create_workspace_user_page(
    request: Request,
    email: str = Form(...),
    display_name: str = Form(...),
    password: str = Form(...),
    role: str = Form("member"),
    session: AsyncSession = Depends(database_session),
    principal: Principal | None = Depends(optional_principal),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    assert principal is not None
    if principal.role not in {"owner", "admin"}:
        raise HTTPException(status_code=403, detail="insufficient permissions")
    normalized_email = email.strip().lower()
    user = await session.scalar(select(User).where(User.email == normalized_email))
    if user is None:
        user = User(
            email=normalized_email,
            display_name=display_name.strip(),
            password_hash=hash_password(password),
        )
        session.add(user)
        await session.flush()
    membership = await session.scalar(
        select(Membership).where(
            Membership.workspace_id == principal.workspace_id,
            Membership.user_id == user.id,
        )
    )
    if membership is None:
        session.add(
            Membership(
                workspace_id=principal.workspace_id,
                user_id=user.id,
                role=role if role in {"admin", "member", "viewer"} else "member",
            )
        )
    await session.commit()
    return RedirectResponse("/app/settings", status_code=303)


@router.get("/app/recipes", response_class=HTMLResponse)
async def recipes_page(
    request: Request,
    principal: Principal | None = Depends(optional_principal),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    return templates.TemplateResponse(
        request,
        "training/recipes.html",
        _context(
            request,
            principal=principal,
            recipes=get_recipe_registry().list(),
            baselines=build_baseline_registry().list(),
        ),
    )


@router.get("/app/datasets", response_class=HTMLResponse)
async def datasets_page(
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    datasets = list(
        (
            await session.scalars(
                select(Dataset).where(Dataset.workspace_id == principal.workspace_id).order_by(Dataset.name)
            )
        ).all()
    )
    versions = list(
        (
            await session.scalars(
                select(DatasetVersion)
                .where(DatasetVersion.dataset_id.in_([item.id for item in datasets] or [uuid.uuid4()]))
                .order_by(DatasetVersion.created_at.desc())
            )
        ).all()
    )
    return templates.TemplateResponse(
        request,
        "training/datasets.html",
        _context(request, principal=principal, datasets=datasets, versions=versions),
    )


@router.post("/app/datasets")
async def create_dataset_page(
    request: Request,
    name: str = Form(...),
    kind: str = Form("tabular"),
    task_type: str = Form("tabular_classification"),
    description: str = Form(""),
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    assert principal is not None
    await create_dataset(
        session,
        workspace_id=principal.workspace_id,
        name=name,
        kind=DatasetKind(kind),
        task_type=TaskType(task_type),
        description=description or None,
        created_by=principal.user_id,
    )
    await session.commit()
    return RedirectResponse("/app/datasets", status_code=303)


@router.post("/app/datasets/{dataset_id}/versions")
async def upload_dataset_page(
    dataset_id: uuid.UUID,
    request: Request,
    file: UploadFile = File(...),
    target_column: str = Form(""),
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    assert principal is not None
    dataset = await session.get(Dataset, dataset_id)
    if dataset is None or dataset.workspace_id != principal.workspace_id:
        raise HTTPException(status_code=404, detail="dataset not found")
    content = await file.read()
    if len(content) > settings.training_max_dataset_bytes:
        raise HTTPException(status_code=413, detail="dataset exceeds configured size limit")
    await upload_dataset_version(
        session,
        dataset=dataset,
        filename=file.filename or "dataset.bin",
        content=content,
        artifact_store=LocalArtifactStore(),
        target_column=target_column or None,
    )
    await session.commit()
    return RedirectResponse("/app/datasets", status_code=303)


@router.get("/app/experiments", response_class=HTMLResponse)
async def experiments_page(
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    experiments = list(
        (
            await session.scalars(
                select(Experiment)
                .where(Experiment.workspace_id == principal.workspace_id)
                .order_by(Experiment.created_at.desc())
            )
        ).all()
    )
    versions = list(
        (
            await session.scalars(
                select(DatasetVersion)
                .join(Dataset, DatasetVersion.dataset_id == Dataset.id)
                .where(Dataset.workspace_id == principal.workspace_id, DatasetVersion.status == "ready")
                .order_by(DatasetVersion.created_at.desc())
            )
        ).all()
    )
    return templates.TemplateResponse(
        request,
        "training/experiments.html",
        _context(request, principal=principal, experiments=experiments, versions=versions),
    )


@router.post("/app/experiments")
async def create_experiment_page(
    request: Request,
    name: str = Form(...),
    dataset_version_id: uuid.UUID = Form(...),
    metric: str = Form(...),
    direction: str = Form("maximize"),
    minimum_improvement: float = Form(0.0),
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    assert principal is not None
    experiment = await TrainingService().create_experiment(
        session,
        workspace_id=principal.workspace_id,
        dataset_version_id=dataset_version_id,
        name=name,
        baseline_strategy_id=None,
        budget=ExperimentBudget(),
        selection_policy=SelectionPolicy(
            validation_metric=metric,
            direction=direction,
            minimum_improvement=minimum_improvement,
        ),
        created_by=principal.user_id,
    )
    await session.commit()
    return RedirectResponse(f"/app/experiments/{experiment.id}", status_code=303)


@router.get("/app/experiments/{experiment_id}", response_class=HTMLResponse)
async def experiment_detail_page(
    experiment_id: uuid.UUID,
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    experiment = await session.get(Experiment, experiment_id)
    if experiment is None or experiment.workspace_id != principal.workspace_id:
        raise HTTPException(status_code=404, detail="experiment not found")
    jobs = list(
        (
            await session.scalars(
                select(TrainingJob).where(TrainingJob.experiment_id == experiment.id).order_by(TrainingJob.created_at)
            )
        ).all()
    )
    decisions = list(
        (
            await session.scalars(
                select(DecisionRecord)
                .where(DecisionRecord.experiment_id == experiment.id)
                .order_by(DecisionRecord.created_at.desc())
            )
        ).all()
    )
    lineage_edges = list(
        (
            await session.scalars(
                select(LineageEdge)
                .where(LineageEdge.workspace_id == experiment.workspace_id)
                .order_by(LineageEdge.created_at.desc())
                .limit(100)
            )
        ).all()
    )
    lineage_nodes = {
        node.id: node
        for node in (
            await session.scalars(
                select(LineageNode).where(
                    LineageNode.workspace_id == experiment.workspace_id
                )
            )
        ).all()
    }
    return templates.TemplateResponse(
        request,
        "training/experiment_detail.html",
        _context(
            request,
            principal=principal,
            experiment=experiment,
            jobs=jobs,
            leaderboard=await _web_experiment_leaderboard(session, experiment),
            decisions=decisions,
            lineage_edges=lineage_edges,
            lineage_nodes=lineage_nodes,
        ),
    )


@router.post("/app/experiments/{experiment_id}/select-best")
async def select_experiment_best_page(
    experiment_id: uuid.UUID,
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    assert principal is not None
    experiment = await session.get(Experiment, experiment_id)
    if experiment is None or experiment.workspace_id != principal.workspace_id:
        raise HTTPException(status_code=404, detail="experiment not found")
    await TrainingService().select_best(
        session, experiment_id=experiment.id, workspace_id=principal.workspace_id, actor=f"user:{principal.user_id}"
    )
    await session.commit()
    return RedirectResponse(f"/app/experiments/{experiment.id}", status_code=303)


@router.get("/app/training", response_class=HTMLResponse)
async def training_jobs_page(
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    jobs = list(
        (
            await session.scalars(
                select(TrainingJob)
                .where(TrainingJob.workspace_id == principal.workspace_id)
                .order_by(TrainingJob.created_at.desc())
            )
        ).all()
    )
    return templates.TemplateResponse(request, "training/jobs.html", _context(request, principal=principal, jobs=jobs))


@router.get("/app/training/{job_id}", response_class=HTMLResponse)
async def training_job_detail_page(
    job_id: uuid.UUID,
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    job = await session.get(TrainingJob, job_id)
    if job is None or job.workspace_id != principal.workspace_id:
        raise HTTPException(status_code=404, detail="training job not found")
    metrics = list(
        (
            await session.scalars(
                select(MetricPoint).where(MetricPoint.job_id == job.id).order_by(MetricPoint.created_at)
            )
        ).all()
    )
    training_events = list(
        (
            await session.scalars(
                select(TrainingEvent).where(TrainingEvent.job_id == job.id).order_by(TrainingEvent.seq)
            )
        ).all()
    )
    attempt = await session.scalar(
        select(TrainingAttempt)
        .where(TrainingAttempt.job_id == job.id)
        .order_by(TrainingAttempt.attempt.desc())
        .limit(1)
    )
    checkpoints = list(
        (
            await session.scalars(
                select(Checkpoint).where(Checkpoint.job_id == job.id).order_by(Checkpoint.created_at)
            )
        ).all()
    )
    return templates.TemplateResponse(
        request,
        "training/job_detail.html",
        _context(
            request,
            principal=principal,
            job=job,
            metrics=metrics,
            training_events=training_events,
            attempt=attempt,
            checkpoints=checkpoints,
        ),
    )


@router.post("/app/training/{job_id}/cancel")
async def cancel_training_job_page(
    job_id: uuid.UUID,
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    assert principal is not None
    job = await session.get(TrainingJob, job_id)
    if (
        job is not None
        and job.workspace_id == principal.workspace_id
        and job.status
        not in {
            "succeeded",
            "failed",
            "cancelled",
        }
    ):
        job.status = "cancelling"
        await TrainingService().record_cancel_request(
            session,
            job=job,
            actor=f"user:{principal.user_id}",
        )
        await session.commit()
        try:
            from agentforge.runtime.queue import get_redis

            await get_redis().set(f"agentforge:training:cancel:{job.id}", "1", ex=86_400)
        except Exception:
            pass
    return RedirectResponse(f"/app/training/{job_id}", status_code=303)


@router.get("/app/model-registry", response_class=HTMLResponse)
async def model_registry_page(
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    assert principal is not None
    models = list(
        (
            await session.scalars(
                select(ModelVersion)
                .where(ModelVersion.workspace_id == principal.workspace_id)
                .order_by(ModelVersion.created_at.desc())
            )
        ).all()
    )
    return templates.TemplateResponse(
        request, "training/models.html", _context(request, principal=principal, model_versions=models)
    )


@router.post("/app/model-registry/{model_version_id}/promote")
async def promote_model_page(
    model_version_id: uuid.UUID,
    request: Request,
    principal: Principal | None = Depends(optional_principal),
    session: AsyncSession = Depends(database_session),
):
    redirect = _login_redirect(principal)
    if redirect:
        return redirect
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))
    assert principal is not None
    model_version = await session.get(ModelVersion, model_version_id)
    if model_version is None or model_version.workspace_id != principal.workspace_id:
        raise HTTPException(status_code=404, detail="model version not found")
    if principal.role not in {"owner", "admin"}:
        raise HTTPException(status_code=403, detail="insufficient permissions")
    others = await session.scalars(
        select(ModelVersion).where(
            ModelVersion.workspace_id == principal.workspace_id,
            ModelVersion.name == model_version.name,
            ModelVersion.stage == ModelStage.PRODUCTION.value,
        )
    )
    for item in others.all():
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
    return RedirectResponse("/app/model-registry", status_code=303)


def _login_redirect(principal: Principal | None):
    if principal is None:
        return RedirectResponse("/login", status_code=303)
    return None


async def _web_experiment_leaderboard(session: AsyncSession, experiment: Experiment) -> list[dict[str, Any]]:
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
        objective = validation.get(experiment.objective_metric)
        rows.append(
            {
                "job": job,
                "validation": validation,
                "objective": float(objective) if objective is not None else None,
                "selected": job.id == experiment.best_job_id,
            }
        )
    scored = [row for row in rows if row["objective"] is not None]
    unscored = [row for row in rows if row["objective"] is None]
    scored.sort(
        key=lambda row: (
            -float(row["objective"])
            if experiment.objective_direction == "maximize"
            else float(row["objective"]),
            row["job"].actual_total_seconds,
            str(row["job"].id),
        )
    )
    return scored + unscored
