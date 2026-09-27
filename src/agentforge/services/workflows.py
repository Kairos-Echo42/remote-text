from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentforge.models import AgentDefinition, ModelProfile, WorkflowDefinition, WorkflowVersion
from agentforge.workflow import (
    ValidationIssue,
    WorkflowDocument,
    load_workflow,
    validate_workflow,
)


class WorkflowValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        super().__init__("; ".join(f"{item.path}: {item.message}" for item in issues))


def checksum_document(document: WorkflowDocument) -> str:
    raw = document.to_yaml().encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_or_raise(document: WorkflowDocument) -> None:
    issues = validate_workflow(document)
    if issues:
        raise WorkflowValidationError(issues)


async def sync_workflow(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    document: WorkflowDocument,
    *,
    published_by: uuid.UUID | None = None,
) -> tuple[WorkflowDefinition, WorkflowVersion]:
    validate_or_raise(document)
    definition = await session.scalar(
        select(WorkflowDefinition).where(
            WorkflowDefinition.workspace_id == workspace_id,
            WorkflowDefinition.name == document.metadata.name,
        )
    )
    checksum = checksum_document(document)
    if definition is None:
        definition = WorkflowDefinition(
            workspace_id=workspace_id,
            name=document.metadata.name,
            description=document.metadata.description or "",
            source_hash=checksum,
        )
        session.add(definition)
        await session.flush()
    existing = await session.scalar(
        select(WorkflowVersion).where(
            WorkflowVersion.workflow_id == definition.id,
            WorkflowVersion.version == document.metadata.version,
        )
    )
    if existing is not None:
        if existing.checksum != checksum:
            raise WorkflowValidationError(
                [
                    ValidationIssue(
                        "immutable_version",
                        "metadata.version",
                        f"version {document.metadata.version} already exists with different content",
                    )
                ]
            )
        definition.latest_version = existing.version
        return definition, existing

    version = WorkflowVersion(
        workflow_id=definition.id,
        version=document.metadata.version,
        definition=document.model_dump(by_alias=True, mode="json"),
        checksum=checksum,
        published_by=published_by,
    )
    session.add(version)
    definition.latest_version = version.version
    definition.source_hash = checksum
    definition.description = document.metadata.description or definition.description
    await session.flush()
    return definition, version


async def sync_workflow_path(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    path: str | Path,
    *,
    published_by: uuid.UUID | None = None,
) -> tuple[WorkflowDefinition, WorkflowVersion]:
    return await sync_workflow(session, workspace_id, load_workflow(path), published_by=published_by)


async def sync_agent(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    document: dict,
    *,
    model_profile_id: uuid.UUID | None = None,
) -> AgentDefinition:
    metadata = document["metadata"]
    spec = document["spec"]
    agent = await session.scalar(
        select(AgentDefinition).where(
            AgentDefinition.workspace_id == workspace_id,
            AgentDefinition.name == metadata["name"],
        )
    )
    source_hash = hashlib.sha256(yaml.safe_dump(document, sort_keys=True).encode("utf-8")).hexdigest()
    if agent is None:
        agent = AgentDefinition(
            workspace_id=workspace_id,
            name=metadata["name"],
            version=metadata.get("version", "1.0.0"),
        )
        session.add(agent)
    agent.description = spec.get("description")
    agent.instructions = spec.get("instructions", "")
    agent.capabilities = spec.get("capabilities", [])
    agent.limits = spec.get("limits", {})
    agent.spec = document
    agent.source_hash = source_hash
    if model_profile_id is not None:
        agent.model_profile_id = model_profile_id
    elif spec.get("model"):
        agent.model_profile_id = await session.scalar(
            select(ModelProfile.id).where(
                ModelProfile.workspace_id == workspace_id,
                ModelProfile.name == spec["model"],
            )
        )
    await session.flush()
    return agent


def issues_as_dict(issues: list[ValidationIssue]) -> list[dict[str, str]]:
    return [asdict(item) for item in issues]
