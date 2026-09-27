from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agentforge.db import Base, JSONType, utcnow


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(512))
    display_name: Mapped[str] = mapped_column(String(120))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)

    memberships: Mapped[list[Membership]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Workspace(Base, TimestampMixin):
    __tablename__ = "workspaces"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(120))
    slug: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    settings: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)

    memberships: Mapped[list[Membership]] = relationship(back_populates="workspace", cascade="all, delete-orphan")


class Membership(Base, TimestampMixin):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("user_id", "workspace_id", name="uq_membership_user_workspace"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(20), default="member")

    user: Mapped[User] = relationship(back_populates="memberships")
    workspace: Mapped[Workspace] = relationship(back_populates="memberships")


class ApiKey(Base, TimestampMixin):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    name: Mapped[str] = mapped_column(String(120))
    key_prefix: Mapped[str] = mapped_column(String(16), index=True)
    key_hash: Mapped[str] = mapped_column(String(128), unique=True)
    scopes: Mapped[list[str]] = mapped_column(JSONType, default=list)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Secret(Base, TimestampMixin):
    __tablename__ = "secrets"
    __table_args__ = (UniqueConstraint("workspace_id", "name", name="uq_secret_workspace_name"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    encrypted_value: Mapped[str] = mapped_column(Text)


class ModelProfile(Base, TimestampMixin):
    __tablename__ = "model_profiles"
    __table_args__ = (UniqueConstraint("workspace_id", "name", name="uq_model_workspace_name"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(80))
    kind: Mapped[str] = mapped_column(String(20), default="chat", index=True)
    provider: Mapped[str] = mapped_column(String(40), default="openai_compatible")
    model: Mapped[str] = mapped_column(String(160))
    base_url: Mapped[str | None] = mapped_column(String(500))
    credential_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("secrets.id", ondelete="SET NULL"))
    capabilities: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    default_params: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    max_context_tokens: Mapped[int] = mapped_column(Integer, default=32_000)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)

    credential: Mapped[Secret | None] = relationship()


class AgentDefinition(Base, TimestampMixin):
    __tablename__ = "agent_definitions"
    __table_args__ = (UniqueConstraint("workspace_id", "name", name="uq_agent_workspace_name"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    version: Mapped[str] = mapped_column(String(60), default="1.0.0")
    description: Mapped[str | None] = mapped_column(Text)
    model_profile_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model_profiles.id", ondelete="SET NULL"))
    instructions: Mapped[str] = mapped_column(Text, default="")
    capabilities: Mapped[list[str]] = mapped_column(JSONType, default=list)
    limits: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    spec: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    source_hash: Mapped[str | None] = mapped_column(String(64))


class WorkflowDefinition(Base, TimestampMixin):
    __tablename__ = "workflow_definitions"
    __table_args__ = (UniqueConstraint("workspace_id", "name", name="uq_workflow_workspace_name"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str | None] = mapped_column(Text)
    latest_version: Mapped[str | None] = mapped_column(String(60))
    source_hash: Mapped[str | None] = mapped_column(String(64))

    versions: Mapped[list[WorkflowVersion]] = relationship(back_populates="workflow", cascade="all, delete-orphan")


class WorkflowVersion(Base, TimestampMixin):
    __tablename__ = "workflow_versions"
    __table_args__ = (UniqueConstraint("workflow_id", "version", name="uq_workflow_version"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workflow_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workflow_definitions.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[str] = mapped_column(String(60))
    definition: Mapped[dict[str, Any]] = mapped_column(JSONType)
    checksum: Mapped[str] = mapped_column(String(64))
    published_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    workflow: Mapped[WorkflowDefinition] = relationship(back_populates="versions")


class MCPServer(Base, TimestampMixin):
    __tablename__ = "mcp_servers"
    __table_args__ = (UniqueConstraint("workspace_id", "name", name="uq_mcp_workspace_name"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    transport: Mapped[str] = mapped_column(String(30), default="streamable_http")
    endpoint: Mapped[str | None] = mapped_column(String(1000))
    command: Mapped[list[str]] = mapped_column(JSONType, default=list)
    env_secret_ids: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    trust_level: Mapped[str] = mapped_column(String(20), default="admin")
    tool_cache: Mapped[list[dict[str, Any]]] = mapped_column(JSONType, default=list)


class SkillRecord(Base, TimestampMixin):
    __tablename__ = "skills"
    __table_args__ = (UniqueConstraint("workspace_id", "name", name="uq_skill_workspace_name"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    version: Mapped[str] = mapped_column(String(60), default="1.0.0")
    description: Mapped[str | None] = mapped_column(Text)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    entry_point: Mapped[str | None] = mapped_column(String(500))
    trusted: Mapped[bool] = mapped_column(Boolean, default=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class KnowledgeBase(Base, TimestampMixin):
    __tablename__ = "knowledge_bases"
    __table_args__ = (UniqueConstraint("workspace_id", "name", name="uq_kb_workspace_name"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str | None] = mapped_column(Text)
    embedding_model_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model_profiles.id", ondelete="SET NULL"))
    chunk_size: Mapped[int] = mapped_column(Integer, default=1000)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=150)


class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(500))
    content_type: Mapped[str | None] = mapped_column(String(200))
    storage_key: Mapped[str] = mapped_column(String(1000))
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(30), default="queued")
    error: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict)


class Chunk(Base, TimestampMixin):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024).with_variant(JSON(), "sqlite"), nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict)


Index("ix_chunks_fts", Chunk.content)


class MemoryRecord(Base, TimestampMixin):
    __tablename__ = "memory_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    agent_name: Mapped[str | None] = mapped_column(String(120), index=True)
    scope: Mapped[str] = mapped_column(String(30), default="workspace")
    kind: Mapped[str] = mapped_column(String(30), default="fact")
    content: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.8)
    source_run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("runs.id", ondelete="SET NULL"))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024).with_variant(JSON(), "sqlite"), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Run(Base, TimestampMixin):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    workflow_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workflow_versions.id", ondelete="RESTRICT"), index=True
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    input: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    output: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    run_metadata: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_event_seq: Mapped[int] = mapped_column(BigInteger, default=0)


class NodeRun(Base, TimestampMixin):
    __tablename__ = "node_runs"
    __table_args__ = (UniqueConstraint("run_id", "node_id", "attempt", name="uq_node_run_attempt"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    node_id: Mapped[str] = mapped_column(String(120), index=True)
    node_type: Mapped[str] = mapped_column(String(30))
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_owner: Mapped[str | None] = mapped_column(String(200), index=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    output: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    error: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class RunEvent(Base):
    __tablename__ = "run_events"
    __table_args__ = (UniqueConstraint("run_id", "seq", name="uq_run_event_seq"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    node_run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("node_runs.id", ondelete="SET NULL"), index=True)
    seq: Mapped[int] = mapped_column(BigInteger)
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Artifact(Base, TimestampMixin):
    __tablename__ = "artifacts"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    node_run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("node_runs.id", ondelete="SET NULL"), index=True)
    name: Mapped[str] = mapped_column(String(500))
    content_type: Mapped[str] = mapped_column(String(200), default="application/octet-stream")
    size: Mapped[int] = mapped_column(BigInteger, default=0)
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    storage_key: Mapped[str] = mapped_column(String(1000), unique=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict)


class OutboxEvent(Base):
    __tablename__ = "outbox_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    topic: Mapped[str] = mapped_column(String(100), index=True)
    aggregate_id: Mapped[uuid.UUID] = mapped_column(Uuid, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
