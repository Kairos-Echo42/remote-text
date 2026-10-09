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


class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("workspace_id", "name", name="uq_dataset_workspace_name"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    description: Mapped[str | None] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(40), index=True)
    task_type: Mapped[str] = mapped_column(String(60), index=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))


class DatasetVersion(Base, TimestampMixin):
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    version: Mapped[str] = mapped_column(String(60))
    status: Mapped[str] = mapped_column(String(30), default="uploading", index=True)
    storage_key: Mapped[str] = mapped_column(String(1000))
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    size: Mapped[int] = mapped_column(BigInteger, default=0)
    row_count: Mapped[int | None] = mapped_column(Integer)
    column_count: Mapped[int | None] = mapped_column(Integer)
    target_column: Mapped[str | None] = mapped_column(String(300))
    profile: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    split_manifest: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    split_checksum: Mapped[str | None] = mapped_column(String(64), index=True)
    split_algorithm: Mapped[str | None] = mapped_column(String(30))
    split_seed: Mapped[int | None] = mapped_column(Integer)
    stratification: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    error: Mapped[str | None] = mapped_column(Text)


class Experiment(Base, TimestampMixin):
    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="RESTRICT"), index=True
    )
    name: Mapped[str] = mapped_column(String(200))
    objective_metric: Mapped[str] = mapped_column(String(100))
    objective_direction: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(30), default="draft", index=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("runs.id", ondelete="SET NULL"))
    baseline_strategy: Mapped[dict[str, Any]] = mapped_column(JSONType)
    search_strategy: Mapped[dict[str, Any]] = mapped_column(JSONType)
    budget: Mapped[dict[str, Any]] = mapped_column(JSONType)
    selection_policy: Mapped[dict[str, Any]] = mapped_column(JSONType)
    reserved_total_seconds: Mapped[float] = mapped_column(Float, default=0)
    consumed_total_seconds: Mapped[float] = mapped_column(Float, default=0)
    reserved_gpu_seconds: Mapped[float] = mapped_column(Float, default=0)
    consumed_gpu_seconds: Mapped[float] = mapped_column(Float, default=0)
    jobs_started: Mapped[int] = mapped_column(Integer, default=0)
    current_round: Mapped[int] = mapped_column(Integer, default=1)
    baseline_job_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, index=True)
    best_job_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, index=True)
    selection_reason: Mapped[str | None] = mapped_column(Text)
    final_test_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TrainingJob(Base, TimestampMixin):
    __tablename__ = "training_jobs"
    __table_args__ = (UniqueConstraint("experiment_id", "idempotency_key", name="uq_training_job_idempotency"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    experiment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="RESTRICT"), index=True
    )
    recipe_id: Mapped[str] = mapped_column(String(220), index=True)
    recipe_version: Mapped[str] = mapped_column(String(60), default="1.0.0")
    recipe_checksum: Mapped[str] = mapped_column(String(64))
    round_number: Mapped[int] = mapped_column(Integer, default=1)
    job_kind: Mapped[str] = mapped_column(String(30), default="candidate", index=True)
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    device_policy: Mapped[str] = mapped_column(String(30), default="preferred_gpu", index=True)
    idempotency_key: Mapped[str] = mapped_column(String(200))
    spec: Mapped[dict[str, Any]] = mapped_column(JSONType)
    config_checksum: Mapped[str] = mapped_column(String(64), index=True)
    reserved_total_seconds: Mapped[float] = mapped_column(Float, default=0)
    reserved_gpu_seconds: Mapped[float] = mapped_column(Float, default=0)
    actual_total_seconds: Mapped[float] = mapped_column(Float, default=0)
    actual_gpu_seconds: Mapped[float] = mapped_column(Float, default=0)
    fallback_reason: Mapped[str | None] = mapped_column(Text)
    container_id: Mapped[str | None] = mapped_column(String(200), index=True)
    image_digest: Mapped[str | None] = mapped_column(String(300))
    queue_name: Mapped[str] = mapped_column(String(30), default="gpu", index=True)
    error: Mapped[str | None] = mapped_column(Text)
    submitted_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TrainingAttempt(Base, TimestampMixin):
    __tablename__ = "training_attempts"
    __table_args__ = (UniqueConstraint("job_id", "attempt", name="uq_training_attempt"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("training_jobs.id", ondelete="CASCADE"), index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(30), default="created", index=True)
    container_id: Mapped[str | None] = mapped_column(String(200), index=True)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TrainingEvent(Base):
    __tablename__ = "training_events"
    __table_args__ = (UniqueConstraint("job_id", "seq", name="uq_training_event_seq"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("training_jobs.id", ondelete="CASCADE"), index=True)
    attempt_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("training_attempts.id", ondelete="SET NULL"), index=True
    )
    seq: Mapped[int] = mapped_column(BigInteger)
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MetricPoint(Base):
    __tablename__ = "metric_points"
    __table_args__ = (Index("ix_metric_points_job_name_step", "job_id", "name", "step"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("training_jobs.id", ondelete="CASCADE"), index=True)
    attempt_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("training_attempts.id", ondelete="SET NULL"), index=True
    )
    name: Mapped[str] = mapped_column(String(120), index=True)
    value: Mapped[float] = mapped_column(Float)
    split: Mapped[str] = mapped_column(String(30), index=True)
    step: Mapped[int] = mapped_column(Integer, default=0)
    epoch: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Checkpoint(Base, TimestampMixin):
    __tablename__ = "checkpoints"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("training_jobs.id", ondelete="CASCADE"), index=True)
    attempt_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("training_attempts.id", ondelete="CASCADE"), index=True)
    artifact_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("artifacts.id", ondelete="RESTRICT"), index=True)
    epoch: Mapped[int | None] = mapped_column(Integer)
    step: Mapped[int | None] = mapped_column(Integer)
    validation_metric: Mapped[str | None] = mapped_column(String(120))
    validation_value: Mapped[float | None] = mapped_column(Float)
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(20), default="valid", index=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict)


class ModelVersion(Base, TimestampMixin):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("workspace_id", "name", "version", name="uq_model_version"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    experiment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("training_jobs.id", ondelete="RESTRICT"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(60))
    stage: Mapped[str] = mapped_column(String(30), default="candidate", index=True)
    bundle_artifact_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("artifacts.id", ondelete="RESTRICT"), index=True)
    bundle_sha256: Mapped[str] = mapped_column(String(64), index=True)
    recipe_id: Mapped[str] = mapped_column(String(220))
    validation_metrics: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    final_test_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    selection_policy: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    reproducibility: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict)
    promoted_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LineageNode(Base, TimestampMixin):
    __tablename__ = "lineage_nodes"
    __table_args__ = (UniqueConstraint("workspace_id", "node_type", "ref_id", name="uq_lineage_node"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    node_type: Mapped[str] = mapped_column(String(40), index=True)
    ref_id: Mapped[uuid.UUID] = mapped_column(Uuid, index=True)
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("artifacts.id", ondelete="SET NULL"), index=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict)


class LineageEdge(Base):
    __tablename__ = "lineage_edges"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    input_node_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("lineage_nodes.id", ondelete="CASCADE"), index=True)
    output_node_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("lineage_nodes.id", ondelete="CASCADE"), index=True)
    operation: Mapped[str] = mapped_column(String(80), index=True)
    actor_type: Mapped[str] = mapped_column(String(40))
    actor_id: Mapped[str | None] = mapped_column(String(200))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class DecisionRecord(Base):
    __tablename__ = "decision_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    experiment_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="SET NULL"), index=True
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("training_jobs.id", ondelete="SET NULL"), index=True)
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("runs.id", ondelete="SET NULL"))
    node_run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("node_runs.id", ondelete="SET NULL"))
    phase: Mapped[str | None] = mapped_column(String(60))
    actor: Mapped[str] = mapped_column(String(120))
    action: Mapped[str] = mapped_column(String(160), index=True)
    result: Mapped[str] = mapped_column(String(30), index=True)
    reason_code: Mapped[str] = mapped_column(String(120), index=True)
    reason: Mapped[str] = mapped_column(Text)
    request: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    validation_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    budget_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    config_checksum: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class BudgetReservation(Base, TimestampMixin):
    __tablename__ = "budget_reservations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("training_jobs.id", ondelete="CASCADE"), unique=True, index=True
    )
    total_seconds: Mapped[float] = mapped_column(Float)
    gpu_seconds: Mapped[float] = mapped_column(Float, default=0)
    consumed_total_seconds: Mapped[float] = mapped_column(Float, default=0)
    consumed_gpu_seconds: Mapped[float] = mapped_column(Float, default=0)
    status: Mapped[str] = mapped_column(String(30), default="active", index=True)
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
