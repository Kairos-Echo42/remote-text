from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class UserRead(ORMModel):
    id: uuid.UUID
    email: str
    display_name: str
    is_active: bool
    is_superuser: bool


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,78}[a-z0-9]$")
    description: str | None = None


class WorkspaceRead(ORMModel):
    id: uuid.UUID
    name: str
    slug: str
    description: str | None
    created_at: datetime


class RunCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    workflow_version_id: uuid.UUID
    input: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunAccepted(BaseModel):
    run_id: uuid.UUID
    status: str
    event_url: str


class RunRead(ORMModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    workflow_version_id: uuid.UUID
    status: str
    input: dict[str, Any]
    output: dict[str, Any] | None
    run_metadata: dict[str, Any]
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    last_event_seq: int


class NodeRunRead(ORMModel):
    id: uuid.UUID
    run_id: uuid.UUID
    node_id: str
    node_type: str
    attempt: int
    status: str
    output: dict[str, Any] | None
    error: str | None
    metrics: dict[str, Any]
    started_at: datetime | None
    finished_at: datetime | None


class RunEventRead(ORMModel):
    id: uuid.UUID
    run_id: uuid.UUID
    node_run_id: uuid.UUID | None
    seq: int
    event_type: str
    payload: dict[str, Any]
    schema_version: int
    created_at: datetime


class ArtifactRead(ORMModel):
    id: uuid.UUID
    name: str
    content_type: str
    size: int
    checksum: str
    created_at: datetime


class WorkflowRead(ORMModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    description: str | None
    latest_version: str | None
    source_hash: str | None
    created_at: datetime


class WorkflowVersionRead(ORMModel):
    id: uuid.UUID
    workflow_id: uuid.UUID
    version: str
    definition: dict[str, Any]
    checksum: str
    created_at: datetime


class ModelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    kind: Literal["chat", "embedding"] = "chat"
    provider: Literal["openai_compatible", "fake"] = "openai_compatible"
    model: str = Field(min_length=1, max_length=160)
    base_url: str | None = None
    api_key: str | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    default_params: dict[str, Any] = Field(default_factory=dict)
    max_context_tokens: int = Field(default=32_000, ge=1_000, le=2_000_000)
    is_default: bool = False


class ModelRead(ORMModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    kind: str
    provider: str
    model: str
    base_url: str | None
    capabilities: dict[str, Any]
    default_params: dict[str, Any]
    max_context_tokens: int
    is_default: bool
    created_at: datetime


class MCPServerCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    transport: Literal["stdio", "streamable_http"] = "streamable_http"
    endpoint: str | None = None
    command: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class MCPServerRead(ORMModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    transport: str
    endpoint: str | None
    command: list[str]
    enabled: bool
    trust_level: str
    tool_cache: list[dict[str, Any]]
    created_at: datetime


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = None
    embedding_model_id: uuid.UUID | None = None
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)


class KnowledgeBaseRead(ORMModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    description: str | None
    embedding_model_id: uuid.UUID | None
    chunk_size: int
    chunk_overlap: int
    created_at: datetime


class MemoryCreate(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)
    agent_name: str | None = None
    scope: Literal["workspace", "agent", "user"] = "workspace"
    kind: Literal["fact", "preference", "summary", "note"] = "fact"
    confidence: float = Field(default=0.8, ge=0, le=1)


class MemoryUpdate(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=20_000)
    agent_name: str | None = None
    scope: Literal["workspace", "agent", "user"] | None = None
    kind: Literal["fact", "preference", "summary", "note"] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class MemoryRead(ORMModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    agent_name: str | None
    scope: str
    kind: str
    content: str
    confidence: float
    source_run_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    scopes: list[str] = Field(default_factory=lambda: ["runs:read", "runs:write"])


class ApiKeyCreated(BaseModel):
    id: uuid.UUID
    name: str
    key: str
    key_prefix: str
    scopes: list[str]


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)
