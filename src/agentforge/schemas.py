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


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    kind: Literal["tabular", "image_folder"]
    task_type: Literal[
        "tabular_classification",
        "tabular_regression",
        "image_classification",
    ]
    description: str | None = None


class DatasetRead(ORMModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    description: str | None
    kind: str
    task_type: str
    created_at: datetime


class DatasetVersionRead(ORMModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    version: str
    status: str
    checksum: str
    size: int
    row_count: int | None
    column_count: int | None
    target_column: str | None
    profile: dict[str, Any]
    split_manifest: dict[str, Any]
    split_checksum: str | None
    split_algorithm: str | None
    split_seed: int | None
    stratification: dict[str, Any] | None
    error: str | None
    created_at: datetime


class BaselineStrategyRead(ORMModel):
    strategy_id: str
    version: str
    display_name: str
    dataset_kind: str
    task_type: str
    recipe_id: str
    explain_template: str
    checksum: str


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    dataset_version_id: uuid.UUID
    baseline_strategy_id: str | None = None
    budget: dict[str, Any] = Field(default_factory=dict)
    search_strategy: dict[str, Any] | None = None
    selection_policy: dict[str, Any] | None = None


class ExperimentRead(ORMModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    dataset_version_id: uuid.UUID
    name: str
    status: str
    objective_metric: str
    objective_direction: str
    baseline_strategy: dict[str, Any]
    search_strategy: dict[str, Any]
    budget: dict[str, Any]
    selection_policy: dict[str, Any]
    reserved_total_seconds: float
    consumed_total_seconds: float
    reserved_gpu_seconds: float
    consumed_gpu_seconds: float
    jobs_started: int
    current_round: int
    baseline_job_id: uuid.UUID | None
    best_job_id: uuid.UUID | None
    selection_reason: str | None
    final_test_metrics: dict[str, Any] | None
    error: str | None
    created_at: datetime
    finished_at: datetime | None


class BudgetReservationRead(ORMModel):
    id: uuid.UUID
    experiment_id: uuid.UUID
    job_id: uuid.UUID
    total_seconds: float
    gpu_seconds: float
    consumed_total_seconds: float
    consumed_gpu_seconds: float
    status: str
    released_at: datetime | None


class ExperimentLeaderboardItem(BaseModel):
    job_id: uuid.UUID
    job_kind: str
    recipe_id: str
    round_number: int
    validation_metrics: dict[str, float]
    objective_value: float | None
    actual_total_seconds: float
    actual_gpu_seconds: float
    selected: bool


class ExperimentReport(BaseModel):
    experiment: ExperimentRead
    baseline_status: str | None
    budget_ledger: dict[str, Any]
    validation_leaderboard: list[ExperimentLeaderboardItem]
    selection_reason: str | None
    selection_policy: dict[str, Any]
    final_test_metrics: dict[str, Any] | None
    final_test_visibility: str


class TrainingJobCreate(BaseModel):
    experiment_id: uuid.UUID
    jobs: list[dict[str, Any]]
    round_number: int = Field(default=1, ge=1, le=20)


class TrainingJobRead(ORMModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    experiment_id: uuid.UUID
    dataset_version_id: uuid.UUID
    recipe_id: str
    recipe_version: str
    round_number: int
    job_kind: str
    status: str
    device_policy: str
    config_checksum: str
    reserved_total_seconds: float
    reserved_gpu_seconds: float
    actual_total_seconds: float
    actual_gpu_seconds: float
    fallback_reason: str | None
    image_digest: str | None
    queue_name: str
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class MetricPointRead(ORMModel):
    id: uuid.UUID
    job_id: uuid.UUID
    name: str
    value: float
    split: str
    step: int
    epoch: int | None
    created_at: datetime


class TrainingEventRead(ORMModel):
    id: uuid.UUID
    job_id: uuid.UUID
    attempt_id: uuid.UUID | None
    seq: int
    event_type: str
    payload: dict[str, Any]
    created_at: datetime


class DecisionRecordRead(ORMModel):
    id: uuid.UUID
    experiment_id: uuid.UUID | None
    job_id: uuid.UUID | None
    phase: str | None
    actor: str
    action: str
    result: str
    reason_code: str
    reason: str
    request: dict[str, Any] | None
    validation_metrics: dict[str, Any] | None
    budget_snapshot: dict[str, Any] | None
    config_checksum: str | None
    created_at: datetime


class ModelVersionRead(ORMModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    experiment_id: uuid.UUID
    job_id: uuid.UUID
    name: str
    version: str
    stage: str
    bundle_artifact_id: uuid.UUID
    bundle_sha256: str
    recipe_id: str
    validation_metrics: dict[str, Any]
    final_test_metrics: dict[str, Any] | None
    selection_policy: dict[str, Any]
    reproducibility: dict[str, Any]
    created_at: datetime
    promoted_at: datetime | None
    archived_at: datetime | None


class CheckpointRead(ORMModel):
    id: uuid.UUID
    job_id: uuid.UUID
    attempt_id: uuid.UUID
    artifact_id: uuid.UUID
    epoch: int | None
    step: int | None
    validation_metric: str | None
    validation_value: float | None
    checksum: str
    status: str
    metadata_: dict[str, Any]
    created_at: datetime
