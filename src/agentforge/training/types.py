from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DatasetKind(StrEnum):
    TABULAR = "tabular"
    IMAGE_FOLDER = "image_folder"


class TaskType(StrEnum):
    TABULAR_CLASSIFICATION = "tabular_classification"
    TABULAR_REGRESSION = "tabular_regression"
    IMAGE_CLASSIFICATION = "image_classification"


class DevicePolicy(StrEnum):
    REQUIRED_GPU = "required_gpu"
    PREFERRED_GPU = "preferred_gpu"
    CPU_ONLY = "cpu_only"


class ObjectiveDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class SplitAlgorithm(StrEnum):
    RANDOM = "random"
    STRATIFIED = "stratified"


class DatasetVersionStatus(StrEnum):
    UPLOADING = "uploading"
    VALIDATING = "validating"
    READY = "ready"
    FAILED = "failed"


class ExperimentStatus(StrEnum):
    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    SELECTING = "selecting"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJobStatus(StrEnum):
    QUEUED = "queued"
    RESERVED = "reserved"
    STARTING = "starting"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class TrainingAttemptStatus(StrEnum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CheckpointStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


class ModelStage(StrEnum):
    CANDIDATE = "candidate"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class DecisionResult(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


class ParameterType(StrEnum):
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    STRING = "string"
    CHOICE = "choice"


class HyperparameterSpec(StrictModel):
    name: str
    type: ParameterType
    default: Any = None
    minimum: float | int | None = None
    maximum: float | int | None = None
    choices: list[Any] = Field(default_factory=list)
    nullable: bool = False

    def validate_value(self, value: Any) -> Any:
        if value is None:
            if self.nullable:
                return None
            raise ValueError(f"parameter {self.name!r} does not allow null")
        if self.type == ParameterType.BOOLEAN:
            if not isinstance(value, bool):
                raise ValueError(f"parameter {self.name!r} must be boolean")
            return value
        if self.type == ParameterType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"parameter {self.name!r} must be integer")
        elif self.type == ParameterType.NUMBER:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"parameter {self.name!r} must be numeric")
        elif self.type == ParameterType.STRING:
            if not isinstance(value, str):
                raise ValueError(f"parameter {self.name!r} must be string")
        elif self.type == ParameterType.CHOICE:
            if value not in self.choices:
                raise ValueError(f"parameter {self.name!r} must be one of {self.choices!r}")
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"parameter {self.name!r} must be >= {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"parameter {self.name!r} must be <= {self.maximum}")
        return value


class RecipeResources(StrictModel):
    default_device_policy: DevicePolicy = DevicePolicy.PREFERRED_GPU
    supports_gpu: bool = True
    minimum_memory_mb: int = 512
    recommended_memory_mb: int = 2048
    cpu_cores: int = Field(default=2, ge=1, le=64)
    max_job_seconds: int = Field(default=900, ge=1, le=86_400)
    min_epochs: int = Field(default=1, ge=1, le=10_000)
    max_epochs: int = Field(default=10_000, ge=1, le=10_000)
    min_batch_size: int = Field(default=1, ge=1, le=65_536)
    max_batch_size: int = Field(default=65_536, ge=1, le=65_536)


class RecipeSpec(StrictModel):
    recipe_id: str
    version: str = "1.0.0"
    display_name: str
    description: str
    backend: Literal["sklearn", "pytorch", "dummy"]
    task_types: list[TaskType]
    dataset_kinds: list[DatasetKind]
    default_metric: str
    default_direction: ObjectiveDirection
    hyperparameters: list[HyperparameterSpec] = Field(default_factory=list)
    resources: RecipeResources = Field(default_factory=RecipeResources)
    immutable: bool = True

    @property
    def qualified_id(self) -> str:
        return f"{self.recipe_id}@{self.version}"

    @property
    def checksum(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def validate_hyperparameters(self, values: dict[str, Any]) -> dict[str, Any]:
        specs = {item.name: item for item in self.hyperparameters}
        unknown = sorted(set(values) - set(specs))
        if unknown:
            raise ValueError(f"unknown hyperparameters: {', '.join(unknown)}")
        return {name: spec.validate_value(values.get(name, spec.default)) for name, spec in specs.items()}


class BaselineStrategySpec(StrictModel):
    strategy_id: str
    version: str = "1.0.0"
    display_name: str
    dataset_kind: DatasetKind
    task_type: TaskType
    recipe_id: str
    explain_template: str

    @property
    def qualified_id(self) -> str:
        return f"{self.strategy_id}@{self.version}"

    @property
    def checksum(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class ExperimentBudget(StrictModel):
    max_jobs: int = Field(default=6, ge=1, le=100)
    max_rounds: int = Field(default=2, ge=1, le=10)
    max_job_seconds: int = Field(default=900, ge=1, le=86_400)
    max_total_seconds: int = Field(default=5400, ge=1, le=604_800)
    max_gpu_seconds: int = Field(default=900, ge=0, le=604_800)


class BudgetUsage(StrictModel):
    reserved_total_seconds: float = 0
    consumed_total_seconds: float = 0
    reserved_gpu_seconds: float = 0
    consumed_gpu_seconds: float = 0
    jobs_started: int = 0

    def check_available(
        self,
        budget: ExperimentBudget,
        *,
        total_reservation: float,
        gpu_reservation: float,
    ) -> None:
        if self.jobs_started + 1 > budget.max_jobs:
            raise ValueError("experiment max_jobs budget exceeded")
        if self.consumed_total_seconds + self.reserved_total_seconds + total_reservation > budget.max_total_seconds:
            raise ValueError("experiment max_total_seconds budget exceeded")
        if self.consumed_gpu_seconds + self.reserved_gpu_seconds + gpu_reservation > budget.max_gpu_seconds:
            raise ValueError("experiment max_gpu_seconds budget exceeded")


class SelectionPolicy(StrictModel):
    validation_metric: str
    direction: ObjectiveDirection
    minimum_improvement: float = Field(default=0.0, ge=0)
    top_k: int = Field(default=3, ge=1, le=20)
    tie_break: Literal["lower_job_seconds", "lower_gpu_seconds", "config_checksum"] = "lower_job_seconds"


class ExperimentSearchStrategy(StrictModel):
    strategy_id: Literal["baseline_then_candidates_v1"] = "baseline_then_candidates_v1"
    baseline_strategy_id: str
    round_one_recipes: list[str] = Field(default_factory=list)
    max_round_one_jobs: int = Field(default=3, ge=1, le=50)
    refinement_rounds: int = Field(default=1, ge=0, le=5)
    top_k: int = Field(default=2, ge=1, le=10)


class TrainingJobSpec(StrictModel):
    recipe_id: str
    dataset_version_id: uuid.UUID
    round_number: int = Field(default=1, ge=1, le=20)
    target_column: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    random_seed: int = 42
    max_epochs: int = Field(default=50, ge=1, le=10_000)
    batch_size: int = Field(default=32, ge=1, le=65_536)
    device_policy: DevicePolicy = DevicePolicy.PREFERRED_GPU
    max_job_seconds: int = Field(default=900, ge=1, le=86_400)
    idempotency_key: str | None = None
    evaluation_only: bool = False
    source_job_id: uuid.UUID | None = None
    bundle_artifact_id: uuid.UUID | None = None


class ReproducibilityManifest(StrictModel):
    dataset_version_checksum: str | None = None
    split_checksum: str | None = None
    baseline_strategy: str | None = None
    recipe_id: str | None = None
    recipe_version: str | None = None
    recipe_checksum: str | None = None
    config_checksum: str | None = None
    random_seed: int | None = None
    python_version: str | None = None
    pytorch_version: str | None = None
    sklearn_version: str | None = None
    cuda_version: str | None = None
    cudnn_version: str | None = None
    docker_image_digest: str | None = None
    git_commit: str | None = None
    hardware: dict[str, Any] = Field(default_factory=dict)
    unavailable: list[str] = Field(default_factory=list)


class MetricSample(StrictModel):
    name: str
    value: float
    split: Literal["train", "validation", "test", "system"]
    step: int = Field(default=0, ge=0)
    epoch: int | None = None
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModelBundleManifest(StrictModel):
    files: dict[str, str]
    zip_sha256: str
    recipe: str
    validation_metrics: dict[str, float] = Field(default_factory=dict)
    final_test_metrics: dict[str, float] = Field(default_factory=dict)


class DecisionRecordData(StrictModel):
    action: str = Field(min_length=1)
    result: DecisionResult
    reason_code: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    actor: str = Field(min_length=1)
    phase: str | None = None
    run_id: uuid.UUID | None = None
    node_run_id: uuid.UUID | None = None
    request: dict[str, Any] | None = None
    validation_metrics: dict[str, float] | None = None
    budget_snapshot: dict[str, Any] | None = None
    config_checksum: str | None = None


class DatasetProfile(StrictModel):
    dataset_kind: DatasetKind
    task_type: TaskType
    row_count: int
    column_count: int | None = None
    target_column: str | None = None
    class_count: int | None = None
    missing_ratio: float | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SplitManifest(StrictModel):
    algorithm: SplitAlgorithm
    random_seed: int
    stratification: dict[str, Any] | None = None
    train: list[str]
    validation: list[str]
    test: list[str]

    @property
    def checksum(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def stable_config_checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
