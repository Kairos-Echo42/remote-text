from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Metadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    version: str = Field(default="1.0.0", max_length=60)
    description: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class RetryPolicy(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    max_attempts: int = Field(default=1, ge=1, le=20, alias="maxAttempts")
    base_delay_seconds: float = Field(default=1.0, ge=0, alias="baseDelaySeconds")
    max_delay_seconds: float = Field(default=60.0, ge=0, alias="maxDelaySeconds")


class NodeSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,119}$")
    type: Literal["agent", "capability"]
    uses: str = Field(min_length=1, max_length=200)
    needs: list[str] = Field(default_factory=list)
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")
    when: dict[str, Any] | bool | None = None
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    timeout_seconds: int = Field(default=300, ge=1, le=86_400, alias="timeoutSeconds")
    capabilities: list[str] | None = None
    continue_on_error: bool = Field(default=False, alias="continueOnError")

    @model_validator(mode="after")
    def validate_type_fields(self) -> Self:
        if self.type == "capability" and self.capabilities:
            raise ValueError("capability nodes cannot declare an Agent capability override")
        return self


class EdgeSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    source: str
    target: str
    when: dict[str, Any] | bool | None = None


class WorkflowSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    input_schema: dict[str, Any] = Field(default_factory=dict, alias="inputSchema")
    output_schema: dict[str, Any] = Field(default_factory=dict, alias="outputSchema")
    max_concurrency: int = Field(default=4, ge=1, le=64, alias="maxConcurrency")
    nodes: list[NodeSpec]
    edges: list[EdgeSpec] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)


class WorkflowDocument(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    api_version: Literal["agentforge/v1"] = Field(default="agentforge/v1", alias="apiVersion")
    kind: Literal["Workflow"] = "Workflow"
    metadata: Metadata
    spec: WorkflowSpec

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(by_alias=True, mode="json"),
            sort_keys=False,
            allow_unicode=True,
        )


class AgentSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    description: str | None = None
    model: str | None = None
    instructions: str = ""
    capabilities: list[str] = Field(default_factory=list)
    limits: dict[str, int | float | bool] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


class AgentDocument(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    api_version: Literal["agentforge/v1"] = Field(default="agentforge/v1", alias="apiVersion")
    kind: Literal["Agent"] = "Agent"
    metadata: Metadata
    spec: AgentSpec


def load_workflow(path: str | Path) -> WorkflowDocument:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return WorkflowDocument.model_validate(payload)


def load_agent(path: str | Path) -> AgentDocument:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AgentDocument.model_validate(payload)


class WorkflowBuilder:
    """Small Python DSL that serializes to the same contract as YAML."""

    def __init__(self, name: str, version: str = "1.0.0", description: str | None = None):
        self._metadata = Metadata(name=name, version=version, description=description)
        self._nodes: list[NodeSpec] = []
        self._edges: list[EdgeSpec] = []
        self._input_schema: dict[str, Any] = {}
        self._output_schema: dict[str, Any] = {}
        self._output: dict[str, Any] = {}
        self._max_concurrency = 4

    def input_schema(self, schema: dict[str, Any]) -> Self:
        self._input_schema = schema
        return self

    def output_schema(self, schema: dict[str, Any]) -> Self:
        self._output_schema = schema
        return self

    def max_concurrency(self, value: int) -> Self:
        self._max_concurrency = value
        return self

    def node(
        self,
        node_id: str,
        node_type: Literal["agent", "capability"],
        uses: str,
        *,
        needs: list[str] | None = None,
        inputs: dict[str, Any] | None = None,
        when: dict[str, Any] | bool | None = None,
        timeout_seconds: int = 300,
        max_attempts: int = 1,
        capabilities: list[str] | None = None,
    ) -> Self:
        self._nodes.append(
            NodeSpec(
                id=node_id,
                type=node_type,
                uses=uses,
                needs=needs or [],
                **{
                    "with": inputs or {},
                    "when": when,
                    "timeoutSeconds": timeout_seconds,
                    "retry": RetryPolicy(max_attempts=max_attempts),
                    "capabilities": capabilities,
                },
            )
        )
        return self

    def edge(self, source: str, target: str, when: dict[str, Any] | bool | None = None) -> Self:
        self._edges.append(EdgeSpec(source=source, target=target, when=when))
        return self

    def output(self, mapping: dict[str, Any]) -> Self:
        self._output = mapping
        return self

    def build(self) -> WorkflowDocument:
        return WorkflowDocument(
            metadata=self._metadata,
            spec=WorkflowSpec(
                input_schema=self._input_schema,
                output_schema=self._output_schema,
                max_concurrency=self._max_concurrency,
                nodes=self._nodes,
                edges=self._edges,
                output=self._output,
            ),
        )
