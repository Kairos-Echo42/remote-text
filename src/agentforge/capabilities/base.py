from __future__ import annotations

import inspect
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from jsonschema import Draft202012Validator, ValidationError

from agentforge.providers.base import ToolDefinition
from agentforge.storage.artifacts import ArtifactStore


class CapabilityError(RuntimeError):
    pass


class SideEffect(StrEnum):
    NONE = "none"
    WORKSPACE = "workspace"
    EXTERNAL = "external"


@dataclass(slots=True)
class CapabilityResult:
    output: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class CapabilityContext:
    workspace_id: uuid.UUID
    run_id: uuid.UUID
    node_run_id: uuid.UUID
    actor_id: uuid.UUID | None = None
    services: dict[str, Any] = field(default_factory=dict)
    artifact_store: ArtifactStore | None = None
    idempotency_key: str | None = None

    def service(self, name: str) -> Any:
        try:
            return self.services[name]
        except KeyError as exc:
            raise CapabilityError(f"service {name!r} is not available") from exc


class Capability(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]
    side_effect: SideEffect
    idempotent: bool

    async def invoke(self, context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult: ...


CapabilityFunction = Callable[[CapabilityContext, dict[str, Any]], Awaitable[CapabilityResult]]


class FunctionCapability:
    def __init__(
        self,
        name: str,
        description: str,
        function: CapabilityFunction,
        *,
        input_schema: dict[str, Any] | None = None,
        side_effect: SideEffect = SideEffect.NONE,
        idempotent: bool = True,
    ):
        self.name = name
        self.description = description
        self.function = function
        self.input_schema = input_schema or {"type": "object", "additionalProperties": True}
        self.side_effect = side_effect
        self.idempotent = idempotent

    async def invoke(self, context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
        result = self.function(context, arguments)
        if inspect.isawaitable(result):
            return await result
        return result


class CapabilityRegistry:
    def __init__(self):
        self._capabilities: dict[str, Capability] = {}

    def register(self, capability: Capability) -> None:
        if capability.name in self._capabilities:
            raise CapabilityError(f"capability {capability.name!r} is already registered")
        self._capabilities[capability.name] = capability

    def unregister(self, name: str) -> None:
        self._capabilities.pop(name, None)

    def get(self, name: str) -> Capability:
        try:
            return self._capabilities[name]
        except KeyError as exc:
            raise CapabilityError(f"capability {name!r} is not registered") from exc

    def all(self) -> list[Capability]:
        return list(self._capabilities.values())

    @staticmethod
    def external_name(name: str) -> str:
        """Return a provider-safe function name while retaining the internal namespace."""
        return re.sub(r"[^a-zA-Z0-9_-]", "__", name)

    def resolve_external_name(self, name: str) -> str:
        if name in self._capabilities:
            return name
        for capability_name in self._capabilities:
            if self.external_name(capability_name) == name:
                return capability_name
        raise CapabilityError(f"capability {name!r} is not registered")

    def openai_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=self.external_name(capability.name),
                description=capability.description,
                input_schema=capability.input_schema,
            )
            for capability in self._capabilities.values()
        ]

    async def invoke(self, name: str, context: CapabilityContext, arguments: dict[str, Any]) -> CapabilityResult:
        capability = self.get(name)
        try:
            Draft202012Validator(capability.input_schema).validate(arguments)
        except ValidationError as exc:
            raise CapabilityError(f"invalid arguments for {name}: {exc.message}") from exc
        return await capability.invoke(context, arguments)
