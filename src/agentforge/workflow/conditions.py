from __future__ import annotations

from copy import deepcopy
from typing import Any

import jmespath
from json_logic import jsonLogic


class MappingError(ValueError):
    pass


def resolve_mapping(mapping: Any, context: dict[str, Any]) -> Any:
    """Resolve JMESPath strings while preserving non-string literal values."""
    if isinstance(mapping, str):
        if mapping.startswith("$"):
            expression = mapping[2:] if mapping.startswith("$.") else mapping[1:]
            try:
                return jmespath.search(expression, context)
            except Exception as exc:
                raise MappingError(f"invalid mapping {mapping!r}: {exc}") from exc
        return deepcopy(mapping)
    if isinstance(mapping, dict):
        return {key: resolve_mapping(value, context) for key, value in mapping.items()}
    if isinstance(mapping, list):
        return [resolve_mapping(value, context) for value in mapping]
    return deepcopy(mapping)


def evaluate_condition(condition: dict[str, Any] | bool | None, context: dict[str, Any]) -> bool:
    if condition is None:
        return True
    if isinstance(condition, bool):
        return condition
    try:
        return bool(jsonLogic(condition, context))
    except Exception as exc:
        raise MappingError(f"invalid JSON Logic condition: {exc}") from exc


def build_evaluation_context(
    *,
    run_input: dict[str, Any],
    node_outputs: dict[str, dict[str, Any]],
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "input": deepcopy(run_input),
        "nodes": {
            node_id: {"output": deepcopy(output), **deepcopy(output)} for node_id, output in node_outputs.items()
        },
        "run": deepcopy(run_metadata or {}),
    }
