from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from agentforge.workflow.dsl import NodeSpec, WorkflowDocument


@dataclass(slots=True, frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str


def _schema_issues(schema: dict[str, Any], path: str) -> list[ValidationIssue]:
    if not schema:
        return []
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:  # jsonschema raises several schema-specific errors
        return [ValidationIssue("invalid_schema", path, str(exc))]
    return []


def validate_workflow(document: WorkflowDocument) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    issues.extend(_schema_issues(document.spec.input_schema, "spec.inputSchema"))
    issues.extend(_schema_issues(document.spec.output_schema, "spec.outputSchema"))

    node_map: dict[str, NodeSpec] = {}
    for index, node in enumerate(document.spec.nodes):
        path = f"spec.nodes[{index}]"
        if node.id in node_map:
            issues.append(ValidationIssue("duplicate_node", f"{path}.id", node.id))
            continue
        node_map[node.id] = node

    dependencies: dict[str, set[str]] = {node_id: set(node.needs) for node_id, node in node_map.items()}
    for index, edge in enumerate(document.spec.edges):
        path = f"spec.edges[{index}]"
        if edge.source not in node_map:
            issues.append(ValidationIssue("unknown_edge_source", f"{path}.source", edge.source))
        if edge.target not in node_map:
            issues.append(ValidationIssue("unknown_edge_target", f"{path}.target", edge.target))
        if edge.source in node_map and edge.target in node_map:
            dependencies[edge.target].add(edge.source)

    for node_id, needs in dependencies.items():
        for dependency in needs:
            if dependency not in node_map:
                issues.append(
                    ValidationIssue(
                        "unknown_dependency",
                        f"spec.nodes[{node_id}].needs",
                        f"node {node_id!r} depends on missing node {dependency!r}",
                    )
                )
            if dependency == node_id:
                issues.append(ValidationIssue("self_dependency", f"spec.nodes[{node_id}].needs", node_id))

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str, path: list[str]) -> None:
        if node_id in visiting:
            cycle = " -> ".join([*path, node_id])
            issues.append(ValidationIssue("cycle", "spec.nodes", cycle))
            return
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in dependencies.get(node_id, set()):
            if dependency in node_map:
                visit(dependency, [*path, node_id])
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in node_map:
        visit(node_id, [])

    for index, node in enumerate(document.spec.nodes):
        path = f"spec.nodes[{index}]"
        if node.type == "agent" and not node.uses.strip():
            issues.append(ValidationIssue("missing_agent", f"{path}.uses", "agent reference is empty"))
        if node.type == "capability" and "." not in node.uses and ":" not in node.uses:
            issues.append(
                ValidationIssue(
                    "invalid_capability_name",
                    f"{path}.uses",
                    "capability names must use a namespace such as sandbox.execute",
                )
            )

    output_refs = _walk_values(document.spec.output)
    for value in output_refs:
        if isinstance(value, str) and value.startswith("$."):
            root = value.split(".", 2)[1] if "." in value else ""
            if root not in {"input", "nodes", "run"}:
                issues.append(ValidationIssue("invalid_output_mapping", "spec.output", f"invalid ref {value!r}"))
    return issues


def resolve_dependencies(document: WorkflowDocument) -> dict[str, set[str]]:
    dependencies: dict[str, set[str]] = defaultdict(set)
    for node in document.spec.nodes:
        dependencies[node.id].update(node.needs)
    for edge in document.spec.edges:
        dependencies[edge.target].add(edge.source)
    return dict(dependencies)


def _walk_values(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value
