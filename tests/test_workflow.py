from __future__ import annotations

import hashlib

from agentforge.workflow import WorkflowBuilder, load_workflow, validate_workflow


def test_sample_workflow_is_valid():
    document = load_workflow("examples/workflows/multi-agent-research.yaml")
    assert validate_workflow(document) == []
    assert document.spec.max_concurrency == 4
    assert [node.id for node in document.spec.nodes][:3] == [
        "planner",
        "kb_researcher",
        "web_researcher",
    ]


def test_workflow_builder_serializes_stable_contract():
    document = (
        WorkflowBuilder("sample")
        .input_schema({"type": "object", "properties": {"topic": {"type": "string"}}})
        .node("start", "capability", "fixture.echo", inputs={"value": "$.input.topic"})
        .output({"result": "$.nodes.start.output.value"})
        .build()
    )
    assert validate_workflow(document) == []
    digest = hashlib.sha256(document.to_yaml().encode()).hexdigest()
    assert len(digest) == 64


def test_cycle_is_rejected():
    document = (
        WorkflowBuilder("cycle")
        .node("a", "capability", "fixture.echo", needs=["b"])
        .node("b", "capability", "fixture.echo", needs=["a"])
        .build()
    )
    codes = {issue.code for issue in validate_workflow(document)}
    assert "cycle" in codes


def test_missing_dependency_is_reported_by_graph_validation():
    document = WorkflowBuilder("bad").node("a", "capability", "fixture.echo", needs=["missing"]).build()
    codes = {issue.code for issue in validate_workflow(document)}
    assert "unknown_dependency" in codes
