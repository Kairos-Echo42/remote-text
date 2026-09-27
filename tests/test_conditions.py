from __future__ import annotations

from agentforge.workflow.conditions import build_evaluation_context, evaluate_condition, resolve_mapping


def test_jmespath_input_and_condition_evaluation():
    context = build_evaluation_context(
        run_input={"topic": "AgentForge"},
        node_outputs={"plan": {"score": 4, "items": ["a", "b"]}},
        run_metadata={"workspace": "demo"},
    )
    assert resolve_mapping("$.input.topic", context) == "AgentForge"
    assert resolve_mapping({"items": "$.nodes.plan.output.items"}, context) == {"items": ["a", "b"]}
    assert evaluate_condition({">": [{"var": "nodes.plan.output.score"}, 3]}, context)
    assert not evaluate_condition({">": [{"var": "nodes.plan.output.score"}, 8]}, context)
