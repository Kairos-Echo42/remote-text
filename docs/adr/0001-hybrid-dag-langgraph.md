# ADR 0001: Hybrid DAG control plane and LangGraph agent loops

- Status: Accepted
- Date: 2026-09-13

## Context

The platform needs inspectable, recoverable workflow orchestration and mature single-Agent reasoning/tool loops.

## Decision

AgentForge owns the outer DAG, persistence, concurrency, leases, retries, events, and recovery. LangGraph is used only inside Agent node execution.

## Consequences

- Workflow behavior remains visible in platform tables and UI.
- Agent reasoning benefits from LangGraph without hiding orchestration semantics.
- Platform state and LangGraph checkpoints require explicit correlation; `node_run_id` is the checkpoint thread id.