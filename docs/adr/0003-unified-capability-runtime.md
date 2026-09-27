# ADR 0003: Unified Capability Runtime

- Status: Accepted
- Date: 2026-09-13

## Context

MCP tools, RAG, Memory, Sandbox, HTTP, and custom Skills otherwise create separate and inconsistent tool models.

## Decision

All executable tools implement one async Capability contract with JSON Schema, side-effect level, idempotency, and a shared execution context.

## Consequences

- Agent tool bindings, permission filtering, audit events, and schema validation are centralized.
- MCP is an adapter rather than a parallel execution model.
- Non-idempotent capabilities must explicitly declare their retry risk.