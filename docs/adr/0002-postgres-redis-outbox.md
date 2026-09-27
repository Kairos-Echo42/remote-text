# ADR 0002: PostgreSQL source of truth with Redis dispatch

- Status: Accepted
- Date: 2026-09-13

## Context

The runtime needs durable state, ordered audit events, low-latency worker dispatch, and practical recovery.

## Decision

PostgreSQL stores definitions, runs, ordered events, vectors, Memory, and transactional outbox records. Redis Streams carry node dispatch, event fan-out, and cancellation signals.

## Consequences

- Node state and its outbox record commit atomically.
- SSE replay survives Redis loss because PostgreSQL is authoritative.
- Consumers are at-least-once and must use idempotency keys.
- Redis is not described as the source of truth.