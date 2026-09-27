# ADR 0004: Docker Sandbox by default outside tests

- Status: Accepted
- Date: 2026-09-13

## Context

Agent-generated code must not execute directly in the Gateway or Worker process.

## Decision

Production and development use the Docker Sandbox provider by default. Tests use the Local Sandbox to stay deterministic. Docker containers use a numeric non-root user, read-only root filesystem, resource limits, no network by default, and an isolated workspace mount.

## Consequences

- The Worker requires access to the Docker engine.
- Network-dependent code must explicitly opt in through configuration.
- Local execution remains possible only for trusted tests and must not be enabled for untrusted content.