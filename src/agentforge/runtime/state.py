from __future__ import annotations

from enum import StrEnum


class RunStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class NodeStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


RUN_TERMINAL_STATES = {
    RunStatus.SUCCEEDED.value,
    RunStatus.FAILED.value,
    RunStatus.CANCELLED.value,
}

NODE_TERMINAL_STATES = {
    NodeStatus.SUCCEEDED.value,
    NodeStatus.FAILED.value,
    NodeStatus.SKIPPED.value,
    NodeStatus.CANCELLED.value,
}

RUN_TRANSITIONS: dict[str, set[str]] = {
    RunStatus.PENDING.value: {RunStatus.QUEUED.value, RunStatus.CANCELLED.value},
    RunStatus.QUEUED.value: {
        RunStatus.RUNNING.value,
        RunStatus.PAUSED.value,
        RunStatus.CANCELLED.value,
    },
    RunStatus.RUNNING.value: {
        RunStatus.PAUSED.value,
        RunStatus.SUCCEEDED.value,
        RunStatus.FAILED.value,
        RunStatus.CANCELLING.value,
    },
    RunStatus.PAUSED.value: {RunStatus.QUEUED.value, RunStatus.CANCELLED.value},
    RunStatus.CANCELLING.value: {RunStatus.CANCELLED.value},
}


def can_transition_run(current: str, target: str) -> bool:
    if current == target:
        return True
    return target in RUN_TRANSITIONS.get(current, set())
