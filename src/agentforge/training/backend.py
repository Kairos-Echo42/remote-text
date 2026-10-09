from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agentforge.config import get_settings
from agentforge.training.types import MetricSample


@dataclass(slots=True)
class ExecutionHandle:
    external_id: str
    device: str
    image_digest: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionStatus:
    state: str
    exit_code: int | None = None
    message: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    actual_gpu_seconds: float = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainingBackend(Protocol):
    async def start(
        self, *, job_id: uuid.UUID, attempt_id: uuid.UUID, spec: dict, device_policy: str
    ) -> ExecutionHandle: ...

    async def inspect(self, handle: ExecutionHandle) -> ExecutionStatus: ...

    async def cancel(self, handle: ExecutionHandle) -> ExecutionStatus: ...

    async def metrics(self, handle: ExecutionHandle) -> list[MetricSample]: ...

    async def logs(self, handle: ExecutionHandle, *, tail: int = 200) -> str: ...


class FakeTrainingBackend:
    """Deterministic backend for tests and the no-Docker development path."""

    def __init__(self, *, step_delay: float = 0.01, force_failure: bool = False):
        self.step_delay = step_delay
        self.force_failure = force_failure
        self._runs: dict[str, dict[str, Any]] = {}
        self._samples: dict[str, list[MetricSample]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def start(
        self, *, job_id: uuid.UUID, attempt_id: uuid.UUID, spec: dict, device_policy: str
    ) -> ExecutionHandle:
        external_id = f"fake-{job_id}-{attempt_id}"
        device = "cpu" if device_policy == "cpu_only" else "cuda"
        async with self._lock:
            self._runs[external_id] = {
                "state": "running",
                "started_at": time.time(),
                "finished_at": None,
                "exit_code": None,
                "message": None,
                "device": device,
            }
            self._samples[external_id] = []
            self._tasks[external_id] = asyncio.create_task(self._simulate(external_id, spec))
        return ExecutionHandle(
            external_id=external_id,
            device=device,
            image_digest="fake-backend",
            metadata={
                "output_dir": str(get_settings().training_root / str(job_id) / str(attempt_id)),
                "metrics_path": str(get_settings().training_root / str(job_id) / str(attempt_id) / "metrics.ndjson"),
            },
        )

    async def inspect(self, handle: ExecutionHandle) -> ExecutionStatus:
        async with self._lock:
            item = dict(self._runs[handle.external_id])
        gpu_seconds = 0.0
        if item["device"] == "cuda" and item["started_at"]:
            end = item["finished_at"] or time.time()
            gpu_seconds = max(0.0, end - item["started_at"])
        return ExecutionStatus(
            state=item["state"],
            exit_code=item["exit_code"],
            message=item["message"],
            started_at=item["started_at"],
            finished_at=item["finished_at"],
            actual_gpu_seconds=gpu_seconds,
        )

    async def cancel(self, handle: ExecutionHandle) -> ExecutionStatus:
        task = self._tasks.get(handle.external_id)
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        async with self._lock:
            item = self._runs[handle.external_id]
            item["state"] = "cancelled"
            item["finished_at"] = time.time()
            item["exit_code"] = 143
            item["message"] = "cancelled"
        return await self.inspect(handle)

    async def metrics(self, handle: ExecutionHandle) -> list[MetricSample]:
        async with self._lock:
            return list(self._samples.get(handle.external_id, []))

    async def logs(self, handle: ExecutionHandle, *, tail: int = 200) -> str:
        samples = await self.metrics(handle)
        return "\n".join(sample.model_dump_json() for sample in samples[-tail:])

    async def _simulate(self, external_id: str, spec: dict) -> None:
        try:
            metric = str(spec.get("objective_metric") or spec.get("validation_metric") or "accuracy")
            direction = str(spec.get("objective_direction", "maximize"))
            epochs = max(1, min(int(spec.get("max_epochs", 3)), 5))
            recipe = str(spec.get("recipe_id", ""))
            if "baseline" in recipe:
                base = 0.40
            elif "logistic" in recipe or "ridge" in recipe:
                base = 0.58
            elif "random_forest" in recipe:
                base = 0.68
            elif "mlp" in recipe or "cnn" in recipe:
                base = 0.74
            else:
                base = 0.55
            split = "test" if spec.get("evaluation_only") else "validation"
            for epoch in range(epochs):
                await asyncio.sleep(self.step_delay)
                improvement = (epoch + 1) * 0.02
                value = base + improvement if direction == "maximize" else max(0.01, base - improvement)
                self._samples[external_id].append(
                    MetricSample(name=metric, value=value, split=split, step=epoch, epoch=epoch)
                )
            if self.force_failure:
                raise RuntimeError("simulated training failure")
            async with self._lock:
                item = self._runs[external_id]
                item["state"] = "succeeded"
                item["exit_code"] = 0
                item["finished_at"] = time.time()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            async with self._lock:
                item = self._runs[external_id]
                item["state"] = "failed"
                item["exit_code"] = 1
                item["message"] = str(exc)
                item["finished_at"] = time.time()


def docker_image_digest(image: Any) -> str | None:
    attrs = getattr(image, "attrs", {}) or {}
    repo_digests = attrs.get("RepoDigests") or []
    return repo_digests[0] if repo_digests else attrs.get("Id")


def write_job_spec(root: Path, payload: dict[str, Any]) -> Path:
    path = root / "job.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def checksum_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
