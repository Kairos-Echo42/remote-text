from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

import docker
from docker.types import DeviceRequest

from agentforge.config import get_settings
from agentforge.training.backend import ExecutionHandle, ExecutionStatus, docker_image_digest, write_job_spec
from agentforge.training.types import MetricSample


class GPUUnavailableError(RuntimeError):
    pass


class DockerTrainingBackend:
    def __init__(self):
        self.settings = get_settings()
        self._client = None

    def client(self):
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def gpu_available(self) -> bool:
        if not self.settings.trainer_gpu_available:
            return False
        try:
            info = self.client().info()
            return "nvidia" in (info.get("Runtimes") or {})
        except Exception:
            return False

    async def start(
        self,
        *,
        job_id: uuid.UUID,
        attempt_id: uuid.UUID,
        spec: dict,
        device_policy: str,
    ) -> ExecutionHandle:
        local_root = self.settings.training_root / str(job_id) / str(attempt_id)
        local_root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(local_root, 0o777)
        except OSError:
            pass
        in_container = await asyncio.to_thread(Path("/.dockerenv").exists)
        if not self.settings.host_training_root and in_container:
            raise ValueError(
                "AGENTFORGE_HOST_TRAINING_ROOT must be an absolute host path when Trainer runs in Docker"
            )
        host_root = self.settings.host_training_root or str(self.settings.training_root.resolve())
        host_root = host_root.rstrip("/\\")
        host_job_root = f"{host_root}/{job_id}/{attempt_id}"

        dataset_storage_key = spec.get("dataset_storage_key")
        if not dataset_storage_key:
            raise ValueError("job spec is missing dataset_storage_key")
        artifact_root = self.settings.artifact_root.resolve()
        host_artifact_root = str(artifact_root)
        if self.settings.host_training_root:
            host_data_root = self.settings.host_training_root.rstrip("/\\").rsplit("/", 1)[0]
            host_artifact_root = f"{host_data_root}/artifacts"
        spec["dataset_path"] = f"/datasets/{dataset_storage_key}"
        spec["output_dir"] = "/output"
        write_job_spec(local_root, spec)

        use_gpu = device_policy == "required_gpu" or (device_policy == "preferred_gpu" and self.gpu_available())
        if device_policy == "required_gpu" and not self.gpu_available():
            raise GPUUnavailableError("required_gpu policy requested but NVIDIA runtime is unavailable")
        resources = spec.get("recipe_resources") or {}
        memory_mb = max(
            int(resources.get("minimum_memory_mb", 512)),
            int(resources.get("recommended_memory_mb", 2048)),
        )
        cpu_cores = max(1, int(resources.get("cpu_cores", 2)))
        device_requests = [DeviceRequest(count=-1, capabilities=[["gpu"]])] if use_gpu else None
        volumes = {
            host_job_root: {"bind": "/workspace", "mode": "rw"},
            host_artifact_root: {"bind": "/datasets", "mode": "ro"},
        }
        if spec.get("source_attempt_id") and spec.get("source_job_id"):
            source_local = self.settings.training_root / str(spec["source_job_id"]) / str(spec["source_attempt_id"])
            if self.settings.host_training_root:
                host_data_root = self.settings.host_training_root.rstrip("/\\").rsplit("/", 1)[0]
                host_source = f"{host_data_root}/training/{spec['source_job_id']}/{spec['source_attempt_id']}"
            else:
                host_source = str(source_local.resolve())
            volumes[host_source] = {"bind": "/source", "mode": "ro"}
        try:
            image = await asyncio.to_thread(self.client().images.get, self.settings.trainer_image)
            container = await asyncio.to_thread(
                self.client().containers.run,
                self.settings.trainer_image,
                ["python", "-m", "agentforge.training.entrypoint", "/workspace/job.json"],
                detach=True,
                working_dir="/workspace",
                volumes=volumes,
                environment={
                    "HOME": "/tmp",
                    "PYTHONUNBUFFERED": "1",
                    "AGENTFORGE_TRAINING_JOB_ID": str(job_id),
                    "AGENTFORGE_TRAINING_ATTEMPT_ID": str(attempt_id),
                    "AGENTFORGE_GIT_COMMIT": self.settings.git_commit or "",
                },
                network_disabled=True,
                read_only=True,
                user="65534:65534",
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                mem_limit=f"{memory_mb}m",
                nano_cpus=cpu_cores * 1_000_000_000,
                pids_limit=256,
                tmpfs={"/tmp": "rw,noexec,nosuid,size=512m"},
                device_requests=device_requests,
                labels={
                    "agentforge.training_job_id": str(job_id),
                    "agentforge.training_attempt_id": str(attempt_id),
                },
            )
        except Exception:
            if device_policy == "preferred_gpu" and use_gpu:
                return await self.start(
                    job_id=job_id,
                    attempt_id=attempt_id,
                    spec=spec,
                    device_policy="cpu_only",
                )
            raise
        digest = docker_image_digest(image)
        return ExecutionHandle(
            external_id=container.id,
            device="cuda" if use_gpu else "cpu",
            image_digest=digest,
            metadata={
                "fallback": device_policy == "preferred_gpu" and not use_gpu,
                "metrics_path": str(local_root / "metrics.ndjson"),
                "output_dir": str(local_root),
            },
        )

    async def inspect(self, handle: ExecutionHandle) -> ExecutionStatus:
        container = await asyncio.to_thread(self.client().containers.get, handle.external_id)
        await asyncio.to_thread(container.reload)
        state = container.attrs.get("State", {})
        docker_state = state.get("Status", "unknown")
        mapped = {
            "created": "starting",
            "running": "running",
            "exited": "succeeded" if state.get("ExitCode") == 0 else "failed",
            "dead": "failed",
            "removing": "cancelling",
        }.get(docker_state, docker_state)
        if state.get("OOMKilled"):
            mapped = "failed"
        started_at = state.get("StartedAt")
        finished_at = state.get("FinishedAt")
        actual_gpu_seconds = 0.0
        if handle.device == "cuda" and started_at:
            from datetime import UTC, datetime

            def parse(value: str | None):
                if not value or value.startswith("0001-01-01"):
                    return None
                return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)

            start = parse(started_at)
            end = parse(finished_at) or datetime.now(UTC)
            if start:
                actual_gpu_seconds = max(0.0, (end - start).total_seconds())
        return ExecutionStatus(
            state=mapped,
            exit_code=state.get("ExitCode"),
            message=state.get("Error") or ("OOMKilled" if state.get("OOMKilled") else None),
            actual_gpu_seconds=actual_gpu_seconds,
        )

    async def cancel(self, handle: ExecutionHandle) -> ExecutionStatus:
        container = await asyncio.to_thread(self.client().containers.get, handle.external_id)
        try:
            await asyncio.to_thread(container.stop, timeout=30)
        except Exception:
            try:
                await asyncio.to_thread(container.kill)
            except Exception:
                pass
        return await self.inspect(handle)

    async def metrics(self, handle: ExecutionHandle) -> list[MetricSample]:
        path = Path(handle.metadata.get("metrics_path", ""))
        if not await asyncio.to_thread(path.exists):
            return []
        content = await asyncio.to_thread(path.read_text, encoding="utf-8")
        samples: list[MetricSample] = []
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                samples.append(MetricSample.model_validate(json.loads(line)))
            except Exception:
                continue
        return samples

    async def logs(self, handle: ExecutionHandle, *, tail: int = 200) -> str:
        container = await asyncio.to_thread(self.client().containers.get, handle.external_id)
        raw = await asyncio.to_thread(container.logs, tail=tail)
        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
