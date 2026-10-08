from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from agentforge.config import get_settings
from agentforge.db import get_session_factory
from agentforge.logging import get_logger
from agentforge.models import (
    Artifact,
    Checkpoint,
    DatasetVersion,
    Experiment,
    MetricPoint,
    TrainingAttempt,
    TrainingEvent,
    TrainingJob,
)
from agentforge.runtime.queue import get_redis
from agentforge.storage.artifacts import LocalArtifactStore
from agentforge.training.audit import ensure_lineage_node, link_lineage
from agentforge.training.backend import ExecutionHandle, FakeTrainingBackend, TrainingBackend
from agentforge.training.budget import finalize_job_budget, release_gpu_reservation
from agentforge.training.docker_backend import DockerTrainingBackend, GPUUnavailableError
from agentforge.training.types import MetricSample, ReproducibilityManifest

logger = get_logger(__name__)

TERMINAL = {"succeeded", "failed", "cancelled"}


class TrainingWorker:
    def __init__(self, backend: TrainingBackend | None = None):
        self.settings = get_settings()
        self.backend = backend or DockerTrainingBackend()
        self.active: dict[uuid.UUID, tuple[ExecutionHandle, datetime, set[tuple[str, str, int, int | None]]]] = {}
        self._stop = asyncio.Event()

    async def run_forever(self, *, interval_seconds: float = 1.0) -> None:
        while not self._stop.is_set():
            try:
                await self.tick()
            except Exception:
                logger.exception("training_worker_tick_failed")
            await asyncio.sleep(interval_seconds)

    def stop(self) -> None:
        self._stop.set()

    async def tick(self) -> None:
        await self._recover_active()
        await self._cancel_queued()
        await self._start_queued()
        await self._poll_active()

    async def _cancel_queued(self) -> None:
        factory = get_session_factory()
        async with factory() as session:
            jobs = list(
                (
                    await session.scalars(
                        select(TrainingJob).where(
                            TrainingJob.status == "cancelling",
                            TrainingJob.container_id.is_(None),
                        )
                    )
                ).all()
            )
        for job in jobs:
            async with factory() as session:
                async with session.begin():
                    row = await session.get(TrainingJob, job.id, with_for_update=True)
                    if row is None or row.status != "cancelling" or row.container_id is not None:
                        continue
                    row.status = "cancelled"
                    row.finished_at = datetime.now(UTC)
                    await finalize_job_budget(
                        session,
                        job_id=row.id,
                        actual_total_seconds=0,
                        actual_gpu_seconds=0,
                    )
                    await self._event(session, row.id, None, "training.job.cancelled", {})
            try:
                await get_redis().delete(f"agentforge:training:cancel:{job.id}")
            except Exception:
                pass

    async def _recover_active(self) -> None:
        factory = get_session_factory()
        async with factory() as session:
            jobs = list(
                (
                    await session.scalars(select(TrainingJob).where(TrainingJob.status.in_(["starting", "running"])))
                ).all()
            )
        for job in jobs:
            if job.id in self.active or not job.container_id:
                continue
            handle = ExecutionHandle(
                external_id=job.container_id,
                device="cpu" if job.fallback_reason else ("cuda" if job.queue_name == "gpu" else "cpu"),
                image_digest=job.image_digest,
            )
            points = list(
                (
                    await session.scalars(
                        select(MetricPoint).where(MetricPoint.job_id == job.id)
                    )
                ).all()
            )
            seen = {(point.name, point.split, point.step, point.epoch) for point in points}
            self.active[job.id] = (handle, job.started_at or datetime.now(UTC), seen)
            logger.info("training_job_reattached", job_id=str(job.id), container_id=job.container_id)

    async def _start_queued(self) -> None:
        gpu_active = sum(1 for handle, _, _ in self.active.values() if handle.device == "cuda")
        cpu_active = len(self.active) - gpu_active
        factory = get_session_factory()
        async with factory() as session:
            jobs = list(
                (
                    await session.scalars(
                        select(TrainingJob)
                        .where(TrainingJob.status == "queued")
                        .order_by(TrainingJob.created_at)
                        .with_for_update(skip_locked=True)
                        .limit(20)
                    )
                ).all()
            )
        for job in jobs:
            if job.id in self.active:
                continue
            wants_gpu = job.queue_name == "gpu" and job.device_policy != "cpu_only"
            if wants_gpu and gpu_active >= self.settings.training_gpu_concurrency:
                continue
            if not wants_gpu and cpu_active >= self.settings.training_cpu_concurrency:
                continue
            try:
                await self._start_job(job.id)
            except GPUUnavailableError as exc:
                await self._fail_before_start(job.id, str(exc))
                continue
            except Exception as exc:
                await self._fail_before_start(job.id, str(exc))
                continue
            gpu_active += int(wants_gpu)
            cpu_active += int(not wants_gpu)

    async def _start_job(self, job_id: uuid.UUID) -> None:
        factory = get_session_factory()
        async with factory() as session:
            async with session.begin():
                job = await session.get(TrainingJob, job_id, with_for_update=True)
                if job is None or job.status != "queued":
                    return
                attempt_number = (
                    await session.scalar(
                        select(func.coalesce(func.max(TrainingAttempt.attempt), 0)).where(
                            TrainingAttempt.job_id == job.id
                        )
                    )
                    or 0
                ) + 1
                attempt = TrainingAttempt(
                    job_id=job.id,
                    attempt=attempt_number,
                    status="starting",
                    started_at=datetime.now(UTC),
                )
                session.add(attempt)
                await session.flush()
                job.status = "starting"
                job.started_at = job.started_at or datetime.now(UTC)
                await self._event(session, job.id, attempt.id, "training.job.starting", {})
                spec = dict(job.spec)

        device_policy = job.device_policy
        dataset_version = None
        async with factory() as session:
            dataset_version = await session.get(DatasetVersion, job.dataset_version_id)
        if dataset_version is None:
            raise RuntimeError("dataset version disappeared")
        spec["dataset_storage_key"] = dataset_version.storage_key
        spec["dataset_version"] = {
            "id": str(dataset_version.id),
            "checksum": dataset_version.checksum,
            "split_checksum": dataset_version.split_checksum,
            "split_manifest": dataset_version.split_manifest,
            "profile": dataset_version.profile,
            "target_column": dataset_version.target_column,
            "kind": (dataset_version.profile or {}).get("dataset_kind"),
            "task_type": (dataset_version.profile or {}).get("task_type"),
        }
        spec["objective_metric"] = job.spec.get("objective_metric")
        spec["objective_direction"] = job.spec.get("objective_direction")
        if job.job_kind == "final_test" and job.spec.get("source_job_id"):
            source_attempt = await session.scalar(
                select(TrainingAttempt)
                .where(TrainingAttempt.job_id == uuid.UUID(str(job.spec["source_job_id"])))
                .order_by(TrainingAttempt.attempt.desc())
                .limit(1)
            )
            if source_attempt is None:
                raise RuntimeError("source training attempt not found for final evaluation")
            spec["source_attempt_id"] = str(source_attempt.id)
            spec["source_path"] = "/source"
        try:
            handle = await self.backend.start(
                job_id=job.id,
                attempt_id=attempt.id,
                spec=spec,
                device_policy=device_policy,
            )
        except GPUUnavailableError:
            raise
        except Exception:
            if device_policy == "preferred_gpu":
                fallback_reason = "preferred_gpu unavailable; using CPU"
                async with factory() as session:
                    async with session.begin():
                        await release_gpu_reservation(session, job_id=job.id, fallback_reason=fallback_reason)
                handle = await self.backend.start(
                    job_id=job.id,
                    attempt_id=attempt.id,
                    spec=spec,
                    device_policy="cpu_only",
                )
            else:
                raise
        async with factory() as session:
            async with session.begin():
                job_row = await session.get(TrainingJob, job.id, with_for_update=True)
                attempt_row = await session.get(TrainingAttempt, attempt.id, with_for_update=True)
                if job_row is None or attempt_row is None:
                    raise RuntimeError("training job attempt disappeared")
                job_row.status = "running"
                job_row.container_id = handle.external_id
                job_row.image_digest = handle.image_digest
                if handle.device == "cpu" and job_row.device_policy == "preferred_gpu":
                    fallback_reason = "preferred_gpu unavailable; using CPU"
                    await release_gpu_reservation(
                        session,
                        job_id=job_row.id,
                        fallback_reason=fallback_reason,
                    )
                    job_row.fallback_reason = fallback_reason
                attempt_row.status = "running"
                attempt_row.container_id = handle.external_id
                started_job_id = job_row.id
                await self._event(
                    session,
                    job_row.id,
                    attempt_row.id,
                    "training.job.running",
                    {"device": handle.device, "image_digest": handle.image_digest},
                )
        self.active[started_job_id] = (handle, datetime.now(UTC), set())
        logger.info("training_job_started", job_id=str(started_job_id), device=handle.device)

    async def _poll_active(self) -> None:
        if not self.active:
            return
        for job_id, (handle, started_at, seen_metrics) in list(self.active.items()):
            try:
                if await get_redis().get(f"agentforge:training:cancel:{job_id}"):
                    await self.backend.cancel(handle)
            except Exception:
                pass
            async with get_session_factory()() as session:
                job_record = await session.get(TrainingJob, job_id)
            max_job_seconds = int((job_record.spec or {}).get("max_job_seconds", 0)) if job_record else 0
            if max_job_seconds and (datetime.now(UTC) - started_at).total_seconds() >= max_job_seconds:
                await self.backend.cancel(handle)
            try:
                status = await self.backend.inspect(handle)
                samples = await self.backend.metrics(handle)
            except Exception as exc:
                await self._finalize(
                    job_id=job_id,
                    handle=handle,
                    status="failed",
                    error=f"trainer recovery failed: {exc}",
                    actual_total_seconds=max(0.0, (datetime.now(UTC) - started_at).total_seconds()),
                    actual_gpu_seconds=0.0,
                )
                self.active.pop(job_id, None)
                continue
            await self._persist_metrics(job_id, samples, seen_metrics)
            if status.state not in TERMINAL:
                continue
            actual_total = max(
                0.0,
                ((status.finished_at or datetime.now(UTC).timestamp()) - started_at.timestamp()),
            )
            await self._finalize(
                job_id=job_id,
                handle=handle,
                status=status.state,
                error=status.message,
                actual_total_seconds=actual_total,
                actual_gpu_seconds=status.actual_gpu_seconds,
            )
            self.active.pop(job_id, None)

    async def _persist_metrics(
        self,
        job_id: uuid.UUID,
        samples: list[MetricSample],
        seen: set[tuple[str, str, int, int | None]],
    ) -> None:
        new_samples = [
            sample for sample in samples if (sample.name, sample.split, sample.step, sample.epoch) not in seen
        ]
        if not new_samples:
            return
        factory = get_session_factory()
        async with factory() as session:
            async with session.begin():
                attempt = await session.scalar(
                    select(TrainingAttempt)
                    .where(TrainingAttempt.job_id == job_id)
                    .order_by(TrainingAttempt.attempt.desc())
                    .limit(1)
                )
                for sample in new_samples:
                    session.add(
                        MetricPoint(
                            job_id=job_id,
                            attempt_id=attempt.id if attempt else None,
                            name=sample.name,
                            value=sample.value,
                            split=sample.split,
                            step=sample.step,
                            epoch=sample.epoch,
                            created_at=sample.observed_at,
                        )
                    )
                    seen.add((sample.name, sample.split, sample.step, sample.epoch))
                    await self._event(
                        session,
                        job_id,
                        attempt.id if attempt else None,
                        "training.metric",
                        sample.model_dump(mode="json"),
                    )

    async def _finalize(
        self,
        *,
        job_id: uuid.UUID,
        handle: ExecutionHandle,
        status: str,
        error: str | None,
        actual_total_seconds: float,
        actual_gpu_seconds: float,
    ) -> None:
        metrics: dict[str, dict[str, float]] = {"train": {}, "validation": {}, "test": {}, "system": {}}
        factory = get_session_factory()
        async with factory() as session:
            points = list(
                (
                    await session.scalars(
                        select(MetricPoint).where(MetricPoint.job_id == job_id).order_by(MetricPoint.created_at)
                    )
                ).all()
            )
            job_record = await session.get(TrainingJob, job_id)
            dataset_version = await session.get(DatasetVersion, job_record.dataset_version_id) if job_record else None
            experiment = await session.get(Experiment, job_record.experiment_id) if job_record else None
        for point in points:
            metrics.setdefault(point.split, {})[point.name] = point.value

        environment: dict[str, Any] = {}
        if handle.metadata.get("output_dir"):
            environment_path = Path(handle.metadata["output_dir"]) / "environment.json"
            if await asyncio.to_thread(environment_path.exists):
                try:
                    environment = json.loads(await asyncio.to_thread(environment_path.read_text, encoding="utf-8"))
                except Exception:
                    environment = {}
        git_commit = self.settings.git_commit or os.environ.get("AGENTFORGE_GIT_COMMIT")
        manifest_values = {
            "python_version": environment.get("python_version"),
            "pytorch_version": environment.get("pytorch_version"),
            "sklearn_version": environment.get("sklearn_version"),
            "cuda_version": environment.get("cuda_version"),
            "cudnn_version": environment.get("cudnn_version"),
            "docker_image_digest": handle.image_digest,
            "git_commit": git_commit,
        }
        unavailable = [name for name, value in manifest_values.items() if not value]
        manifest = ReproducibilityManifest(
            dataset_version_checksum=dataset_version.checksum if dataset_version else None,
            split_checksum=dataset_version.split_checksum if dataset_version else None,
            baseline_strategy=((experiment.baseline_strategy or {}).get("qualified_id") if experiment else None),
            recipe_id=job_record.recipe_id if job_record else None,
            recipe_version=job_record.recipe_version if job_record else None,
            recipe_checksum=job_record.recipe_checksum if job_record else None,
            config_checksum=job_record.config_checksum if job_record else None,
            random_seed=int((job_record.spec or {}).get("random_seed", 42)) if job_record else None,
            python_version=environment.get("python_version"),
            pytorch_version=environment.get("pytorch_version"),
            sklearn_version=environment.get("sklearn_version"),
            cuda_version=environment.get("cuda_version"),
            cudnn_version=environment.get("cudnn_version"),
            docker_image_digest=handle.image_digest,
            git_commit=git_commit,
            hardware={"device": handle.device},
            unavailable=unavailable,
        )
        manifest = manifest.model_copy(
            update={
                "hardware": {
                    **dict(environment.get("hardware") or {}),
                    "device": handle.device,
                }
            }
        )

        artifact_id = None
        if status == "succeeded" and job_record and job_record.job_kind != "final_test":
            try:
                artifact_id, _checkpoint_id = await self._package_outputs(
                    job_id,
                    handle,
                    metrics,
                    manifest=manifest,
                    actual_total_seconds=actual_total_seconds,
                    actual_gpu_seconds=actual_gpu_seconds,
                )
            except Exception as exc:
                status = "failed"
                error = f"artifact packaging failed: {exc}"
        async with factory() as session:
            async with session.begin():
                job = await session.get(TrainingJob, job_id, with_for_update=True)
                if job is None:
                    return
                attempt = await session.scalar(
                    select(TrainingAttempt)
                    .where(TrainingAttempt.job_id == job_id)
                    .order_by(TrainingAttempt.attempt.desc())
                    .limit(1)
                )
                if attempt is None:
                    return
                job.status = status
                job.error = error
                job.finished_at = datetime.now(UTC)
                attempt.status = status
                attempt.finished_at = datetime.now(UTC)
                attempt.metrics = metrics
                attempt.manifest = {
                    **manifest.model_dump(mode="json"),
                    "actual_total_seconds": actual_total_seconds,
                    "actual_gpu_seconds": actual_gpu_seconds,
                }
                await finalize_job_budget(
                    session,
                    job_id=job.id,
                    actual_total_seconds=actual_total_seconds,
                    actual_gpu_seconds=actual_gpu_seconds,
                )
                if artifact_id:
                    job.spec = {**job.spec, "bundle_artifact_id": str(artifact_id)}
                await self._event(
                    session,
                    job.id,
                    attempt.id,
                    f"training.job.{status}",
                    {"error": error, "bundle_artifact_id": str(artifact_id) if artifact_id else None},
                )
        try:
            await get_redis().delete(f"agentforge:training:cancel:{job_id}")
        except Exception:
            pass

    async def _package_outputs(
        self,
        job_id: uuid.UUID,
        handle: ExecutionHandle,
        metrics: dict[str, dict[str, float]],
        *,
        manifest: ReproducibilityManifest,
        actual_total_seconds: float,
        actual_gpu_seconds: float,
    ) -> tuple[uuid.UUID | None, uuid.UUID | None]:
        output_dir = Path(handle.metadata.get("output_dir", "")) if handle.metadata else Path()
        if not output_dir or not output_dir.exists():
            output_dir = self.settings.training_root / str(job_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.pt"
        preprocessor_path = output_dir / "preprocessor.joblib"
        checkpoint_path = output_dir / "checkpoint.pt"
        is_fake_backend = str(handle.external_id).startswith("fake-")
        placeholders = {
            model_path: b"agentforge-placeholder-model",
            preprocessor_path: b"agentforge-placeholder-preprocessor",
            checkpoint_path: b"agentforge-placeholder-checkpoint",
        }
        for path, placeholder in placeholders.items():
            if not path.exists():
                if not is_fake_backend:
                    raise RuntimeError(f"trainer did not produce required bundle file {path.name}")
                path.write_bytes(placeholder)
        schema_path = output_dir / "schema.json"
        if not schema_path.exists():
            schema_path.write_text(json.dumps({"input": {}, "output": {}}, indent=2), encoding="utf-8")
        reproducibility = {
            **manifest.model_dump(mode="json"),
            "job_id": str(job_id),
            "actual_total_seconds": actual_total_seconds,
            "actual_gpu_seconds": actual_gpu_seconds,
        }
        bundle_root = output_dir / "bundle"
        bundle_root.mkdir(parents=True, exist_ok=True)
        files = {
            "model.pt": model_path.read_bytes(),
            "preprocessor.joblib": preprocessor_path.read_bytes(),
            "schema.json": schema_path.read_bytes(),
            "metrics.json": json.dumps(metrics, indent=2).encode(),
            "training-config.json": json.dumps({"job_id": str(job_id)}, indent=2).encode(),
            "reproducibility.json": json.dumps(reproducibility, indent=2).encode(),
            "lineage.json": json.dumps({"job_id": str(job_id)}, indent=2).encode(),
        }
        checksums = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
        files["checksums.json"] = json.dumps(checksums, indent=2, sort_keys=True).encode()
        bundle_path = output_dir / "model.agentforge.zip"
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        zip_checksum = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        store = LocalArtifactStore()
        async with get_session_factory()() as session:
            async with session.begin():
                job = await session.get(TrainingJob, job_id)
                if job is None:
                    return None, None
                training_config = {
                    "job_id": str(job.id),
                    "experiment_id": str(job.experiment_id),
                    "dataset_version_id": str(job.dataset_version_id),
                    "recipe_id": job.recipe_id,
                    "recipe_version": job.recipe_version,
                    "recipe_checksum": job.recipe_checksum,
                    "config_checksum": job.config_checksum,
                    "spec": job.spec,
                }
                lineage = {
                    "dataset_version_id": str(job.dataset_version_id),
                    "experiment_id": str(job.experiment_id),
                    "training_job_id": str(job.id),
                }
                files["training-config.json"] = json.dumps(training_config, indent=2).encode()
                files["lineage.json"] = json.dumps(lineage, indent=2).encode()
                checksums = {
                    name: hashlib.sha256(content).hexdigest()
                    for name, content in files.items()
                    if name != "checksums.json"
                }
                files["checksums.json"] = json.dumps(checksums, indent=2, sort_keys=True).encode()
                with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for name, content in files.items():
                        archive.writestr(name, content)
                zip_checksum = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
                stored_bundle = await store.put(
                    bundle_path.read_bytes(),
                    workspace_id=job.workspace_id,
                    run_id=None,
                    filename="model.agentforge.zip",
                    content_type="application/zip",
                )
                bundle_artifact = Artifact(
                    workspace_id=job.workspace_id,
                    run_id=None,
                    name="model.agentforge.zip",
                    content_type="application/zip",
                    size=stored_bundle.size,
                    checksum=zip_checksum,
                    storage_key=stored_bundle.key,
                    metadata_={"job_id": str(job_id), "kind": "model_bundle"},
                )
                session.add(bundle_artifact)
                await session.flush()
                stored_checkpoint = await store.put(
                    checkpoint_path.read_bytes(),
                    workspace_id=job.workspace_id,
                    run_id=None,
                    filename="checkpoint.pt",
                    content_type="application/octet-stream",
                )
                checkpoint_artifact = Artifact(
                    workspace_id=job.workspace_id,
                    run_id=None,
                    name="checkpoint.pt",
                    content_type="application/octet-stream",
                    size=stored_checkpoint.size,
                    checksum=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                    storage_key=stored_checkpoint.key,
                    metadata_={"job_id": str(job_id), "kind": "checkpoint"},
                )
                session.add(checkpoint_artifact)
                await session.flush()
                attempt = await session.scalar(
                    select(TrainingAttempt)
                    .where(TrainingAttempt.job_id == job_id)
                    .order_by(TrainingAttempt.attempt.desc())
                    .limit(1)
                )
                if attempt is None:
                    raise RuntimeError("training attempt disappeared before checkpoint packaging")
                checkpoint = Checkpoint(
                    job_id=job.id,
                    attempt_id=attempt.id,
                    artifact_id=checkpoint_artifact.id,
                    epoch=_checkpoint_integer(output_dir, "epoch"),
                    step=_checkpoint_integer(output_dir, "step"),
                    validation_metric=job.spec.get("objective_metric"),
                    validation_value=_checkpoint_validation_value(
                        output_dir,
                        str(job.spec.get("objective_metric") or ""),
                        metrics,
                    ),
                    checksum=checkpoint_artifact.checksum,
                    metadata_={"resume_supported": False},
                )
                session.add(checkpoint)
                await session.flush()
                job_node = await ensure_lineage_node(
                    session,
                    workspace_id=job.workspace_id,
                    node_type="training_job",
                    ref_id=job.id,
                )
                attempt_node = await ensure_lineage_node(
                    session,
                    workspace_id=job.workspace_id,
                    node_type="training_attempt",
                    ref_id=attempt.id,
                )
                checkpoint_node = await ensure_lineage_node(
                    session,
                    workspace_id=job.workspace_id,
                    node_type="checkpoint",
                    ref_id=checkpoint.id,
                    artifact_id=checkpoint_artifact.id,
                    metadata={"checksum": checkpoint_artifact.checksum},
                )
                bundle_node = await ensure_lineage_node(
                    session,
                    workspace_id=job.workspace_id,
                    node_type="artifact",
                    ref_id=bundle_artifact.id,
                    artifact_id=bundle_artifact.id,
                    metadata={"checksum": bundle_artifact.checksum},
                )
                await link_lineage(
                    session,
                    workspace_id=job.workspace_id,
                    input_node=job_node,
                    output_node=attempt_node,
                    operation="attempt",
                    actor_type="trainer",
                )
                await link_lineage(
                    session,
                    workspace_id=job.workspace_id,
                    input_node=attempt_node,
                    output_node=checkpoint_node,
                    operation="write_checkpoint",
                    actor_type="trainer",
                )
                await link_lineage(
                    session,
                    workspace_id=job.workspace_id,
                    input_node=checkpoint_node,
                    output_node=bundle_node,
                    operation="package_model",
                    actor_type="trainer",
                )
                return bundle_artifact.id, checkpoint.id

    async def _fail_before_start(self, job_id: uuid.UUID, error: str) -> None:
        factory = get_session_factory()
        async with factory() as session:
            async with session.begin():
                job = await session.get(TrainingJob, job_id, with_for_update=True)
                if job is None:
                    return
                attempt = await session.scalar(
                    select(TrainingAttempt)
                    .where(TrainingAttempt.job_id == job_id)
                    .order_by(TrainingAttempt.attempt.desc())
                    .limit(1)
                )
                if attempt is not None:
                    attempt.status = "failed"
                    attempt.error = error
                    attempt.finished_at = datetime.now(UTC)
                job.status = "failed"
                job.error = error
                job.finished_at = datetime.now(UTC)
                await finalize_job_budget(session, job_id=job.id, actual_total_seconds=0, actual_gpu_seconds=0)
                await self._event(session, job.id, None, "training.job.failed", {"error": error})

    async def _event(
        self,
        session,
        job_id: uuid.UUID,
        attempt_id: uuid.UUID | None,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        next_seq = (
            await session.scalar(
                select(func.coalesce(func.max(TrainingEvent.seq), 0)).where(TrainingEvent.job_id == job_id)
            )
            or 0
        ) + 1
        session.add(
            TrainingEvent(
                job_id=job_id,
                attempt_id=attempt_id,
                seq=next_seq,
                event_type=event_type,
                payload=payload,
            )
        )


async def run_training_worker() -> None:
    backend: TrainingBackend
    if get_settings().env == "test":
        backend = FakeTrainingBackend()
    else:
        backend = DockerTrainingBackend()
    await TrainingWorker(backend).run_forever()


def _checkpoint_metadata(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "checkpoint-metadata.json"
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _checkpoint_integer(output_dir: Path, key: str) -> int | None:
    value = _checkpoint_metadata(output_dir).get(key)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _checkpoint_validation_value(
    output_dir: Path,
    metric: str,
    metrics: dict[str, dict[str, float]],
) -> float | None:
    value = metrics.get("validation", {}).get(metric)
    if value is not None:
        return float(value)
    raw_validation = _checkpoint_metadata(output_dir).get("validation")
    if isinstance(raw_validation, dict) and metric in raw_validation:
        try:
            return float(raw_validation[metric])
        except (TypeError, ValueError):
            return None
    return None
