from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentforge.models import BudgetReservation, Experiment, TrainingJob
from agentforge.training.types import BudgetUsage, ExperimentBudget


class BudgetExceededError(ValueError):
    pass


async def reserve_job_budget(
    session: AsyncSession,
    *,
    job: TrainingJob,
) -> BudgetReservation:
    experiment = await session.get(Experiment, job.experiment_id, with_for_update=True)
    if experiment is None:
        raise ValueError("experiment not found")
    budget = ExperimentBudget.model_validate(experiment.budget)
    usage = BudgetUsage(
        reserved_total_seconds=experiment.reserved_total_seconds,
        consumed_total_seconds=experiment.consumed_total_seconds,
        reserved_gpu_seconds=experiment.reserved_gpu_seconds,
        consumed_gpu_seconds=experiment.consumed_gpu_seconds,
        jobs_started=experiment.jobs_started,
    )
    total_reservation = float(min(job.spec["max_job_seconds"], budget.max_job_seconds))
    gpu_reservation = total_reservation if job.device_policy != "cpu_only" else 0.0
    try:
        if job.job_kind != "final_test":
            usage.check_available(
                budget,
                total_reservation=total_reservation,
                gpu_reservation=gpu_reservation,
            )
        else:
            usage.check_available(
                budget.model_copy(update={"max_jobs": budget.max_jobs + 1}),
                total_reservation=total_reservation,
                gpu_reservation=gpu_reservation,
            )
    except ValueError as exc:
        raise BudgetExceededError(str(exc)) from exc

    experiment.reserved_total_seconds += total_reservation
    experiment.reserved_gpu_seconds += gpu_reservation
    job.reserved_total_seconds = total_reservation
    job.reserved_gpu_seconds = gpu_reservation
    if job.job_kind != "final_test":
        experiment.jobs_started += 1
    reservation = BudgetReservation(
        experiment_id=experiment.id,
        job_id=job.id,
        total_seconds=total_reservation,
        gpu_seconds=gpu_reservation,
        status="active",
    )
    session.add(reservation)
    await session.flush()
    return reservation


async def release_gpu_reservation(
    session: AsyncSession,
    *,
    job_id: uuid.UUID,
    fallback_reason: str,
) -> None:
    reservation = await session.scalar(
        select(BudgetReservation).where(BudgetReservation.job_id == job_id).with_for_update()
    )
    if reservation is None or reservation.gpu_seconds <= 0:
        return
    experiment = await session.get(Experiment, reservation.experiment_id, with_for_update=True)
    if experiment is None:
        return
    experiment.reserved_gpu_seconds = max(0.0, experiment.reserved_gpu_seconds - reservation.gpu_seconds)
    reservation.gpu_seconds = 0
    reservation.status = "total_only"
    job = await session.get(TrainingJob, job_id, with_for_update=True)
    if job is not None:
        job.fallback_reason = fallback_reason
        job.reserved_gpu_seconds = 0


async def finalize_job_budget(
    session: AsyncSession,
    *,
    job_id: uuid.UUID,
    actual_total_seconds: float,
    actual_gpu_seconds: float,
) -> BudgetReservation:
    reservation = await session.scalar(
        select(BudgetReservation).where(BudgetReservation.job_id == job_id).with_for_update()
    )
    if reservation is None:
        raise ValueError("budget reservation not found")
    if reservation.status == "finalized":
        return reservation
    experiment = await session.get(Experiment, reservation.experiment_id, with_for_update=True)
    if experiment is None:
        raise ValueError("experiment not found")

    reserved_total = reservation.total_seconds
    reserved_gpu = reservation.gpu_seconds
    consumed_total = max(0.0, min(actual_total_seconds, reserved_total))
    consumed_gpu = max(0.0, min(actual_gpu_seconds, reserved_gpu))

    experiment.reserved_total_seconds = max(0.0, experiment.reserved_total_seconds - reserved_total)
    experiment.reserved_gpu_seconds = max(0.0, experiment.reserved_gpu_seconds - reserved_gpu)
    experiment.consumed_total_seconds += consumed_total
    experiment.consumed_gpu_seconds += consumed_gpu

    job = await session.get(TrainingJob, job_id, with_for_update=True)
    if job is not None:
        job.actual_total_seconds = consumed_total
        job.actual_gpu_seconds = consumed_gpu

    reservation.consumed_total_seconds = consumed_total
    reservation.consumed_gpu_seconds = consumed_gpu
    reservation.status = "finalized"
    reservation.released_at = datetime.now(UTC)
    return reservation
