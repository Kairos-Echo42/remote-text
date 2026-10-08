from __future__ import annotations

import csv
import hashlib
import io
import random
import uuid
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from agentforge.models import Dataset, DatasetVersion
from agentforge.storage.artifacts import ArtifactStore
from agentforge.training.types import (
    DatasetKind,
    SplitAlgorithm,
    SplitManifest,
    TaskType,
)


async def create_dataset(
    session: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    name: str,
    kind: DatasetKind,
    task_type: TaskType,
    description: str | None = None,
    created_by: uuid.UUID | None = None,
) -> Dataset:
    valid_tasks = {
        DatasetKind.TABULAR: {
            TaskType.TABULAR_CLASSIFICATION.value,
            TaskType.TABULAR_REGRESSION.value,
        },
        DatasetKind.IMAGE_FOLDER: {TaskType.IMAGE_CLASSIFICATION.value},
    }
    if task_type.value not in valid_tasks[kind]:
        raise ValueError(f"task type {task_type.value!r} is not valid for {kind.value!r}")
    dataset = Dataset(
        workspace_id=workspace_id,
        name=name,
        kind=kind.value,
        task_type=task_type.value,
        description=description,
        created_by=created_by,
    )
    session.add(dataset)
    await session.flush()
    return dataset


async def upload_dataset_version(
    session: AsyncSession,
    *,
    dataset: Dataset,
    filename: str,
    content: bytes,
    artifact_store: ArtifactStore,
    target_column: str | None = None,
    split_seed: int = 42,
) -> DatasetVersion:
    checksum = hashlib.sha256(content).hexdigest()
    existing = await session.scalar(
        select(DatasetVersion).where(
            DatasetVersion.dataset_id == dataset.id,
            DatasetVersion.checksum == checksum,
        )
    )
    if existing is not None and existing.status == "ready":
        return existing
    version_number = (
        await session.scalar(select(func.count(DatasetVersion.id)).where(DatasetVersion.dataset_id == dataset.id)) or 0
    ) + 1
    stored = await artifact_store.put(
        content,
        workspace_id=dataset.workspace_id,
        run_id=None,
        filename=filename,
    )
    version = DatasetVersion(
        dataset_id=dataset.id,
        version=f"v{version_number}",
        status="validating",
        storage_key=stored.key,
        checksum=checksum,
        size=len(content),
        split_seed=split_seed,
        target_column=target_column,
    )
    session.add(version)
    await session.flush()
    try:
        if dataset.kind == DatasetKind.TABULAR.value:
            profile, split, algorithm, stratification = _tabular_profile_and_split(
                filename, content, target_column=target_column, seed=split_seed
            )
        elif dataset.kind == DatasetKind.IMAGE_FOLDER.value:
            profile, split, algorithm, stratification = _image_profile_and_split(filename, content, seed=split_seed)
        else:
            raise ValueError(f"unsupported dataset kind {dataset.kind!r}")
        version.profile = profile
        version.row_count = profile.get("row_count")
        version.column_count = profile.get("column_count")
        version.split_manifest = split.model_dump(mode="json")
        version.split_checksum = split.checksum
        version.split_algorithm = algorithm.value
        version.stratification = stratification
        dataset.task_type = str(profile.get("task_type") or dataset.task_type)
        if not split.train or not split.validation or not split.test:
            raise ValueError("dataset split must produce non-empty train, validation and test sets")
        version.status = "ready"
    except Exception as exc:
        version.status = "failed"
        version.error = str(exc)
    return version


def _tabular_profile_and_split(
    filename: str,
    content: bytes,
    *,
    target_column: str | None,
    seed: int,
) -> tuple[dict[str, Any], SplitManifest, SplitAlgorithm, dict[str, Any] | None]:
    if not target_column:
        raise ValueError("target_column is required for tabular datasets")
    decoded = content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(decoded))
    if reader.fieldnames is None:
        raise ValueError("CSV has no header")
    if target_column not in reader.fieldnames:
        raise ValueError(f"target column {target_column!r} not found")
    rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no rows")
    targets = [str(row[target_column]) for row in rows]
    class_counts = Counter(targets)
    missing = {column: sum(1 for row in rows if row.get(column) in {None, ""}) for column in reader.fieldnames}
    profile = {
        "dataset_kind": DatasetKind.TABULAR.value,
        "task_type": (
            TaskType.TABULAR_REGRESSION.value
            if _is_regression_target(targets)
            else TaskType.TABULAR_CLASSIFICATION.value
        ),
        "row_count": len(rows),
        "column_count": len(reader.fieldnames),
        "target_column": target_column,
        "class_count": len(class_counts) if not _is_regression_target(targets) else None,
        "missing_ratio": sum(missing.values()) / max(1, len(rows) * len(reader.fieldnames)),
        "warnings": [],
        "columns": reader.fieldnames,
    }
    if profile["task_type"] == TaskType.TABULAR_CLASSIFICATION.value:
        split = _stratified_split(
            [str(index) for index in range(len(rows))],
            targets,
            seed=seed,
        )
        algorithm = SplitAlgorithm.STRATIFIED
        stratification = {"target_column": target_column}
    else:
        split = _random_split([str(index) for index in range(len(rows))], seed=seed)
        algorithm = SplitAlgorithm.RANDOM
        stratification = None
    return profile, split, algorithm, stratification


def _image_profile_and_split(
    filename: str,
    content: bytes,
    *,
    seed: int,
) -> tuple[dict[str, Any], SplitManifest, SplitAlgorithm, dict[str, Any] | None]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = [
            item.filename.replace("\\", "/")
            for item in archive.infolist()
            if not item.is_dir() and Path(item.filename).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        ]
    if not members:
        raise ValueError("image ZIP contains no supported image files")
    by_class: dict[str, list[str]] = defaultdict(list)
    for member in members:
        path = Path(member)
        if len(path.parts) < 2:
            raise ValueError("image dataset must use class folders: class/image.png")
        by_class[path.parts[-2]].append(member)
    if len(by_class) < 2:
        raise ValueError("image dataset must contain at least two classes")
    split_paths: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for index, class_name in enumerate(sorted(by_class)):
        items = by_class[class_name]
        random.Random(seed + index).shuffle(items)
        train_count, validation_count, _test_count = _split_sizes(len(items))
        validation_end = train_count + validation_count
        split_paths["train"].extend(items[:train_count])
        split_paths["validation"].extend(items[train_count:validation_end])
        split_paths["test"].extend(items[validation_end:])
    split = SplitManifest(
        algorithm=SplitAlgorithm.STRATIFIED,
        random_seed=seed,
        stratification={"target_column": "class_folder"},
        train=sorted(split_paths["train"]),
        validation=sorted(split_paths["validation"]),
        test=sorted(split_paths["test"]),
    )
    profile = {
        "dataset_kind": DatasetKind.IMAGE_FOLDER.value,
        "task_type": TaskType.IMAGE_CLASSIFICATION.value,
        "row_count": len(members),
        "class_count": len(by_class),
        "classes": sorted(by_class),
        "warnings": [],
        "source_filename": filename,
    }
    return profile, split, SplitAlgorithm.STRATIFIED, {"target_column": "class_folder"}


def _random_split(items: list[str], *, seed: int) -> SplitManifest:
    values = list(items)
    random.Random(seed).shuffle(values)
    train_count, validation_count, _test_count = _split_sizes(len(values))
    train_end = train_count
    validation_end = train_count + validation_count
    return SplitManifest(
        algorithm=SplitAlgorithm.RANDOM,
        random_seed=seed,
        train=sorted(values[:train_end]),
        validation=sorted(values[train_end:validation_end]),
        test=sorted(values[validation_end:]),
    )


def _stratified_split(items: list[str], labels: list[str], *, seed: int) -> SplitManifest:
    buckets: dict[str, list[str]] = defaultdict(list)
    for item, label in zip(items, labels, strict=True):
        buckets[label].append(item)
    train: list[str] = []
    validation: list[str] = []
    test: list[str] = []
    for index, class_name in enumerate(sorted(buckets)):
        bucket = buckets[class_name]
        random.Random(seed + index).shuffle(bucket)
        train_count, validation_count, _test_count = _split_sizes(len(bucket))
        train_end = train_count
        validation_end = train_count + validation_count
        train.extend(bucket[:train_end])
        validation.extend(bucket[train_end:validation_end])
        test.extend(bucket[validation_end:])
    return SplitManifest(
        algorithm=SplitAlgorithm.STRATIFIED,
        random_seed=seed,
        train=sorted(train),
        validation=sorted(validation),
        test=sorted(test),
    )


def _split_sizes(count: int) -> tuple[int, int, int]:
    if count < 3:
        raise ValueError("each split group needs at least three samples")
    validation_count = max(1, round(count * 0.15))
    test_count = max(1, round(count * 0.15))
    train_count = count - validation_count - test_count
    if train_count < 1:
        train_count = 1
        validation_count = 1
        test_count = count - 2
    return train_count, validation_count, test_count


def _is_regression_target(values: list[str]) -> bool:
    try:
        numeric = [float(value) for value in values]
    except ValueError:
        return False
    return len(set(numeric)) > max(10, int(len(numeric) * 0.2))
