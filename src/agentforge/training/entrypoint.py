from __future__ import annotations

import json
import random
import signal
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

_STOP_REQUESTED = False


def _handle_signal(signum, frame):
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _write_event(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _record_metric(
    metrics_path: Path, name: str, value: float, split: str, step: int, epoch: int | None = None
) -> None:
    _write_event(
        metrics_path,
        {
            "name": name,
            "value": float(value),
            "split": split,
            "step": step,
            "epoch": epoch,
        },
    )


def _safe_extract(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise ValueError(f"unsafe archive path: {member.filename}")
        archive.extractall(destination)


def _environment_manifest(output_dir: Path) -> dict[str, Any]:
    import platform

    versions: dict[str, Any] = {"python_version": platform.python_version()}
    hardware: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "machine": platform.machine(),
    }
    try:
        import torch

        versions["pytorch_version"] = torch.__version__
        versions["cuda_version"] = torch.version.cuda
        versions["cudnn_version"] = str(torch.backends.cudnn.version())
        hardware["cuda_device_count"] = torch.cuda.device_count()
        hardware["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    except Exception:
        versions["pytorch_version"] = None
    try:
        import sklearn

        versions["sklearn_version"] = sklearn.__version__
    except Exception:
        versions["sklearn_version"] = None
    versions["hardware"] = hardware
    (output_dir / "environment.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    return versions


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _classification_metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _metric_for_recipe(metrics: dict[str, float], recipe_id: str, objective_metric: str | None) -> str:
    if objective_metric and objective_metric in metrics:
        return objective_metric
    if "rmse" in metrics:
        return "rmse"
    if recipe_id.endswith("logistic") or recipe_id.endswith("random_forest") or recipe_id.endswith("mlp"):
        return "f1_macro"
    return "accuracy"


def _save_checkpoint(path: Path, model: Any, metadata: dict[str, Any]) -> None:
    import torch

    torch.save({"model": model, "metadata": metadata, "resume_supported": False}, path)
    (path.parent / "checkpoint-metadata.json").write_text(
        json.dumps({**metadata, "resume_supported": False}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_tabular(spec: dict[str, Any], *, include_test: bool = True):
    import numpy as np
    import pandas as pd

    dataset_path = Path(spec["dataset_path"])
    if dataset_path.suffix.lower() == ".json":
        frame = pd.read_json(dataset_path)
    else:
        frame = pd.read_csv(dataset_path)
    dataset_version = spec.get("dataset_version", {})
    target = spec.get("target_column") or dataset_version.get("target_column")
    if not target or target not in frame.columns:
        raise ValueError("target_column is required for tabular training")
    manifest = dataset_version.get("split_manifest") or {}
    index_frame = frame.reset_index(drop=True)
    index_frame["_agentforge_row_id"] = [str(index) for index in range(len(index_frame))]
    if manifest.get("train") and manifest.get("validation"):
        train = index_frame[index_frame["_agentforge_row_id"].isin(manifest["train"])]
        validation = index_frame[index_frame["_agentforge_row_id"].isin(manifest["validation"])]
        test = (
            index_frame[index_frame["_agentforge_row_id"].isin(manifest.get("test", []))]
            if include_test
            else index_frame.iloc[0:0]
        )
    else:
        from sklearn.model_selection import train_test_split

        train, remainder = train_test_split(index_frame, test_size=0.3, random_state=int(spec.get("random_seed", 42)))
        if include_test:
            validation, test = train_test_split(
                remainder,
                test_size=0.5,
                random_state=int(spec.get("random_seed", 42)),
            )
        else:
            validation = remainder
            test = index_frame.iloc[0:0]
    x_train = train.drop(columns=[target, "_agentforge_row_id"])
    y_train = train[target]
    x_validation = validation.drop(columns=[target, "_agentforge_row_id"])
    y_validation = validation[target]
    x_test = test.drop(columns=[target, "_agentforge_row_id"])
    y_test = test[target]
    return (x_train, y_train, x_validation, y_validation, x_test, y_test, list(x_train.columns), np)


def _build_preprocessor(x_train):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = list(x_train.select_dtypes(include=["number", "bool"]).columns)
    categorical = [column for column in x_train.columns if column not in numeric]
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def _run_tabular(spec: dict[str, Any], output_dir: Path, metrics_path: Path) -> None:
    import joblib
    import numpy as np
    import torch
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    recipe = spec["recipe_id"]
    params = dict(spec.get("parameters", {}))
    params["max_epochs"] = int(spec.get("max_epochs", 30))
    params["batch_size"] = int(spec.get("batch_size", 32))
    params["device_policy"] = str(spec.get("device_policy", "preferred_gpu"))
    params["random_seed"] = int(spec.get("random_seed", 42))
    x_train, y_train, x_validation, y_validation, _x_test, _y_test, feature_names, _ = _load_tabular(
        spec,
        include_test=False,
    )
    is_regression = "regression" in recipe
    preprocessor = _build_preprocessor(x_train)
    train_matrix = preprocessor.fit_transform(x_train)
    validation_matrix = preprocessor.transform(x_validation)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    if recipe.startswith("baseline.tabular.classification.dummy"):
        model = DummyClassifier(strategy=params.get("strategy", "prior"), random_state=spec.get("random_seed", 42))
    elif recipe.startswith("baseline.tabular.regression.dummy"):
        model = DummyRegressor(strategy=params.get("strategy", "mean"), constant=params.get("constant", 0.0))
    elif recipe.endswith("logistic"):
        model = LogisticRegression(C=params.get("C", 1.0), class_weight=params.get("class_weight"), max_iter=2000)
    elif recipe.endswith("random_forest") and is_regression:
        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth"),
            random_state=spec.get("random_seed", 42),
            n_jobs=max(1, int(params.get("cpu_cores", 2))),
        )
    elif recipe.endswith("random_forest"):
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth"),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=spec.get("random_seed", 42),
            n_jobs=max(1, int(params.get("cpu_cores", 2))),
        )
    elif recipe.endswith("ridge"):
        model = Ridge(alpha=params.get("alpha", 1.0))
    elif recipe.endswith("mlp"):
        model, torch_metadata = _train_torch_mlp(
            train_matrix,
            np.asarray(y_train),
            validation_matrix,
            np.asarray(y_validation),
            params,
            output_dir,
            metrics_path,
            regression=is_regression,
            stop_requested=lambda: _STOP_REQUESTED,
        )
    else:
        raise ValueError(f"unsupported tabular recipe: {recipe}")

    if not recipe.endswith("mlp"):
        _save_checkpoint(
            output_dir / "checkpoint.pt",
            model,
            {"recipe": recipe, "state": "before_fit", "resume_supported": False},
        )
        model.fit(train_matrix, y_train)
        if _STOP_REQUESTED:
            _save_checkpoint(
                output_dir / "checkpoint.pt",
                model,
                {"recipe": recipe, "state": "after_fit_cancelled", "resume_supported": False},
            )
            raise SystemExit(143)
    if recipe.endswith("mlp"):
        model.eval()
        with torch.no_grad():
            validation_output = model(torch.tensor(validation_matrix, dtype=torch.float32))
        if is_regression:
            validation_predictions = validation_output.detach().cpu().numpy().reshape(-1)
        else:
            encoded = validation_output.argmax(dim=1).detach().cpu().tolist()
            inverse = {
                int(index): label
                for label, index in (torch_metadata.get("label_mapping") or {}).items()
            }
            validation_predictions = [inverse[int(item)] for item in encoded]
    else:
        validation_predictions = model.predict(validation_matrix)
    if is_regression:
        validation_metrics = _regression_metrics(y_validation, validation_predictions)
    else:
        validation_metrics = _classification_metrics(y_validation, validation_predictions)
    for step, (name, value) in enumerate(validation_metrics.items()):
        _record_metric(metrics_path, name, value, "validation", step)
    if recipe.endswith("mlp"):
        torch.save(
            {
                "format": "tabular_mlp",
                "state_dict": model.state_dict(),
                **torch_metadata,
            },
            output_dir / "model.pt",
        )
    else:
        torch.save(model, output_dir / "model.pt")
    _save_checkpoint(
        output_dir / "checkpoint.pt",
        model,
        {"recipe": recipe, "validation": validation_metrics, "resume_supported": False},
    )
    (output_dir / "metrics.json").write_text(
        json.dumps({"validation": validation_metrics}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "schema.json").write_text(
        json.dumps({"features": feature_names, "target": spec.get("target_column")}, indent=2),
        encoding="utf-8",
    )


def _train_torch_mlp(
    x_train,
    y_train,
    x_validation,
    y_validation,
    params: dict,
    output_dir: Path,
    metrics_path: Path,
    *,
    regression: bool,
    stop_requested,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    import torch
    from torch import nn

    device_policy = str(params.get("device_policy", "preferred_gpu"))
    use_cuda = torch.cuda.is_available() and device_policy != "cpu_only"
    if device_policy == "required_gpu" and not use_cuda:
        raise RuntimeError("required_gpu requested but CUDA is unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    hidden_sizes = [int(size) for size in params.get("hidden_sizes", [128, 64])]
    dropout = float(params.get("dropout", 0.1))
    input_size = int(x_train.shape[1])
    output_size = 1 if regression else len(np.unique(y_train))
    layers: list[nn.Module] = []
    previous = input_size
    for size in hidden_sizes:
        layers.extend([nn.Linear(previous, size), nn.ReLU(), nn.Dropout(dropout)])
        previous = size
    layers.append(nn.Linear(previous, output_size))
    model = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params.get("learning_rate", 0.001)),
        weight_decay=float(params.get("weight_decay", 0.0)),
    )
    criterion = nn.MSELoss() if regression else nn.CrossEntropyLoss()
    x_train_tensor = torch.tensor(np.asarray(x_train), dtype=torch.float32)
    x_validation_tensor = torch.tensor(np.asarray(x_validation), dtype=torch.float32).to(device)
    if regression:
        y_train_tensor = torch.tensor(np.asarray(y_train), dtype=torch.float32).reshape(-1, 1)
        y_validation_tensor = torch.tensor(np.asarray(y_validation), dtype=torch.float32).reshape(-1, 1).to(device)
    else:
        classes = {label: index for index, label in enumerate(sorted(np.unique(y_train)))}
        labels = np.asarray([classes[label] for label in y_train])
        y_train_tensor = torch.tensor(labels, dtype=torch.long)
        y_validation_tensor = torch.tensor([classes.get(label, 0) for label in y_validation], dtype=torch.long).to(
            device
        )
    batch_size = max(1, int(params.get("batch_size", 32)))
    best_loss = float("inf")
    patience = int(params.get("early_stopping_patience", 5))
    stale = 0
    for epoch in range(int(params.get("max_epochs", 30))):
        if stop_requested():
            _save_checkpoint(output_dir / "checkpoint.pt", model, {"epoch": epoch, "resume_supported": False})
            raise SystemExit(143)
        model.train()
        permutation = torch.randperm(len(x_train_tensor))
        total = 0.0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = x_train_tensor[indices].to(device)
            batch_y = y_train_tensor[indices].to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            if regression:
                loss = criterion(output, batch_y)
            else:
                loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(indices)
        model.eval()
        with torch.no_grad():
            validation_output = model(x_validation_tensor)
            validation_loss = float(criterion(validation_output, y_validation_tensor).item())
        _record_metric(metrics_path, "train_loss", total / max(1, len(permutation)), "train", epoch, epoch)
        _record_metric(metrics_path, "validation_loss", validation_loss, "validation", epoch, epoch)
        if validation_loss < best_loss:
            best_loss = validation_loss
            stale = 0
            _save_checkpoint(
                output_dir / "checkpoint.pt",
                model,
                {"epoch": epoch, "validation_loss": validation_loss, "resume_supported": False},
            )
        else:
            stale += 1
            if stale >= patience:
                break
    label_mapping = None
    if not regression:
        label_mapping = {str(label): index for label, index in classes.items()}
    return model, {
        "input_size": input_size,
        "hidden_sizes": hidden_sizes,
        "output_size": output_size,
        "dropout": dropout,
        "regression": regression,
        "label_mapping": label_mapping,
    }


def _run_image(spec: dict[str, Any], output_dir: Path, metrics_path: Path) -> None:
    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    dataset_path = Path(spec["dataset_path"])
    root = Path(tempfile.mkdtemp(prefix="agentforge-images-"))
    if dataset_path.suffix.lower() == ".zip":
        _safe_extract(dataset_path, root)
    else:
        root = dataset_path
    manifest = spec.get("dataset_version", {}).get("split_manifest", {})
    if not manifest.get("train"):
        from torchvision.datasets import ImageFolder

        dataset = ImageFolder(root)
        classes = dataset.classes
        samples = [path for path, _ in dataset.samples]
        indices = list(range(len(samples)))
        from sklearn.model_selection import train_test_split

        train_indices, remainder = train_test_split(
            indices, test_size=0.3, random_state=int(spec.get("random_seed", 42))
        )
        validation_indices, _test_indices = train_test_split(
            remainder, test_size=0.5, random_state=int(spec.get("random_seed", 42))
        )
        train_paths = [samples[index] for index in train_indices]
        validation_paths = [samples[index] for index in validation_indices]
        labels = [dataset.classes[dataset.samples[index][1]] for index in range(len(samples))]
        sample_labels = dict(zip(samples, labels, strict=True))
    else:
        classes = sorted({Path(item).parent.name for item in manifest["train"]})
        sample_labels = {
            item: Path(item).parent.name for item in manifest["train"] + manifest["validation"]
        }
        train_paths = [root / item for item in manifest["train"]]
        validation_paths = [root / item for item in manifest["validation"]]

    class ImageList(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            path = self.paths[index]
            image = Image.open(path).convert("RGB")
            label = classes.index(
                sample_labels[str(path).replace("\\\\", "/")]
                if str(path).replace("\\\\", "/") in sample_labels
                else Path(path).parent.name
            )
            return self.transform(image), label

    train_transform = _image_transform(augment=True)
    eval_transform = _image_transform(augment=False)
    train_dataset = ImageList(train_paths, train_transform)
    validation_dataset = ImageList(validation_paths, eval_transform)
    batch_size = max(1, int(spec.get("batch_size", 32)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device_policy = spec.get("device_policy", "preferred_gpu")
    use_cuda = torch.cuda.is_available() and device_policy != "cpu_only"
    if device_policy == "required_gpu" and not use_cuda:
        raise RuntimeError("required_gpu requested but CUDA is unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    channels = [int(value) for value in spec.get("parameters", {}).get("channels", [32, 64])]
    model = _cnn_model(len(classes), channels, float(spec.get("parameters", {}).get("dropout", 0.2))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec.get("parameters", {}).get("learning_rate", 0.001)))
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    for epoch in range(int(spec.get("max_epochs", 20))):
        if _STOP_REQUESTED:
            _save_checkpoint(output_dir / "checkpoint.pt", model, {"epoch": epoch, "resume_supported": False})
            raise SystemExit(143)
        model.train()
        for images, labels_batch in train_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels_batch)
            loss.backward()
            optimizer.step()
        validation_labels, validation_predictions = _predict_image(model, validation_loader, device)
        metrics = _classification_metrics(validation_labels, validation_predictions)
        for name, value in metrics.items():
            _record_metric(metrics_path, name, value, "validation", epoch, epoch)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            _save_checkpoint(
                output_dir / "checkpoint.pt",
                model,
                {"epoch": epoch, "validation": metrics, "resume_supported": False},
            )
    torch.save(model, output_dir / "model.pt")
    (output_dir / "preprocessor.joblib").write_bytes(b"image-transform-v1")
    (output_dir / "metrics.json").write_text(
        json.dumps({"validation": {"accuracy": best_accuracy}}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "schema.json").write_text(
        json.dumps({"classes": classes, "input": "RGB image"}, indent=2),
        encoding="utf-8",
    )


def _image_transform(*, augment: bool):
    from torchvision import transforms

    operations = []
    if augment:
        operations.extend(
            [
                transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
            ]
        )
    else:
        operations.append(transforms.Resize((32, 32)))
    operations.extend([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transforms.Compose(operations)


def _cnn_model(class_count: int, channels: list[int], dropout: float):
    from torch import nn

    layers: list[nn.Module] = []
    in_channels = 3
    for channel in channels:
        layers.extend(
            [
                nn.Conv2d(in_channels, channel, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
        )
        in_channels = channel
    layers.extend(
        [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_channels, class_count)]
    )
    return nn.Sequential(*layers)


def _predict_image(model, loader, device):
    import torch

    labels: list[int] = []
    predictions: list[int] = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            output = model(images.to(device)).argmax(dim=1).cpu().tolist()
            predictions.extend(output)
            labels.extend(batch_labels.tolist())
    return labels, predictions


def _run_final_evaluation(spec: dict[str, Any], output_dir: Path, metrics_path: Path) -> None:
    import joblib
    import torch

    source_dir = Path(spec.get("source_path", "/source"))
    loaded = torch.load(source_dir / "model.pt", map_location="cpu", weights_only=False)
    dataset_kind = spec.get("dataset_version", {}).get("kind")
    if dataset_kind == "image_folder":
        _evaluate_image_final(spec, source_dir, loaded, metrics_path, output_dir)
        return
    preprocessor = joblib.load(source_dir / "preprocessor.joblib")
    x_train, y_train, x_validation, y_validation, x_test, y_test, _, _ = _load_tabular(
        spec,
        include_test=True,
    )
    del x_train, y_train, x_validation, y_validation
    test_matrix = preprocessor.transform(x_test)
    if isinstance(loaded, dict) and loaded.get("format") == "tabular_mlp":
        model = _restore_tabular_mlp(loaded)
        model.eval()
        with torch.no_grad():
            output = model(torch.tensor(test_matrix, dtype=torch.float32))
        if loaded.get("regression"):
            predictions = output.detach().cpu().numpy().reshape(-1)
        else:
            encoded = output.argmax(dim=1).detach().cpu().tolist()
            inverse = {int(index): label for label, index in (loaded.get("label_mapping") or {}).items()}
            predictions = [inverse[int(item)] for item in encoded]
    else:
        predictions = loaded.predict(test_matrix)
    metrics = (
        _regression_metrics(y_test, predictions)
        if "regression" in spec["recipe_id"]
        else _classification_metrics(y_test, predictions)
    )
    for step, (name, value) in enumerate(metrics.items()):
        _record_metric(metrics_path, name, value, "test", step)
    (output_dir / "metrics.json").write_text(json.dumps({"test": metrics}, indent=2), encoding="utf-8")


def _restore_tabular_mlp(payload: dict[str, Any]):
    from torch import nn

    hidden_sizes = [int(size) for size in payload["hidden_sizes"]]
    input_size = int(payload["input_size"])
    output_size = int(payload["output_size"])
    dropout = float(payload.get("dropout", 0.1))
    layers: list[nn.Module] = []
    previous = input_size
    for size in hidden_sizes:
        layers.extend([nn.Linear(previous, size), nn.ReLU(), nn.Dropout(dropout)])
        previous = size
    layers.append(nn.Linear(previous, output_size))
    model = nn.Sequential(*layers)
    model.load_state_dict(payload["state_dict"])
    return model


def _evaluate_image_final(spec: dict, source_dir: Path, model, metrics_path: Path, output_dir: Path) -> None:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    del source_dir
    dataset_path = Path(spec["dataset_path"])
    root = Path(tempfile.mkdtemp(prefix="agentforge-images-final-"))
    if dataset_path.suffix.lower() == ".zip":
        _safe_extract(dataset_path, root)
    else:
        root = dataset_path
    manifest = spec.get("dataset_version", {}).get("split_manifest", {})
    test_items = manifest.get("test", [])
    classes = sorted(
        {Path(item).parent.name for item in manifest.get("train", []) + manifest.get("validation", []) + test_items}
    )

    class ImageList(Dataset):
        def __len__(self):
            return len(test_items)

        def __getitem__(self, index):
            item = test_items[index]
            image = Image.open(root / item).convert("RGB")
            label = classes.index(Path(item).parent.name)
            return _image_transform(augment=False)(image), label

    loader = DataLoader(ImageList(), batch_size=int(spec.get("batch_size", 32)), shuffle=False, num_workers=0)
    labels, predictions = _predict_image(model, loader, torch.device("cpu"))
    metrics = _classification_metrics(labels, predictions)
    for step, (name, value) in enumerate(metrics.items()):
        _record_metric(metrics_path, name, value, "test", step)
    (output_dir / "metrics.json").write_text(json.dumps({"test": metrics}, indent=2), encoding="utf-8")


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    spec_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/job.json")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    _seed_everything(int(spec.get("random_seed", 42)))
    output_dir = Path(spec.get("output_dir", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.ndjson"
    metrics_path.write_text("", encoding="utf-8")
    environment = _environment_manifest(output_dir)
    _record_metric(metrics_path, "gpu_available", float(bool(environment.get("cuda_version"))), "system", 0)
    try:
        if spec.get("evaluation_only"):
            _run_final_evaluation(spec, output_dir, metrics_path)
        else:
            dataset_kind = spec.get("dataset_version", {}).get("kind")
            if dataset_kind == "image_folder":
                _run_image(spec, output_dir, metrics_path)
            else:
                _run_tabular(spec, output_dir, metrics_path)
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:
        _write_event(metrics_path, {"event": "failure", "error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
