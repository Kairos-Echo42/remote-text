from __future__ import annotations

import threading
from collections.abc import Iterable

from agentforge.training.types import (
    DatasetKind,
    HyperparameterSpec,
    ObjectiveDirection,
    ParameterType,
    RecipeResources,
    RecipeSpec,
    TaskType,
    TrainingJobSpec,
)


class RecipeRegistry:
    """Trusted, process-local registry. Agent capabilities never mutate it."""

    def __init__(self, *, allow_mutation: bool = False):
        self._recipes: dict[str, RecipeSpec] = {}
        self._allow_mutation = allow_mutation
        self._frozen = False

    def register(self, spec: RecipeSpec) -> None:
        if not self._allow_mutation or self._frozen:
            raise RuntimeError("RecipeRegistry is immutable; only platform bootstrap may register")
        if spec.qualified_id in self._recipes:
            raise ValueError(f"recipe {spec.qualified_id!r} already registered")
        self._recipes[spec.qualified_id] = spec

    def freeze(self) -> RecipeRegistry:
        self._frozen = True
        return self

    def get(self, qualified_id: str) -> RecipeSpec:
        try:
            return self._recipes[qualified_id]
        except KeyError as exc:
            raise ValueError(f"recipe {qualified_id!r} is not registered") from exc

    def list(self) -> list[RecipeSpec]:
        return [self._recipes[key] for key in sorted(self._recipes)]

    def validate_job(self, spec: TrainingJobSpec) -> tuple[RecipeSpec, dict]:
        recipe = self.get(spec.recipe_id)
        if spec.max_job_seconds > recipe.resources.max_job_seconds:
            raise ValueError(f"job max_job_seconds exceeds recipe limit {recipe.resources.max_job_seconds}")
        if not recipe.resources.min_epochs <= spec.max_epochs <= recipe.resources.max_epochs:
            raise ValueError(
                f"max_epochs must be between {recipe.resources.min_epochs} and {recipe.resources.max_epochs}"
            )
        if not recipe.resources.min_batch_size <= spec.batch_size <= recipe.resources.max_batch_size:
            raise ValueError(
                f"batch_size must be between {recipe.resources.min_batch_size} and {recipe.resources.max_batch_size}"
            )
        parameters = recipe.validate_hyperparameters(spec.parameters)
        return recipe, parameters


_registry: RecipeRegistry | None = None
_lock = threading.Lock()


def get_recipe_registry() -> RecipeRegistry:
    global _registry
    if _registry is None:
        with _lock:
            if _registry is None:
                _registry = build_recipe_registry()
    return _registry


def _common_training_parameters(*, learning_rate: bool = False) -> list[HyperparameterSpec]:
    parameters = [
        HyperparameterSpec(
            name="early_stopping_patience",
            type=ParameterType.INTEGER,
            default=5,
            minimum=1,
            maximum=50,
        ),
        HyperparameterSpec(
            name="validation_metric",
            type=ParameterType.CHOICE,
            default="auto",
            choices=["auto", "accuracy", "f1_macro", "rmse", "mae", "r2"],
        ),
    ]
    if learning_rate:
        parameters.extend(
            [
                HyperparameterSpec(
                    name="learning_rate",
                    type=ParameterType.NUMBER,
                    default=0.001,
                    minimum=0.000001,
                    maximum=0.1,
                ),
                HyperparameterSpec(
                    name="weight_decay",
                    type=ParameterType.NUMBER,
                    default=0.0,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ]
        )
    return parameters


def build_recipe_registry() -> RecipeRegistry:
    registry = RecipeRegistry(allow_mutation=True)
    for spec in _builtin_recipes():
        registry.register(spec)
    return registry.freeze()


def _builtin_recipes() -> Iterable[RecipeSpec]:
    yield RecipeSpec(
        recipe_id="baseline.tabular.classification.dummy",
        display_name="Dummy Classifier Prior",
        description="Predicts training class priors; mandatory explainable baseline.",
        backend="dummy",
        task_types=[TaskType.TABULAR_CLASSIFICATION],
        dataset_kinds=[DatasetKind.TABULAR],
        default_metric="accuracy",
        default_direction=ObjectiveDirection.MAXIMIZE,
        hyperparameters=[
            HyperparameterSpec(
                name="strategy",
                type=ParameterType.CHOICE,
                default="prior",
                choices=["prior", "most_frequent", "stratified", "uniform"],
            )
        ],
        resources=RecipeResources(
            default_device_policy="cpu_only",
            supports_gpu=False,
            minimum_memory_mb=128,
            recommended_memory_mb=256,
            cpu_cores=1,
            max_job_seconds=300,
        ),
    )
    yield RecipeSpec(
        recipe_id="baseline.tabular.regression.dummy",
        display_name="Dummy Regressor Mean",
        description="Predicts the training target mean; mandatory explainable baseline.",
        backend="dummy",
        task_types=[TaskType.TABULAR_REGRESSION],
        dataset_kinds=[DatasetKind.TABULAR],
        default_metric="rmse",
        default_direction=ObjectiveDirection.MINIMIZE,
        hyperparameters=[
            HyperparameterSpec(
                name="strategy",
                type=ParameterType.CHOICE,
                default="mean",
                choices=["mean", "median", "quantile", "constant"],
            ),
            HyperparameterSpec(name="constant", type=ParameterType.NUMBER, default=0.0, minimum=-1e12, maximum=1e12),
        ],
        resources=RecipeResources(
            default_device_policy="cpu_only",
            supports_gpu=False,
            minimum_memory_mb=128,
            recommended_memory_mb=256,
            cpu_cores=1,
            max_job_seconds=300,
        ),
    )
    yield RecipeSpec(
        recipe_id="baseline.image.classification.prior",
        display_name="Image Prior Baseline",
        description="Predicts class priors for image classification.",
        backend="dummy",
        task_types=[TaskType.IMAGE_CLASSIFICATION],
        dataset_kinds=[DatasetKind.IMAGE_FOLDER],
        default_metric="accuracy",
        default_direction=ObjectiveDirection.MAXIMIZE,
        hyperparameters=[],
        resources=RecipeResources(
            default_device_policy="cpu_only",
            supports_gpu=False,
            minimum_memory_mb=128,
            recommended_memory_mb=256,
            cpu_cores=1,
            max_job_seconds=300,
        ),
    )
    yield RecipeSpec(
        recipe_id="tabular.classification.logistic",
        display_name="Logistic Regression",
        description="Regularized linear classification baseline.",
        backend="sklearn",
        task_types=[TaskType.TABULAR_CLASSIFICATION],
        dataset_kinds=[DatasetKind.TABULAR],
        default_metric="f1_macro",
        default_direction=ObjectiveDirection.MAXIMIZE,
        hyperparameters=[
            HyperparameterSpec(name="C", type=ParameterType.NUMBER, default=1.0, minimum=0.0001, maximum=1000.0),
            HyperparameterSpec(
                name="class_weight",
                type=ParameterType.CHOICE,
                default=None,
                choices=[None, "balanced"],
                nullable=True,
            ),
        ],
        resources=RecipeResources(
            default_device_policy="cpu_only",
            supports_gpu=False,
            minimum_memory_mb=256,
            recommended_memory_mb=512,
            cpu_cores=2,
            max_job_seconds=600,
        ),
    )
    yield RecipeSpec(
        recipe_id="tabular.classification.random_forest",
        display_name="Random Forest Classifier",
        description="Non-linear tree ensemble baseline.",
        backend="sklearn",
        task_types=[TaskType.TABULAR_CLASSIFICATION],
        dataset_kinds=[DatasetKind.TABULAR],
        default_metric="f1_macro",
        default_direction=ObjectiveDirection.MAXIMIZE,
        hyperparameters=[
            HyperparameterSpec(
                name="n_estimators",
                type=ParameterType.INTEGER,
                default=200,
                minimum=10,
                maximum=2000,
            ),
            HyperparameterSpec(
                name="max_depth",
                type=ParameterType.INTEGER,
                default=20,
                minimum=1,
                maximum=100,
                nullable=True,
            ),
            HyperparameterSpec(
                name="min_samples_leaf",
                type=ParameterType.INTEGER,
                default=1,
                minimum=1,
                maximum=100,
            ),
        ],
        resources=RecipeResources(
            default_device_policy="cpu_only",
            supports_gpu=False,
            minimum_memory_mb=512,
            recommended_memory_mb=2048,
            cpu_cores=4,
            max_job_seconds=900,
        ),
    )
    yield RecipeSpec(
        recipe_id="tabular.classification.mlp",
        display_name="PyTorch Tabular MLP",
        description="Feed-forward neural network for tabular classification.",
        backend="pytorch",
        task_types=[TaskType.TABULAR_CLASSIFICATION],
        dataset_kinds=[DatasetKind.TABULAR],
        default_metric="f1_macro",
        default_direction=ObjectiveDirection.MAXIMIZE,
        hyperparameters=[
            HyperparameterSpec(
                name="hidden_sizes",
                type=ParameterType.CHOICE,
                default=[128, 64],
                choices=[[64], [128, 64], [256, 128], [256, 128, 64]],
            ),
            HyperparameterSpec(name="dropout", type=ParameterType.NUMBER, default=0.1, minimum=0.0, maximum=0.7),
            *_common_training_parameters(learning_rate=True),
        ],
        resources=RecipeResources(
            default_device_policy="preferred_gpu",
            minimum_memory_mb=1024,
            recommended_memory_mb=3072,
            cpu_cores=2,
            max_job_seconds=900,
        ),
    )
    yield RecipeSpec(
        recipe_id="tabular.regression.ridge",
        display_name="Ridge Regression",
        description="Regularized linear regression baseline.",
        backend="sklearn",
        task_types=[TaskType.TABULAR_REGRESSION],
        dataset_kinds=[DatasetKind.TABULAR],
        default_metric="rmse",
        default_direction=ObjectiveDirection.MINIMIZE,
        hyperparameters=[
            HyperparameterSpec(name="alpha", type=ParameterType.NUMBER, default=1.0, minimum=0.0, maximum=10000.0)
        ],
        resources=RecipeResources(
            default_device_policy="cpu_only",
            supports_gpu=False,
            minimum_memory_mb=256,
            recommended_memory_mb=512,
            cpu_cores=2,
            max_job_seconds=600,
        ),
    )
    yield RecipeSpec(
        recipe_id="tabular.regression.random_forest",
        display_name="Random Forest Regressor",
        description="Non-linear tree ensemble regression baseline.",
        backend="sklearn",
        task_types=[TaskType.TABULAR_REGRESSION],
        dataset_kinds=[DatasetKind.TABULAR],
        default_metric="rmse",
        default_direction=ObjectiveDirection.MINIMIZE,
        hyperparameters=[
            HyperparameterSpec(
                name="n_estimators",
                type=ParameterType.INTEGER,
                default=200,
                minimum=10,
                maximum=2000,
            ),
            HyperparameterSpec(
                name="max_depth",
                type=ParameterType.INTEGER,
                default=20,
                minimum=1,
                maximum=100,
                nullable=True,
            ),
        ],
        resources=RecipeResources(
            default_device_policy="cpu_only",
            supports_gpu=False,
            minimum_memory_mb=512,
            recommended_memory_mb=2048,
            cpu_cores=4,
            max_job_seconds=900,
        ),
    )
    yield RecipeSpec(
        recipe_id="tabular.regression.mlp",
        display_name="PyTorch Tabular MLP Regressor",
        description="Feed-forward neural network for tabular regression.",
        backend="pytorch",
        task_types=[TaskType.TABULAR_REGRESSION],
        dataset_kinds=[DatasetKind.TABULAR],
        default_metric="rmse",
        default_direction=ObjectiveDirection.MINIMIZE,
        hyperparameters=[
            HyperparameterSpec(
                name="hidden_sizes",
                type=ParameterType.CHOICE,
                default=[128, 64],
                choices=[[64], [128, 64], [256, 128], [256, 128, 64]],
            ),
            HyperparameterSpec(name="dropout", type=ParameterType.NUMBER, default=0.1, minimum=0.0, maximum=0.7),
            *_common_training_parameters(learning_rate=True),
        ],
        resources=RecipeResources(
            default_device_policy="preferred_gpu",
            minimum_memory_mb=1024,
            recommended_memory_mb=3072,
            cpu_cores=2,
            max_job_seconds=900,
        ),
    )
    yield RecipeSpec(
        recipe_id="image.classification.cnn",
        display_name="Small Image CNN",
        description="From-scratch convolutional network for small image classification datasets.",
        backend="pytorch",
        task_types=[TaskType.IMAGE_CLASSIFICATION],
        dataset_kinds=[DatasetKind.IMAGE_FOLDER],
        default_metric="f1_macro",
        default_direction=ObjectiveDirection.MAXIMIZE,
        hyperparameters=[
            HyperparameterSpec(
                name="channels",
                type=ParameterType.CHOICE,
                default=[32, 64],
                choices=[[16, 32], [32, 64], [32, 64, 128]],
            ),
            HyperparameterSpec(name="dropout", type=ParameterType.NUMBER, default=0.2, minimum=0.0, maximum=0.7),
            *_common_training_parameters(learning_rate=True),
        ],
        resources=RecipeResources(
            default_device_policy="preferred_gpu",
            minimum_memory_mb=2048,
            recommended_memory_mb=4096,
            cpu_cores=4,
            max_job_seconds=900,
        ),
    )
