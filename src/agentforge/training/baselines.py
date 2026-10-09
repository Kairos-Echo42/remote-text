from __future__ import annotations

from dataclasses import dataclass

from agentforge.training.types import (
    BaselineStrategySpec,
    DatasetKind,
    ObjectiveDirection,
    TaskType,
)


@dataclass(frozen=True, slots=True)
class BaselineStrategy:
    spec: BaselineStrategySpec

    def supports(self, *, dataset_kind: DatasetKind, task_type: TaskType) -> bool:
        return self.spec.dataset_kind == dataset_kind and self.spec.task_type == task_type

    def build_recipe(self) -> str:
        return self.spec.recipe_id

    def explain(self) -> str:
        return self.spec.explain_template


class BaselineRegistry:
    """Trusted baseline registry; specialized strategies can only be registered at platform bootstrap."""

    def __init__(self, *, allow_mutation: bool = False):
        self._strategies: dict[str, BaselineStrategy] = {}
        self._allow_mutation = allow_mutation
        self._frozen = False

    def register(self, strategy: BaselineStrategy) -> None:
        if not self._allow_mutation or self._frozen:
            raise RuntimeError("BaselineRegistry is immutable")
        key = strategy.spec.qualified_id
        if key in self._strategies:
            raise ValueError(f"baseline strategy {key!r} already registered")
        self._strategies[key] = strategy

    def freeze(self) -> BaselineRegistry:
        self._frozen = True
        return self

    def get(self, strategy_id: str) -> BaselineStrategy:
        try:
            return self._strategies[strategy_id]
        except KeyError as exc:
            raise ValueError(f"baseline strategy {strategy_id!r} is not registered") from exc

    def resolve(self, *, dataset_kind: DatasetKind, task_type: TaskType) -> BaselineStrategy:
        candidates = [
            strategy
            for strategy in self._strategies.values()
            if strategy.supports(dataset_kind=dataset_kind, task_type=task_type)
        ]
        if not candidates:
            raise ValueError(f"no baseline strategy for dataset={dataset_kind.value}, task={task_type.value}")
        return sorted(candidates, key=lambda item: item.spec.qualified_id)[0]

    def list(self) -> list[BaselineStrategy]:
        return [self._strategies[key] for key in sorted(self._strategies)]


def build_baseline_registry() -> BaselineRegistry:
    registry = BaselineRegistry(allow_mutation=True)
    registry.register(
        BaselineStrategy(
            BaselineStrategySpec(
                strategy_id="baseline.tabular.classification.dummy.prior",
                display_name="Dummy Classifier Prior",
                dataset_kind=DatasetKind.TABULAR,
                task_type=TaskType.TABULAR_CLASSIFICATION,
                recipe_id="baseline.tabular.classification.dummy@1.0.0",
                explain_template="Predicts training class priors without using feature information.",
            )
        )
    )
    registry.register(
        BaselineStrategy(
            BaselineStrategySpec(
                strategy_id="baseline.tabular.regression.dummy.mean",
                display_name="Dummy Regressor Mean",
                dataset_kind=DatasetKind.TABULAR,
                task_type=TaskType.TABULAR_REGRESSION,
                recipe_id="baseline.tabular.regression.dummy@1.0.0",
                explain_template="Predicts the training target mean without using feature information.",
            )
        )
    )
    registry.register(
        BaselineStrategy(
            BaselineStrategySpec(
                strategy_id="baseline.image.classification.prior",
                display_name="Image Prior Baseline",
                dataset_kind=DatasetKind.IMAGE_FOLDER,
                task_type=TaskType.IMAGE_CLASSIFICATION,
                recipe_id="baseline.image.classification.prior@1.0.0",
                explain_template="Predicts training class priors for image classification.",
            )
        )
    )
    return registry.freeze()


def default_direction(task_type: TaskType) -> ObjectiveDirection:
    if task_type == TaskType.TABULAR_REGRESSION:
        return ObjectiveDirection.MINIMIZE
    return ObjectiveDirection.MAXIMIZE


def default_metric(task_type: TaskType) -> str:
    if task_type == TaskType.TABULAR_REGRESSION:
        return "rmse"
    if task_type == TaskType.TABULAR_CLASSIFICATION:
        return "f1_macro"
    return "accuracy"
