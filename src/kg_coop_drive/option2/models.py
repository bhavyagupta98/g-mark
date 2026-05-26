from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    GradientBoostingRegressor = None  # type: ignore

try:
    import lightgbm as lgb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    lgb = None  # type: ignore


class RegressionBackend(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> None: ...

    def predict(self, x: np.ndarray) -> np.ndarray: ...

    def export_metadata(self) -> dict[str, object]: ...


@dataclass
class SklearnGBDTMultiOutputBackend:
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_samples_leaf: int
    subsample: float

    def __post_init__(self) -> None:
        if GradientBoostingRegressor is None:
            raise RuntimeError("scikit-learn is required for sklearn_gbdt backend")
        self._models: list[object] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._models = []
        for target_index in range(y.shape[1]):
            reg = GradientBoostingRegressor(
                n_estimators=int(self.n_estimators),
                learning_rate=float(self.learning_rate),
                max_depth=int(self.max_depth),
                min_samples_leaf=int(self.min_samples_leaf),
                subsample=float(self.subsample),
                random_state=42 + target_index,
            )
            reg.fit(x, y[:, target_index])
            self._models.append(reg)

    def predict(self, x: np.ndarray) -> np.ndarray:
        cols = [np.asarray(model.predict(x), dtype=float) for model in self._models]
        return np.column_stack(cols)

    def export_metadata(self) -> dict[str, object]:
        return {
            "backend": "sklearn_gbdt",
            "n_estimators": int(self.n_estimators),
            "learning_rate": float(self.learning_rate),
            "max_depth": int(self.max_depth),
            "min_samples_leaf": int(self.min_samples_leaf),
            "subsample": float(self.subsample),
        }


@dataclass
class LightGBMMultiOutputBackend:
    num_leaves: int
    learning_rate: float
    n_estimators: int
    min_data_in_leaf: int

    def __post_init__(self) -> None:
        if lgb is None:
            raise RuntimeError("lightgbm is required for lightgbm backend")
        self._models: list[object] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._models = []
        for target_index in range(y.shape[1]):
            reg = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=int(self.n_estimators),
                learning_rate=float(self.learning_rate),
                num_leaves=int(self.num_leaves),
                min_data_in_leaf=int(self.min_data_in_leaf),
                random_state=42 + target_index,
                n_jobs=-1,
            )
            reg.fit(x, y[:, target_index])
            self._models.append(reg)

    def predict(self, x: np.ndarray) -> np.ndarray:
        cols = [np.asarray(model.predict(x), dtype=float) for model in self._models]
        return np.column_stack(cols)

    def export_metadata(self) -> dict[str, object]:
        return {
            "backend": "lightgbm",
            "num_leaves": int(self.num_leaves),
            "learning_rate": float(self.learning_rate),
            "n_estimators": int(self.n_estimators),
            "min_data_in_leaf": int(self.min_data_in_leaf),
        }


def build_regression_backend(
    *,
    backend: str,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_samples_leaf: int,
    subsample: float,
    num_leaves: int,
) -> RegressionBackend:
    if backend == "lightgbm":
        return LightGBMMultiOutputBackend(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_data_in_leaf=min_samples_leaf,
        )
    if backend == "sklearn_gbdt":
        return SklearnGBDTMultiOutputBackend(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
        )
    if backend == "auto":
        if lgb is not None:
            return LightGBMMultiOutputBackend(
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                min_data_in_leaf=min_samples_leaf,
            )
        return SklearnGBDTMultiOutputBackend(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
        )
    raise ValueError(f"Unsupported backend: {backend}")
