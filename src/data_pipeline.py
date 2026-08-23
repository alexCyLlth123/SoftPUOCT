
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.config import DEFAULT_CONFIG, ExperimentConfig, dataset_path, min_samples_5pct


DROP_COLUMNS = {
    "breast_cancer_diagnostic": ("X1",),
    "ionosphere": ("X2",),
}
CATEGORICAL_COLUMNS = {"wholesale": ("X1",)}


def load_dataset(dataset: str, data_dir: Path, target_col: str = "target") -> tuple[pd.DataFrame, np.ndarray]:
    path = dataset_path(dataset, data_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"{path.name} does not contain target column {target_col!r}")
    if df.isna().any().any():
        raise ValueError(f"{path.name} contains missing values")
    y_raw = df[target_col].to_numpy()
    labels = np.sort(pd.unique(y_raw))
    if len(labels) != 2:
        raise ValueError(f"{path.name} must have exactly two target values; found {labels.tolist()}")
    y = (y_raw == labels[-1]).astype(np.int8)
    X = df.drop(columns=[target_col]).copy()
    for col in DROP_COLUMNS.get(dataset, ()):
        if col not in X.columns:
            raise ValueError(f"Expected column {col!r} in {path.name}")
        X = X.drop(columns=[col])
    X.index = np.arange(len(X), dtype=int)
    return X, y


def generate_pu_mask(y: np.ndarray, mask_ratio: float, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int8)
    rng = np.random.RandomState(random_state)
    positive_indices = np.flatnonzero(y == 1)
    n_mask = int(len(positive_indices) * mask_ratio)
    masked_positive_indices = (
        np.sort(rng.choice(positive_indices, size=n_mask, replace=False))
        if n_mask > 0 else np.array([], dtype=int)
    )
    pos_mask = y == 1
    pos_mask[masked_positive_indices] = False
    unlabeled_mask = ~pos_mask
    y_pu = pos_mask.astype(np.int8)
    if not np.all(pos_mask ^ unlabeled_mask):
        raise AssertionError("PU masks must be mutually exclusive and exhaustive")
    return y_pu, pos_mask, unlabeled_mask, masked_positive_indices


def _stratified_subsample(indices: np.ndarray, y: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    if ratio >= 1.0:
        return np.sort(indices.copy())
    chosen, _ = train_test_split(
        indices,
        train_size=ratio,
        stratify=y[indices],
        random_state=seed,
    )
    return np.sort(np.asarray(chosen, dtype=int))


def make_outer_splits(y: np.ndarray, config: ExperimentConfig = DEFAULT_CONFIG) -> list[dict[str, np.ndarray]]:
    skf = StratifiedKFold(
        n_splits=config.outer_folds,
        shuffle=True,
        random_state=config.random_seed,
    )
    all_indices = np.arange(len(y), dtype=int)
    splits: list[dict[str, np.ndarray]] = []
    for fold, (outer_train, outer_test) in enumerate(skf.split(all_indices, y), start=1):
        inner_train, inner_val = train_test_split(
            outer_train,
            test_size=config.inner_validation_ratio,
            stratify=y[outer_train],
            random_state=config.random_seed + 1000 + fold,
        )
        tuning = _stratified_subsample(
            np.asarray(inner_train, dtype=int), y, config.tuning_subsample_ratio,
            config.random_seed + 2000 + fold,
        )
        _, tuning_pos, tuning_unl, tuning_masked_local = generate_pu_mask(
            y[tuning], config.mask_ratio, config.random_seed + 3000 + fold
        )
        _, final_pos, final_unl, final_masked_local = generate_pu_mask(
            y[outer_train], config.mask_ratio, config.random_seed + 4000 + fold
        )
        splits.append({
            "fold": np.array([fold], dtype=int),
            "outer_train": np.sort(outer_train),
            "outer_test": np.sort(outer_test),
            "inner_train": np.sort(inner_train),
            "inner_validation": np.sort(inner_val),
            "tuning_subsample": tuning,
            "tuning_pos_mask": tuning_pos,
            "tuning_unlabeled_mask": tuning_unl,
            "tuning_masked_global": tuning[tuning_masked_local],
            "final_pos_mask": final_pos,
            "final_unlabeled_mask": final_unl,
            "final_masked_global": outer_train[final_masked_local],
        })
    return splits


def validate_outer_splits(splits: list[dict[str, np.ndarray]], y: np.ndarray) -> None:
    tests = np.concatenate([s["outer_test"] for s in splits])
    if not np.array_equal(np.sort(tests), np.arange(len(y))):
        raise AssertionError("Every sample must appear in exactly one outer test fold")
    for s in splits:
        train, test = set(s["outer_train"]), set(s["outer_test"])
        if train & test:
            raise AssertionError("Outer train/test leakage detected")
        if set(s["inner_validation"]) & test or set(s["tuning_subsample"]) & test:
            raise AssertionError("Outer test used by inner selection")


@dataclass
class FeatureRule:
    source: str
    output: str
    kind: str
    threshold: float | None = None
    category: float | str | None = None
    fallback: bool = False


class FoldPreprocessor:
    """Fit only on the current training portion and transform later partitions."""

    def __init__(self, dataset: str, flow: bool, min_samples_leaf: int, random_state: int = 42):
        self.dataset = dataset
        self.flow = bool(flow)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)
        self.rules: list[FeatureRule] = []
        self.input_columns: list[str] = []
        self.output_columns: list[str] = []

    def fit(self, X: pd.DataFrame, y_supervision: np.ndarray) -> "FoldPreprocessor":
        X = X.copy()
        y = np.asarray(y_supervision, dtype=np.int8)
        if len(X) != len(y):
            raise ValueError("X/y length mismatch in preprocessor.fit")
        self.input_columns = list(X.columns)
        self.rules = []
        categorical = set(CATEGORICAL_COLUMNS.get(self.dataset, ()))
        for col in self.input_columns:
            values = X[col].to_numpy()
            unique = np.sort(pd.unique(values))
            if len(unique) <= 1:
                self.rules.append(FeatureRule(col, col, "drop_constant"))
                continue
            if col in categorical:
                for category in unique:
                    self.rules.append(FeatureRule(col, f"{col}=={category:g}", "categorical", category=category))
                continue
            is_binary = len(unique) <= 2 and set(np.asarray(unique, dtype=float)).issubset({0.0, 1.0})
            if is_binary or not self.flow:
                self.rules.append(FeatureRule(col, col, "passthrough"))
                continue
            threshold, fallback = self._fit_threshold(values, y)
            self.rules.append(FeatureRule(col, f"{col}>{threshold:.12g}", "threshold", threshold=threshold, fallback=fallback))
        self.output_columns = [r.output for r in self.rules if r.kind != "drop_constant"]
        if not self.output_columns:
            raise ValueError("Preprocessing removed every feature")
        return self

    def _fit_threshold(self, values: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
        tree = DecisionTreeClassifier(
            criterion="entropy",
            splitter="best",
            max_depth=1,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=None,
            random_state=self.random_state,
            min_impurity_decrease=0.0,
        )
        tree.fit(np.asarray(values, dtype=float).reshape(-1, 1), y)
        threshold = float(tree.tree_.threshold[0])
        if threshold == -2.0 or not np.isfinite(threshold):
            return float(np.median(values)), True
        return threshold, False

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        missing = set(self.input_columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing transform columns: {sorted(missing)}")
        columns: list[np.ndarray] = []
        for rule in self.rules:
            values = X[rule.source].to_numpy()
            if rule.kind == "drop_constant":
                continue
            if rule.kind == "passthrough":
                out = values.astype(float)
            elif rule.kind == "categorical":
                out = (values == rule.category).astype(float)
            elif rule.kind == "threshold":
                out = (values.astype(float) > float(rule.threshold)).astype(float)
            else:
                raise RuntimeError(f"Unknown preprocessing rule {rule.kind}")
            columns.append(out)
        matrix = np.column_stack(columns).astype(float)
        if self.flow:
            assert_binary_matrix(matrix)
        return matrix

    def fit_transform(self, X: pd.DataFrame, y_supervision: np.ndarray) -> np.ndarray:
        return self.fit(X, y_supervision).transform(X)

    def metadata(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "flow": self.flow,
            "min_samples_leaf": self.min_samples_leaf,
            "input_columns": self.input_columns,
            "output_columns": self.output_columns,
            "rules": [r.__dict__.copy() for r in self.rules],
        }


def assert_binary_matrix(X: np.ndarray) -> None:
    unique = np.unique(np.asarray(X))
    if not set(unique.astype(float)).issubset({0.0, 1.0}):
        raise ValueError(f"Flow input must be strictly binary; found values {unique[:10].tolist()}")


def new_preprocessor(dataset: str, model_name: str, n_total: int, seed: int) -> FoldPreprocessor:
    return FoldPreprocessor(
        dataset=dataset,
        flow=model_name in {"flowoct", "flowpuoct"},
        min_samples_leaf=min_samples_5pct(n_total),
        random_state=seed,
    )
