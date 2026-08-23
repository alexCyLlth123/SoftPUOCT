"""Classification and PU probability metrics used by every entry point."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if y_score is not None:
        score = np.asarray(y_score, dtype=float)
        try:
            out["pr_auc"] = float(average_precision_score(y_true, score))
        except ValueError:
            out["pr_auc"] = None
        out["brier_score"] = float(brier_score_loss(y_true, score))
    return out


def summarize_fold_metrics(rows: list[dict]) -> dict:
    """Summarize prediction metrics only; never average IDs, counts, or settings."""
    keys = (
        "accuracy", "f1",  "pr_auc", "brier_score",
    )
    summary = {}
    for key in keys:
        values = [row.get(key) for row in rows]
        values = [float(value) for value in values if value is not None]
        if not values:
            continue
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
    return summary
