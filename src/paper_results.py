from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, PU_MODELS
from src.data_pipeline import load_dataset
from src.metrics import classification_metrics
from src.result_io import atomic_csv, exclusive_file_lock, is_success, read_json


PREDICTIVE_METRICS = (
    "accuracy", "recall", "precision", "f1", "specificity",
    "balanced_accuracy", "roc_auc", "pr_auc", "brier_score",
)
OOF_METRICS = (*PREDICTIVE_METRICS, "tp", "fp", "fn", "tn")

FOLD_COLUMNS = (
    "dataset", "model", "depth", "fold", "best_alpha", "search_strategy",
    "tuning_time_limit_seconds_per_optimize", "final_time_limit_seconds_per_optimize",
    "n_outer_train",
    "n_outer_test", "final_attempt", "result_schema_version",
    *PREDICTIVE_METRICS, "tp", "fp", "fn", "tn",
    "final_training_wall_time_s", "tuning_wall_time_s", "end_to_end_wall_time_s",
    "final_training_solver_time_s", "final_iteration_solver_time_s",
    "tuning_solver_time_s", "end_to_end_solver_time_s", "solver_status",
    "solver_status_name", "solver_solve_count", "solution_count",
    "objective_value", "objective_bound", "objective_sense", "final_mip_gap",
    "worst_iteration_mip_gap", "time_to_first_incumbent_s",
    "time_to_best_incumbent_s", "final_iteration_time_to_first_incumbent_s",
    "final_iteration_time_to_best_incumbent_s", "node_count", "actual_iterations",
    "converged", "final_diff", "init_pos_prob", "tuning_candidates_planned",
    "tuning_candidates_completed", "tuning_candidates_failed",
    "paper_result_complete",
)

PREDICTIVE_SUMMARY_COLUMNS = (
    "dataset", "model", "depth", "n_folds_completed", "expected_folds",
    "is_complete_5fold", "missing_folds",
    *(name for metric in PREDICTIVE_METRICS for name in (f"{metric}_mean", f"{metric}_std")),
    *(f"oof_{metric}" for metric in OOF_METRICS),
)

OOF_SUMMARY_COLUMNS = (
    "dataset", "model", "depth", "n_folds_completed", "n_oof_samples",
    "expected_samples", "duplicate_sample_count", "is_complete_5fold",
    *OOF_METRICS,
)

COMPUTATIONAL_SUMMARY_COLUMNS = (
    "dataset", "model", "depth", "objective_sense", "n_folds_completed", "optimal_folds",
    "time_limit_folds", "other_feasible_folds", "failed_or_missing_folds",
    *(name for field in (
        "final_training_solver_time_s", "tuning_solver_time_s",
        "end_to_end_solver_time_s", "final_training_wall_time_s",
        "end_to_end_wall_time_s", "time_to_best_incumbent_s",
    ) for name in (f"{field}_mean", f"{field}_std", f"{field}_sum")),
    *(name for field in ("objective_value", "objective_bound", "node_count", "solution_count")
      for name in (f"{field}_mean", f"{field}_std")),
    "final_mip_gap_mean_pct", "final_mip_gap_max_pct", "worst_iteration_gap_max_pct",
)

ALPHA_COLUMNS = (
    "dataset", "model", "depth", "fold", "search_stage", "alpha", "is_selected",
    "validation_accuracy", "validation_recall", "validation_precision", "validation_f1",
    "validation_specificity", "validation_balanced_accuracy", "tuning_solver_status",
    "tuning_solver_time_s", "tuning_mip_gap_pct", "tuning_objective_value",
    "tuning_objective_bound", "tuning_time_to_best_s", "tuning_actual_iterations",
    "tuning_converged", "tuning_final_diff", "final_actual_iterations",
    "final_converged", "final_diff",
)


def _successful_attempt(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    attempts = []
    for path in parent.glob("attempt*"):
        if path.is_dir() and path.name.replace("attempt", "").isdigit():
            attempts.append(path)
    for attempt in sorted(attempts, key=lambda p: int(p.name[7:]), reverse=True):
        if is_success(attempt / "status.json"):
            return attempt
    return None


def _official_rows(results_dir: Path) -> list[tuple[Path, dict]]:
    rows = []
    for path in results_dir.glob("*/*/depth*/fold*/fold_result.json"):
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            continue
        if payload.get("status") == "success":
            rows.append((path, payload))
    return sorted(rows, key=lambda item: (
        item[1].get("dataset", ""), item[1].get("model", ""),
        int(item[1].get("depth", 0)), int(item[1].get("fold", 0)),
    ))


def _attempt_for_fold(results_dir: Path, fold_path: Path, payload: dict) -> Path | None:
    relative = payload.get("final_attempt_relative_path")
    if relative:
        candidate = results_dir / relative
        if is_success(candidate / "status.json"):
            return candidate
    return _successful_attempt(fold_path.parent / "final")


def _pick(payload: dict, *names, default=None):
    for name in names:
        if payload.get(name) is not None:
            return payload[name]
    return default


def _fold_frame(results_dir: Path, official: list[tuple[Path, dict]]) -> pd.DataFrame:
    rows = []
    for path, payload in official:
        attempt = _attempt_for_fold(results_dir, path, payload)
        final = read_json(attempt / "status.json") if attempt else {}
        metrics = final.get("metrics", {})
        solver = final.get("solver_summary", {})
        history = final.get("solver_history", [])
        last_history = history[-1] if history else {}
        result = {
            "dataset": payload.get("dataset"), "model": payload.get("model"),
            "depth": payload.get("depth"), "fold": payload.get("fold"),
            "best_alpha": payload.get("best_alpha"),
            "search_strategy": payload.get("search_strategy"),
            "tuning_time_limit_seconds_per_optimize": payload.get("tuning_time_limit_seconds_per_optimize"),
            "final_time_limit_seconds_per_optimize": payload.get("final_time_limit_seconds_per_optimize"),
            "n_outer_train": _pick(payload, "n_outer_train", default=final.get("n_train")),
            "n_outer_test": _pick(payload, "n_outer_test", default=final.get("n_evaluate")),
            "final_attempt": str(attempt.relative_to(results_dir)) if attempt else None,
            "result_schema_version": payload.get("result_schema_version", 1),
        }
        for metric in (*PREDICTIVE_METRICS, "tp", "fp", "fn", "tn"):
            result[metric] = _pick(payload, metric, default=metrics.get(metric))
        result.update({
            "final_training_wall_time_s": _pick(payload, "final_training_wall_time_s", "final_elapsed_seconds", default=final.get("elapsed_seconds")),
            "tuning_wall_time_s": _pick(payload, "tuning_wall_time_s", "tuning_elapsed_seconds"),
            "end_to_end_wall_time_s": _pick(payload, "end_to_end_wall_time_s", "total_elapsed_seconds"),
            "final_training_solver_time_s": _pick(payload, "final_training_solver_time_s", "final_solver_runtime_seconds", default=solver.get("solver_runtime_seconds")),
            "final_iteration_solver_time_s": _pick(payload, "final_iteration_solver_time_s", default=last_history.get("runtime_seconds")),
            "tuning_solver_time_s": _pick(payload, "tuning_solver_time_s", "tuning_solver_runtime_seconds"),
            "end_to_end_solver_time_s": _pick(payload, "end_to_end_solver_time_s", "total_solver_runtime_seconds"),
            "solver_status": _pick(payload, "solver_status", default=solver.get("solver_status")),
            "solver_status_name": _pick(payload, "solver_status_name", default=solver.get("solver_status_name")),
            "solver_solve_count": _pick(payload, "solver_solve_count", "solver_solves", default=solver.get("solver_solves")),
            "solution_count": _pick(payload, "solution_count", default=solver.get("solution_count")),
            "objective_value": _pick(payload, "objective_value", default=solver.get("objective_value")),
            "objective_bound": _pick(payload, "objective_bound", default=solver.get("objective_bound")),
            "objective_sense": _pick(payload, "objective_sense", default=solver.get("objective_sense")),
            "final_mip_gap": _pick(payload, "final_mip_gap", "mip_gap", default=solver.get("mip_gap")),
            "worst_iteration_mip_gap": _pick(payload, "worst_iteration_mip_gap", "max_mip_gap", default=solver.get("max_mip_gap")),
            "time_to_first_incumbent_s": _pick(payload, "time_to_first_incumbent_s", default=last_history.get("time_to_first_incumbent_s")),
            "time_to_best_incumbent_s": _pick(payload, "time_to_best_incumbent_s", default=last_history.get("time_to_best_incumbent_s")),
            "final_iteration_time_to_first_incumbent_s": _pick(payload, "final_iteration_time_to_first_incumbent_s", default=last_history.get("time_to_first_incumbent_s")),
            "final_iteration_time_to_best_incumbent_s": _pick(payload, "final_iteration_time_to_best_incumbent_s", default=last_history.get("time_to_best_incumbent_s")),
            "node_count": _pick(payload, "node_count", default=solver.get("node_count")),
            "actual_iterations": _pick(payload, "actual_iterations", default=final.get("actual_iterations")),
            "converged": _pick(payload, "converged", default=final.get("converged")),
            "final_diff": _pick(payload, "final_diff", default=final.get("final_diff")),
            "init_pos_prob": _pick(payload, "init_pos_prob", default=final.get("init_pos_prob")),
            "tuning_candidates_planned": payload.get("tuning_candidates_planned"),
            "tuning_candidates_completed": payload.get("tuning_candidates_completed"),
            "tuning_candidates_failed": payload.get("tuning_candidates_failed"),
            "paper_result_complete": bool(payload.get("paper_result_complete", False)),
        })
        rows.append(result)
    return pd.DataFrame(rows, columns=FOLD_COLUMNS)


def _mean_std(group: pd.DataFrame, metric: str) -> tuple[float | None, float | None]:
    values = pd.to_numeric(group[metric], errors="coerce").dropna()
    if values.empty:
        return None, None
    return float(values.mean()), (float(values.std(ddof=1)) if len(values) > 1 else None)


def _predictive_summary(folds: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in folds.groupby(["dataset", "model", "depth"], sort=True, dropna=False):
        dataset, model, depth = key
        completed = sorted(int(v) for v in group["fold"].dropna().unique())
        row = {
            "dataset": dataset, "model": model, "depth": depth,
            "n_folds_completed": len(completed), "expected_folds": 5,
            "is_complete_5fold": completed == [1, 2, 3, 4, 5],
            "missing_folds": ",".join(str(v) for v in range(1, 6) if v not in completed),
        }
        for metric in PREDICTIVE_METRICS:
            mean, std = _mean_std(group, metric)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        subset = oof[(oof["dataset"] == dataset) & (oof["model"] == model) & (oof["depth"] == depth)]
        if not subset.empty:
            score = (
                subset["y_score"].to_numpy()
                if model in PU_MODELS and subset["y_score"].notna().all() else None
            )
            combined = classification_metrics(
                subset["y_true"].to_numpy(), subset["y_pred"].to_numpy(), score
            )
            row.update({f"oof_{name}": combined.get(name) for name in OOF_METRICS})
        else:
            row.update({f"oof_{name}": None for name in OOF_METRICS})
        rows.append(row)
    return pd.DataFrame(rows, columns=PREDICTIVE_SUMMARY_COLUMNS)


def _oof_frame(results_dir: Path, official: list[tuple[Path, dict]]) -> pd.DataFrame:
    columns = ("dataset", "model", "depth", "fold", "sample_index", "y_true", "y_pred", "y_score", "best_alpha")
    frames = []
    for path, payload in official:
        attempt = _attempt_for_fold(results_dir, path, payload)
        if not attempt or not (attempt / "predictions.csv").exists():
            continue
        predictions = pd.read_csv(attempt / "predictions.csv")
        predictions.insert(0, "fold", payload.get("fold"))
        predictions.insert(0, "depth", payload.get("depth"))
        predictions.insert(0, "model", payload.get("model"))
        predictions.insert(0, "dataset", payload.get("dataset"))
        predictions["best_alpha"] = payload.get("best_alpha")
        frames.append(predictions[list(columns)])
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["dataset", "model", "depth", "sample_index"]
    ).reset_index(drop=True)


def _expected_sample_count(results_dir: Path, dataset: str) -> int | None:
    path = results_dir / dataset / "shared" / "splits_and_masks.json"
    if not path.exists():
        return None
    return int(read_json(path).get("n_samples"))


def _oof_metrics_summary(oof: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    rows = []
    if oof.empty:
        return pd.DataFrame(columns=OOF_SUMMARY_COLUMNS)
    for (dataset, model, depth), group in oof.groupby(
        ["dataset", "model", "depth"], sort=True
    ):
        expected = _expected_sample_count(results_dir, dataset)
        duplicates = int(group.duplicated("sample_index", keep=False).sum())
        folds = sorted(int(value) for value in group["fold"].unique())
        score = (
            group["y_score"].to_numpy()
            if model in PU_MODELS and group["y_score"].notna().all() else None
        )
        metrics = classification_metrics(
            group["y_true"].to_numpy(), group["y_pred"].to_numpy(), score
        )
        rows.append({
            "dataset": dataset,
            "model": model,
            "depth": depth,
            "n_folds_completed": len(folds),
            "n_oof_samples": len(group),
            "expected_samples": expected,
            "duplicate_sample_count": duplicates,
            "is_complete_5fold": bool(
                folds == [1, 2, 3, 4, 5]
                and duplicates == 0
                and expected is not None
                and len(group) == expected
                and group["sample_index"].nunique() == expected
            ),
            **{name: metrics.get(name) for name in OOF_METRICS},
        })
    return pd.DataFrame(rows, columns=OOF_SUMMARY_COLUMNS)


def _oof_confusion(oof: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    columns = (
        "dataset", "model", "depth", "n_folds_completed", "n_oof_samples",
        "expected_samples", "duplicate_sample_count", "label_mismatch_count",
        "is_complete_5fold", "tp", "fp", "fn", "tn",
    )
    rows = []
    if oof.empty:
        return pd.DataFrame(columns=columns)
    results_dir = getattr(oof, "attrs", {}).get("results_dir")
    for (dataset, model, depth), group in oof.groupby(["dataset", "model", "depth"], sort=True):
        duplicates = int(group.duplicated("sample_index", keep=False).sum())
        expected = _expected_sample_count(Path(results_dir), dataset) if results_dir else None
        mismatch = 0
        try:
            _, true_y = load_dataset(dataset, data_dir)
            idx = group["sample_index"].astype(int).to_numpy()
            valid = (idx >= 0) & (idx < len(true_y))
            mismatch = int((~valid).sum())
            if valid.any():
                mismatch += int(np.sum(true_y[idx[valid]] != group.loc[valid, "y_true"].astype(int).to_numpy()))
        except Exception:
            mismatch = -1
        metrics = classification_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        folds = sorted(int(v) for v in group["fold"].unique())
        complete = (
            folds == [1, 2, 3, 4, 5] and duplicates == 0 and mismatch == 0
            and expected is not None and len(group) == expected
            and group["sample_index"].nunique() == expected
        )
        rows.append({
            "dataset": dataset, "model": model, "depth": depth,
            "n_folds_completed": len(folds), "n_oof_samples": len(group),
            "expected_samples": expected, "duplicate_sample_count": duplicates,
            "label_mismatch_count": mismatch, "is_complete_5fold": complete,
            **{name: metrics[name] for name in ("tp", "fp", "fn", "tn")},
        })
    return pd.DataFrame(rows, columns=columns)


def _computational_summary(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    time_fields = (
        "final_training_solver_time_s", "tuning_solver_time_s",
        "end_to_end_solver_time_s", "final_training_wall_time_s",
        "end_to_end_wall_time_s", "time_to_best_incumbent_s",
    )
    value_fields = ("objective_value", "objective_bound", "node_count", "solution_count")
    for (dataset, model, depth), group in folds.groupby(["dataset", "model", "depth"], sort=True):
        names = group["solver_status_name"].fillna("").astype(str)
        row = {
            "dataset": dataset, "model": model, "depth": depth,
            "objective_sense": ",".join(sorted(set(
                group["objective_sense"].dropna().astype(str)
            ))) or None,
            "n_folds_completed": len(group),
            "optimal_folds": int((names == "OPTIMAL").sum()),
            "time_limit_folds": int((names == "TIME_LIMIT").sum()),
            "other_feasible_folds": int((~names.isin(["OPTIMAL", "TIME_LIMIT", ""])).sum()),
            "failed_or_missing_folds": max(0, 5 - len(group)),
        }
        for field in time_fields:
            values = pd.to_numeric(group[field], errors="coerce").dropna()
            row[f"{field}_mean"] = float(values.mean()) if len(values) else None
            row[f"{field}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
            row[f"{field}_sum"] = float(values.sum()) if len(values) else None
        for field in value_fields:
            mean, std = _mean_std(group, field)
            row[f"{field}_mean"] = mean
            row[f"{field}_std"] = std
        gap = pd.to_numeric(group["final_mip_gap"], errors="coerce").dropna()
        worst = pd.to_numeric(group["worst_iteration_mip_gap"], errors="coerce").dropna()
        row["final_mip_gap_mean_pct"] = float(100.0 * gap.mean()) if len(gap) else None
        row["final_mip_gap_max_pct"] = float(100.0 * gap.max()) if len(gap) else None
        row["worst_iteration_gap_max_pct"] = float(100.0 * worst.max()) if len(worst) else None
        rows.append(row)
    return pd.DataFrame(rows, columns=COMPUTATIONAL_SUMMARY_COLUMNS)


def _alpha_convergence(results_dir: Path, official: list[tuple[Path, dict]]) -> pd.DataFrame:
    rows = []
    for fold_path, fold in official:
        summary_path = fold_path.parent / "alpha_search_summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        best = float(summary.get("best_alpha"))
        for candidate in summary.get("candidates", []):
            alpha = float(candidate.get("alpha"))
            metrics = candidate.get("metrics", {})
            solver = candidate.get("solver_summary", {})
            selected = bool(np.isclose(alpha, best))
            row = {
                "dataset": fold.get("dataset"), "model": fold.get("model"),
                "depth": fold.get("depth"), "fold": fold.get("fold"),
                "search_stage": candidate.get("search_stage", "legacy_unknown"),
                "alpha": alpha, "is_selected": selected,
            }
            for metric in ("accuracy", "recall", "precision", "f1", "specificity", "balanced_accuracy"):
                row[f"validation_{metric}"] = metrics.get(metric)
            row.update({
                "tuning_solver_status": solver.get("solver_status_name"),
                "tuning_solver_time_s": solver.get("solver_runtime_seconds"),
                "tuning_mip_gap_pct": (100.0 * solver["mip_gap"] if solver.get("mip_gap") is not None else None),
                "tuning_objective_value": solver.get("objective_value"),
                "tuning_objective_bound": solver.get("objective_bound"),
                "tuning_time_to_best_s": solver.get("time_to_best_incumbent_s"),
                "tuning_actual_iterations": candidate.get("actual_iterations"),
                "tuning_converged": candidate.get("converged"),
                "tuning_final_diff": candidate.get("final_diff"),
                "final_actual_iterations": fold.get("actual_iterations") if selected else None,
                "final_converged": fold.get("converged") if selected else None,
                "final_diff": fold.get("final_diff") if selected else None,
            })
            rows.append(row)
    return pd.DataFrame(rows, columns=ALPHA_COLUMNS)


def build_paper_results(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, Path]:
    """Atomically refresh all paper tables from every successful official fold."""
    results_dir = Path(results_dir)
    output = results_dir / "paper_results"
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(output / ".refresh.lock"):
        official = _official_rows(results_dir)
        folds = _fold_frame(results_dir, official)
        oof = _oof_frame(results_dir, official)
        oof.attrs["results_dir"] = str(results_dir)
        oof_metrics = _oof_metrics_summary(oof, results_dir)
        confusion = _oof_confusion(oof, Path(data_dir))
        predictive = _predictive_summary(folds, oof)
        computational = _computational_summary(folds)
        alpha = _alpha_convergence(results_dir, official)
        tables = {
            "fold_results.csv": folds,
            "predictive_summary.csv": predictive,
            "computational_summary.csv": computational,
            "oof_predictions.csv": oof,
            "oof_metrics.csv": oof_metrics,
            "oof_confusion_matrix.csv": confusion,
            "alpha_and_convergence.csv": alpha,
        }
        paths = {}
        for name, frame in tables.items():
            path = output / name
            atomic_csv(path, frame)
            paths[name] = path
        return paths
