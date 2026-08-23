from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.config import (
    DATASETS, DEFAULT_CONFIG, DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR,
    FLOW_MODELS, MODELS, PU_MODELS, ExperimentConfig, min_samples_5pct,
)
from src.data_pipeline import (
    FoldPreprocessor, generate_pu_mask, load_dataset, make_outer_splits,
    new_preprocessor, validate_outer_splits,
)
from src.metrics import classification_metrics, summarize_fold_metrics
from src.result_io import (
    alpha_key, atomic_csv, atomic_json, atomic_pickle, is_success,
    next_attempt_dir, read_json,
)


def _successful_attempt(parent: Path) -> tuple[Path, dict] | None:
    if not parent.exists():
        return None
    attempts = sorted(
        (p for p in parent.glob("attempt*") if p.is_dir()),
        key=lambda p: int(p.name.replace("attempt", "")),
    )
    for attempt in reversed(attempts):
        status = attempt / "status.json"
        if is_success(status):
            return attempt, read_json(status)
    return None


def _iteration_writer(attempt_dir: Path) -> Callable:
    history: list[dict] = []

    def callback(record: dict, _model) -> None:
        history.append(record.copy())
        atomic_json(attempt_dir / "iteration_history.json", history)

    return callback


_SOLVER_STATUS_NAMES = {
    1: "LOADED",
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    6: "CUTOFF",
    7: "ITERATION_LIMIT",
    8: "NODE_LIMIT",
    9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT",
    11: "INTERRUPTED",
    12: "NUMERIC",
    13: "SUBOPTIMAL",
    14: "INPROGRESS",
    15: "USER_OBJ_LIMIT",
    16: "WORK_LIMIT",
    17: "MEM_LIMIT",
}


def _solver_summary(history: list[dict] | None) -> dict:
    """Flatten one or several Gurobi solves into fold-level reporting fields."""
    rows = list(history or [])
    if not rows:
        return {
            "solver_status": None,
            "solver_status_name": None,
            "solver_solves": 0,
            "solver_runtime_seconds": 0.0,
            "mip_gap": None,
            "max_mip_gap": None,
            "objective_value": None,
            "objective_bound": None,
            "objective_sense": None,
            "node_count": 0.0,
            "solution_count": 0,
            "time_to_first_incumbent_s": None,
            "time_to_best_incumbent_s": None,
            "final_iteration_time_to_first_incumbent_s": None,
            "final_iteration_time_to_best_incumbent_s": None,
            "final_iteration_solver_time_s": None,
        }
    last = rows[-1]
    status = last.get("status")
    gaps = [row.get("mip_gap") for row in rows if row.get("mip_gap") is not None]
    final_iteration_first = last.get("time_to_first_incumbent_s")
    final_iteration_best = last.get("time_to_best_incumbent_s")
    runtime_before_final = sum(
        float(row.get("runtime_seconds") or 0.0) for row in rows[:-1]
    )
    cumulative_best = (
        float(runtime_before_final + float(final_iteration_best))
        if final_iteration_best is not None else None
    )
    cumulative_first = (
        float(runtime_before_final + float(final_iteration_first))
        if final_iteration_first is not None else None
    )
    return {
        "solver_status": status,
        "solver_status_name": _SOLVER_STATUS_NAMES.get(status, str(status)),
        "solver_solves": len(rows),
        "solver_runtime_seconds": float(sum(float(row.get("runtime_seconds") or 0.0) for row in rows)),
        "mip_gap": last.get("mip_gap"),
        "max_mip_gap": float(max(gaps)) if gaps else None,
        "objective_value": last.get("objective_value"),
        "objective_bound": last.get("objective_bound"),
        "objective_sense": last.get("objective_sense"),
        "node_count": float(sum(float(row.get("node_count") or 0.0) for row in rows)),
        "solution_count": int(last.get("solution_count", last.get("sol_count", 0)) or 0),
        "time_to_first_incumbent_s": cumulative_first,
        "time_to_best_incumbent_s": cumulative_best,
        "final_iteration_time_to_first_incumbent_s": final_iteration_first,
        "final_iteration_time_to_best_incumbent_s": final_iteration_best,
        "final_iteration_solver_time_s": float(last.get("runtime_seconds") or 0.0),
    }


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _fmt_gap(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def _pick_result_gap(payload: dict):
    return payload.get("final_mip_gap", payload.get("mip_gap"))


def _progress(message: str) -> None:
    print(message, flush=True)


def _model_factory(
    model_name: str,
    depth: int,
    alpha: float,
    n_total: int,
    config: ExperimentConfig,
    solve_time_limit: int,
    attempt_dir: Path,
    show_solver_progress: bool = False,
    *,
    show_concise_progress: bool = True,
    heartbeat_seconds: int = 60,
    save_gurobi_log: bool = True,
    progress_context: str = "",
):
    callback = _iteration_writer(attempt_dir)
    log_base = attempt_dir / "gurobi.log"
    common = dict(
        max_depth=depth,
        alpha=alpha,
        warmstart=config.warm_start,
        timelimit=solve_time_limit,
        output=show_solver_progress,
        log_file=str(log_base),
        random_state=config.random_seed,
        concise_progress=show_concise_progress,
        heartbeat_seconds=heartbeat_seconds,
        save_gurobi_log=save_gurobi_log,
        progress_context=progress_context,
    )
    try:
        if model_name == "oct":
            from src.oct import optimalDecisionTreeClassifier
            return optimalDecisionTreeClassifier(
                min_samples_split=min_samples_5pct(n_total), **common
            )
        if model_name == "softpuoct":
            from src.softPUOCT import softPUOCT
            return softPUOCT(
                min_samples_split=min_samples_5pct(n_total),
                iteration_callback=callback,
                **common,
            )
        if model_name == "flowoct":
            from mfoctoriginal import FlowOCT
            return FlowOCT(**common)
        if model_name == "flowpuoct":
            from mfoct import FlowPUOCT
            return FlowPUOCT(iteration_callback=callback, **common)
    except ModuleNotFoundError as exc:
        if exc.name == "gurobipy":
            raise RuntimeError(
                "gurobipy is not installed. Run the actual MIP experiment in the "
                "environment that has both gurobipy and a valid Gurobi license."
            ) from exc
        raise
    raise ValueError(model_name)


def _flow_leaf_probabilities(model, X_train: np.ndarray) -> dict[int, float]:
    pi = np.asarray(model._pu_pi, dtype=float)
    leaf_samples: dict[int, list[int]] = {}
    for i, row in enumerate(X_train):
        node = 0
        while node not in model.labels:
            feature = model.branches[node]
            node = 2 * node + 1 if row[feature] == 0 else 2 * node + 2
        leaf_samples.setdefault(node, []).append(i)
    return {
        node: float(np.mean(pi[indices])) if indices else 0.5
        for node, indices in leaf_samples.items()
    }


def _flow_predict_score(model, X: np.ndarray, leaf_probs: dict[int, float]) -> np.ndarray:
    scores = []
    for row in X:
        node = 0
        while node not in model.labels:
            feature = model.branches[node]
            node = 2 * node + 1 if row[feature] == 0 else 2 * node + 2
        scores.append(leaf_probs.get(node, 0.5))
    return np.asarray(scores, dtype=float)


def _tree_snapshot(model, feature_names: list[str]) -> dict:
    snapshot = {
        "class": type(model).__name__,
        "feature_names": feature_names,
        "optgap": getattr(model, "optgap", None),
        "solver_history": getattr(model, "solve_history", []),
        "converged": getattr(model, "converged_", None),
        "actual_iterations": getattr(model, "n_iter_", None),
    }
    for name in ("branches", "labels", "_a", "_b", "_c", "_d", "_l", "leaf_pos_prob"):
        if hasattr(model, name):
            value = getattr(model, name)
            snapshot[name] = {str(k): v for k, v in value.items()} if isinstance(value, dict) else value
    return snapshot


def _fit_evaluate(
    *,
    model_name: str,
    dataset: str,
    depth: int,
    alpha: float,
    X_frame: pd.DataFrame,
    y_true: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    pos_mask: np.ndarray | None,
    unlabeled_mask: np.ndarray | None,
    config: ExperimentConfig,
    solve_time_limit: int,
    attempt_dir: Path,
    stage: str,
    show_solver_progress: bool,
    show_concise_progress: bool = True,
    heartbeat_seconds: int = 60,
    save_gurobi_log: bool = True,
) -> dict:
    started = time.time()
    train_indices = np.asarray(train_indices, dtype=int)
    eval_indices = np.asarray(eval_indices, dtype=int)
    y_train_true = y_true[train_indices]
    if pos_mask is None or unlabeled_mask is None:
        raise ValueError("All models require the saved shared pos/unlabeled masks")
    pos_mask = np.asarray(pos_mask, dtype=bool)
    unlabeled_mask = np.asarray(unlabeled_mask, dtype=bool)
    if len(pos_mask) != len(train_indices) or len(unlabeled_mask) != len(train_indices):
        raise ValueError("Saved mask length does not match the training partition")
    if not np.all(pos_mask ^ unlabeled_mask):
        raise ValueError("pos_mask and unlabeled_mask must be exclusive and exhaustive")
    if np.any(pos_mask & (y_train_true != config.positive_label)):
        raise ValueError("A saved labeled-positive mask includes a true negative")

    # Shared observed labels for all four models: labeled P=1 and every U=0.
    # For Flow models these labels also supervise fold-local discretization.
    y_train_pu = pos_mask.astype(np.int8)
    supervision = y_train_pu

    preprocessor = new_preprocessor(
        dataset, model_name, len(X_frame),
        config.random_seed,
    )
    X_train = preprocessor.fit_transform(X_frame.iloc[train_indices], supervision)
    X_eval = preprocessor.transform(X_frame.iloc[eval_indices])
    atomic_json(attempt_dir / "preprocessor.json", preprocessor.metadata())
    atomic_pickle(attempt_dir / "preprocessor.pkl", preprocessor)

    model = _model_factory(
        model_name, depth, alpha, len(X_frame), config, solve_time_limit, attempt_dir,
        show_solver_progress,
        show_concise_progress=show_concise_progress,
        heartbeat_seconds=heartbeat_seconds,
        save_gurobi_log=save_gurobi_log,
        progress_context=f"stage={stage} | {dataset} | D={depth} | alpha={alpha:g}",
    )
    if model_name in PU_MODELS:
        observed_positive_ratio = float(np.mean(pos_mask))
        init_pos_prob = float(np.clip(
            observed_positive_ratio / (1.0 - config.mask_ratio), 1e-6, 1.0 - 1e-6
        ))
        model.fit_pu(
            X_train,
            pos_mask=np.asarray(pos_mask, dtype=bool),
            unlabeled_mask=np.asarray(unlabeled_mask, dtype=bool),
            max_iter=config.em_max_iter,
            tol=config.em_tol,
            init_pos_prob=init_pos_prob,
            verbose=False,
        )
    else:
        init_pos_prob = None
        model.fit(X_train, y_train_pu)

    y_pred = np.asarray(model.predict(X_eval), dtype=int)
    y_score = None
    if model_name == "softpuoct":
        y_score = np.asarray(model.predict_proba(X_eval), dtype=float)
    elif model_name == "flowpuoct":
        y_score = _flow_predict_score(model, X_eval, _flow_leaf_probabilities(model, X_train))
    metrics = classification_metrics(y_true[eval_indices], y_pred, y_score)
    predictions = pd.DataFrame({
        "sample_index": eval_indices,
        "y_true": y_true[eval_indices],
        "y_pred": y_pred,
        "y_score": y_score if y_score is not None else np.nan,
    })
    atomic_csv(attempt_dir / "predictions.csv", predictions)
    atomic_json(attempt_dir / "tree.json", _tree_snapshot(model, preprocessor.output_columns))
    try:
        if hasattr(model, "iteration_callback"):
            model.iteration_callback = None
        atomic_pickle(attempt_dir / "model.pkl", model)
        model_pickle_saved = True
    except Exception as exc:
        model_pickle_saved = False
        atomic_json(attempt_dir / "model_pickle_error.json", {"error": repr(exc)})
    result = {
        "status": "success",
        "stage": stage,
        "dataset": dataset,
        "model": model_name,
        "depth": depth,
        "alpha": float(alpha),
        "mask_ratio": float(config.mask_ratio),
        "supervision_policy": config.supervision_policy,
        "training_label_source": "shared_observed_pu_labels",
        "preprocessing_label_source": "shared_observed_pu_labels",
        "n_train": int(len(train_indices)),
        "n_evaluate": int(len(eval_indices)),
        "init_pos_prob": init_pos_prob,
        "solve_time_limit_seconds": int(solve_time_limit),
        "metrics": metrics,
        "elapsed_seconds": float(time.time() - started),
        "solver_history": getattr(model, "solve_history", []),
        "solver_summary": _solver_summary(getattr(model, "solve_history", [])),
        "converged": getattr(model, "converged_", None),
        "actual_iterations": getattr(model, "n_iter_", None),
        "final_diff": (
            float(model.diff_history[-1])
            if getattr(model, "diff_history", None) else None
        ),
        "model_pickle_saved": model_pickle_saved,
    }
    atomic_json(attempt_dir / "status.json", result)
    return result


def _run_attempt(
    parent: Path,
    runner: Callable[[Path], dict],
    *,
    reuse_success: bool = True,
) -> tuple[Path, dict]:
    completed = _successful_attempt(parent) if reuse_success else None
    if completed is not None:
        return completed
    attempt = next_attempt_dir(parent)
    atomic_json(attempt / "status.json", {"status": "running", "started_at_unix": time.time()})
    try:
        return attempt, runner(attempt)
    except BaseException as exc:
        atomic_json(attempt / "status.json", {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise


def _select_best(results: list[dict]) -> dict:
    if not results:
        raise RuntimeError("No alpha candidate completed successfully")
    return max(results, key=lambda row: (row["metrics"]["f1"], -row["alpha"]))


def _save_shared_splits(
    dataset_root: Path,
    dataset: str,
    y: np.ndarray,
    splits: list[dict[str, np.ndarray]],
    config: ExperimentConfig,
) -> None:
    shared = dataset_root / "shared"
    payload = {
        "dataset": dataset,
        "n_samples": len(y),
        "config": config.as_dict(),
        "folds": [
            {key: value for key, value in split.items()}
            for split in splits
        ],
    }
    path = shared / "splits_and_masks.json"
    if path.exists():
        prior = read_json(path)
        if prior != json.loads(json.dumps(payload, default=lambda x: x.tolist())):
            raise RuntimeError(
                f"{path} already contains different splits/configuration. "
                "Use a new results directory rather than mixing protocols."
            )
    else:
        atomic_json(path, payload)


def _load_saved_tuning(summary_path: Path) -> tuple[list[dict], dict]:
    if not summary_path.exists():
        raise RuntimeError(
            f"Cannot run final-only/backfill because {summary_path} is missing. "
            "Complete alpha search first."
        )
    summary = read_json(summary_path)
    if summary.get("status") != "success" or summary.get("best_alpha") is None:
        raise RuntimeError(f"Saved alpha search is incomplete: {summary_path}")
    rows = []
    for candidate in summary.get("candidates", []):
        rows.append({
            "alpha": float(candidate["alpha"]),
            "metrics": candidate.get("metrics", {}),
            "elapsed_seconds": float(candidate.get("elapsed_seconds") or 0.0),
            "solver_summary": candidate.get("solver_summary", {}),
            "stage": candidate.get("search_stage", "legacy_unknown"),
            "actual_iterations": candidate.get("actual_iterations"),
            "converged": candidate.get("converged"),
            "final_diff": candidate.get("final_diff"),
        })
    best = next(
        (row for row in rows if np.isclose(row["alpha"], float(summary["best_alpha"]))),
        {"alpha": float(summary["best_alpha"]), "metrics": {}},
    )
    return rows, best


def _monitoring_complete(payload: dict) -> bool:
    return bool(
        payload.get("paper_result_complete")
        and int(payload.get("result_schema_version", 1)) >= 3
        and payload.get("time_to_best_incumbent_s") is not None
        and payload.get("final_attempt_relative_path")
    )


def _refresh_paper_results(results_dir: Path, data_dir: Path) -> None:
    try:
        from src.paper_results import build_paper_results
        build_paper_results(results_dir, data_dir)
    except Exception as exc:
        print(f"[paper_results warning] refresh failed: {exc}", file=sys.stderr, flush=True)


def _write_depth_outputs(
    results_dir: Path,
    out_root: Path,
    dataset: str,
    model_name: str,
    depth: int,
    fold_rows: list[dict],
) -> dict:
    """Write the complete five-fold table and recomputed OOF metrics."""
    ordered = sorted(fold_rows, key=lambda row: int(row["fold"]))
    atomic_csv(out_root / "fold_metrics.csv", pd.DataFrame(ordered))
    prediction_frames = []
    for row in ordered:
        attempt = results_dir / row["final_attempt_relative_path"]
        predictions = pd.read_csv(attempt / "predictions.csv")
        predictions.insert(0, "fold", int(row["fold"]))
        prediction_frames.append(predictions)
    oof = pd.concat(prediction_frames, ignore_index=True).sort_values("sample_index")
    atomic_csv(out_root / "oof_predictions.csv", oof)
    y_score = oof["y_score"].to_numpy() if oof["y_score"].notna().all() else None
    oof_metrics = classification_metrics(
        oof["y_true"].to_numpy(), oof["y_pred"].to_numpy(), y_score
    )
    duplicate_count = int(oof.duplicated("sample_index", keep=False).sum())
    oof_payload = {
        "status": "success",
        "dataset": dataset,
        "model": model_name,
        "depth": depth,
        "n_outer_folds": len(ordered),
        "n_oof_samples": int(len(oof)),
        "duplicate_sample_count": duplicate_count,
        "is_complete_oof": bool(len(ordered) == 5 and duplicate_count == 0),
        **oof_metrics,
    }
    atomic_json(out_root / "oof_metrics.json", oof_payload)
    fold_summary = summarize_fold_metrics(ordered)
    atomic_json(out_root / "summary.json", {
        "status": "success",
        "dataset": dataset,
        "model": model_name,
        "depth": depth,
        "n_outer_folds": len(ordered),
        "summary": fold_summary,
        "fold_mean_std": fold_summary,
        "oof_metrics": oof_payload,
    })
    return fold_summary


def run_experiment(
    model_name: str,
    datasets: list[str],
    depths: list[int],
    data_dir: Path,
    results_dir: Path,
    config: ExperimentConfig = DEFAULT_CONFIG,
    folds: list[int] | None = None,
    show_solver_progress: bool = False,
    show_concise_progress: bool = True,
    heartbeat_seconds: int = 60,
    save_gurobi_log: bool = True,
    run_mode: str = "full",
) -> None:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name!r}")
    if not datasets:
        raise ValueError("DATASETS_TO_RUN cannot be empty")
    unknown_datasets = [name for name in datasets if name not in DATASETS]
    if unknown_datasets:
        raise ValueError(
            f"Unknown dataset(s): {unknown_datasets}. Expected one or more of: {list(DATASETS)}"
        )
    if not depths or any(not isinstance(depth, int) or depth < 1 for depth in depths):
        raise ValueError("DEPTHS_TO_RUN must contain positive integers")
    if folds is not None:
        if not folds or any(fold not in range(1, config.outer_folds + 1) for fold in folds):
            raise ValueError(
                f"FOLDS_TO_RUN must contain integers from 1 to {config.outer_folds}"
            )
    if run_mode not in {"full", "final_only", "backfill_missing_monitoring"}:
        raise ValueError(
            "run_mode must be 'full', 'final_only', or 'backfill_missing_monitoring'"
        )
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    if config.tuning_time_limit <= 0 or config.final_time_limit <= 0:
        raise ValueError("tuning_time_limit and final_time_limit must be positive")

    selected_folds = folds if folds is not None else list(range(1, config.outer_folds + 1))
    run_started = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_rows: list[dict] = []
    total_fold_tasks = len(datasets) * len(depths) * len(selected_folds)
    completed_fold_tasks = 0
    _progress(
        "[run config] "
        f"model={model_name}, datasets={datasets}, depths={depths}, "
        f"folds={selected_folds}, alpha={list(config.alpha_broad)}, search=broad_only, "
        f"mask_ratio={config.mask_ratio:.0%}, supervision={config.supervision_policy}, "
        f"inner_validation={config.inner_validation_ratio:.0%}, "
        f"tuning_subsample={config.tuning_subsample_ratio:.0%}, "
        f"tuning_limit={config.tuning_time_limit}s/optimize, "
        f"final_limit={config.final_time_limit}s/optimize, "
        f"run_mode={run_mode}, heartbeat={heartbeat_seconds}s, "
        f"concise_progress={'on' if show_concise_progress else 'off'}, "
        f"raw_gurobi_console={'on' if show_solver_progress else 'off'}, "
        f"save_gurobi_log={'on' if save_gurobi_log else 'off'}"
    )
    for dataset_number, dataset in enumerate(datasets, start=1):
        _progress(f"\n[dataset {dataset_number}/{len(datasets)}] loading {dataset}")
        X, y = load_dataset(dataset, data_dir, config.target_col)
        splits = make_outer_splits(y, config)
        validate_outer_splits(splits, y)
        dataset_root = Path(results_dir) / dataset
        _save_shared_splits(dataset_root, dataset, y, splits, config)
        for depth in depths:
            fold_rows = []
            for split in splits:
                fold = int(split["fold"][0])
                if folds and fold not in folds:
                    continue
                fold_root = dataset_root / model_name / f"depth{depth}" / f"fold{fold}"
                official = fold_root / "fold_result.json"
                prior_fold = read_json(official) if is_success(official) else None
                needs_backfill = bool(
                    prior_fold is not None
                    and run_mode == "backfill_missing_monitoring"
                    and not _monitoring_complete(prior_fold)
                )
                if prior_fold is not None and not needs_backfill and run_mode != "final_only":
                    resumed = prior_fold
                    fold_rows.append(resumed)
                    run_rows.append(resumed)
                    _progress(
                        f"[resume] {dataset} {model_name} D={depth} fold={fold} "
                        f"ACC={_fmt(resumed.get('accuracy'))} F1={_fmt(resumed.get('f1'))} "
                        f"GAP={_fmt_gap(_pick_result_gap(resumed))}"
                    )
                    completed_fold_tasks += 1
                    _progress(
                        f"[progress] completed={completed_fold_tasks}/{total_fold_tasks} "
                        f"| cumulative={time.time() - run_started:.2f}s"
                    )
                    continue
                tuning_results: list[dict] = []
                alpha_root = fold_root / "alpha_search"
                alpha_summary_path = fold_root / "alpha_search_summary.json"
                use_saved_tuning = needs_backfill or run_mode == "final_only"
                if use_saved_tuning:
                    tuning_results, best = _load_saved_tuning(alpha_summary_path)
                    saved_summary = read_json(alpha_summary_path)
                    tuning_candidates_planned = int(saved_summary.get("tuning_candidates_planned", len(tuning_results)))
                    tuning_candidates_completed = int(saved_summary.get("tuning_candidates_completed", len(tuning_results)))
                    tuning_candidates_failed = int(saved_summary.get(
                        "tuning_candidates_failed",
                        max(0, tuning_candidates_planned - tuning_candidates_completed),
                    ))
                    _progress(
                        f"\n[fold start] {dataset} {model_name} D={depth} fold={fold}/{config.outer_folds} "
                        f"| {'monitoring backfill' if needs_backfill else 'final only'}"
                    )
                    _progress(
                        f"[alpha reused] best_alpha={best['alpha']:g} | alpha search skipped"
                    )
                else:
                    _progress(
                        f"\n[fold start] {dataset} {model_name} D={depth} fold={fold}/{config.outer_folds} "
                        f"| broad alpha search"
                    )
                    _progress(
                        f"[tuning] train={len(split['tuning_subsample'])} "
                        f"validation={len(split['inner_validation'])} "
                        f"candidates={len(config.alpha_broad)}"
                    )
                    for alpha_number, alpha in enumerate(config.alpha_broad, start=1):
                        parent = alpha_root / f"alpha_{alpha_key(alpha)}"
                        _progress(
                            f"[solve start] stage=alpha_broad | candidate={alpha_number}/{len(config.alpha_broad)} "
                            f"| {dataset} D={depth} fold={fold} alpha={alpha:g} "
                            f"| limit={config.tuning_time_limit}s/optimize"
                        )
                        try:
                            _, result = _run_attempt(parent, lambda attempt, alpha=alpha: _fit_evaluate(
                                model_name=model_name, dataset=dataset, depth=depth, alpha=alpha,
                                X_frame=X, y_true=y,
                                train_indices=split["tuning_subsample"],
                                eval_indices=split["inner_validation"],
                                pos_mask=split["tuning_pos_mask"],
                                unlabeled_mask=split["tuning_unlabeled_mask"],
                                config=config, solve_time_limit=config.tuning_time_limit,
                                attempt_dir=attempt, stage="alpha_tuning_broad",
                                show_solver_progress=show_solver_progress,
                                show_concise_progress=show_concise_progress,
                                heartbeat_seconds=heartbeat_seconds,
                                save_gurobi_log=save_gurobi_log,
                            ))
                            result["search_stage"] = "broad"
                            tuning_results.append(result)
                            solver = result["solver_summary"]
                            _progress(
                                f"[solve done] stage=alpha_broad | alpha={alpha:g} "
                                f"VAL_ACC={_fmt(result['metrics'].get('accuracy'))} "
                                f"VAL_F1={_fmt(result['metrics'].get('f1'))} "
                                f"status={solver['solver_status_name']} | GAP={_fmt_gap(solver.get('mip_gap'))} "
                                f"| solver_time={solver['solver_runtime_seconds']:.2f}s"
                            )
                        except Exception as exc:
                            print(f"[failed alpha] {dataset} {model_name} D={depth} fold={fold} alpha={alpha}: {exc}", file=sys.stderr, flush=True)
                    best = _select_best(tuning_results)
                    tuning_candidates_planned = len(config.alpha_broad)
                    tuning_candidates_completed = len(tuning_results)
                    tuning_candidates_failed = max(0, tuning_candidates_planned - tuning_candidates_completed)
                    atomic_json(alpha_summary_path, {
                        "status": "success", "best_alpha": best["alpha"],
                        "search_strategy": "broad_only",
                        "mask_ratio": float(config.mask_ratio),
                        "supervision_policy": config.supervision_policy,
                        "alpha_selection_scope": "within_each_outer_fold",
                        "selection_metric": "inner_validation_f1",
                        "tuning_time_limit_seconds_per_optimize": config.tuning_time_limit,
                        "tuning_candidates_planned": tuning_candidates_planned,
                        "tuning_candidates_completed": tuning_candidates_completed,
                        "tuning_candidates_failed": tuning_candidates_failed,
                        "candidates": [{
                            "alpha": row["alpha"], "search_stage": row.get("search_stage"),
                            "metrics": row["metrics"], "elapsed_seconds": row["elapsed_seconds"],
                            "solver_summary": row["solver_summary"],
                            "actual_iterations": row.get("actual_iterations"),
                            "converged": row.get("converged"), "final_diff": row.get("final_diff"),
                        } for row in tuning_results],
                    })
                    _progress(
                        f"[alpha selected] {dataset} D={depth} fold={fold} "
                        f"best_alpha={best['alpha']:g} VAL_F1={_fmt(best['metrics'].get('f1'))}"
                    )
                final_parent = fold_root / "final"
                _progress(
                    f"[solve start] stage=final outer-test model "
                    f"| {dataset} D={depth} fold={fold} alpha={best['alpha']:g} "
                    f"| outer_train={len(split['outer_train'])} outer_test={len(split['outer_test'])} "
                    f"| limit={config.final_time_limit}s/optimize"
                )
                reuse_final_success = not use_saved_tuning
                prior_final_attempt = _successful_attempt(final_parent)
                if (
                    reuse_final_success
                    and run_mode == "backfill_missing_monitoring"
                    and prior_final_attempt is not None
                ):
                    prior_final_status = read_json(prior_final_attempt[0] / "status.json")
                    prior_history = prior_final_status.get("solver_history", [])
                    if not prior_history or prior_history[-1].get("time_to_best_incumbent_s") is None:
                        reuse_final_success = False
                final_attempt, final = _run_attempt(final_parent, lambda attempt: _fit_evaluate(
                    model_name=model_name, dataset=dataset, depth=depth, alpha=best["alpha"],
                    X_frame=X, y_true=y,
                    train_indices=split["outer_train"],
                    eval_indices=split["outer_test"],
                    pos_mask=split["final_pos_mask"],
                    unlabeled_mask=split["final_unlabeled_mask"],
                    config=config, solve_time_limit=config.final_time_limit,
                    attempt_dir=attempt, stage="outer_test",
                    show_solver_progress=show_solver_progress,
                    show_concise_progress=show_concise_progress,
                    heartbeat_seconds=heartbeat_seconds,
                    save_gurobi_log=save_gurobi_log,
                ), reuse_success=reuse_final_success)
                final_solver = final["solver_summary"]
                tuning_solver_seconds = float(sum(
                    float(row.get("solver_summary", {}).get("solver_runtime_seconds") or 0.0)
                    for row in tuning_results
                ))
                tuning_elapsed_seconds = float(sum(float(row.get("elapsed_seconds") or 0.0) for row in tuning_results))
                final_relative = str(final_attempt.relative_to(Path(results_dir)))
                fold_result = {
                    "status": "success", "dataset": dataset, "model": model_name,
                    "depth": depth, "fold": fold, "best_alpha": best["alpha"],
                    "result_schema_version": 4,
                    "mask_ratio": float(config.mask_ratio),
                    "supervision_policy": config.supervision_policy,
                    "training_label_source": "shared_observed_pu_labels",
                    "alpha_selection_scope": "within_each_outer_fold",
                    "final_attempt_relative_path": final_relative,
                    "search_strategy": "broad_only",
                    "tuning_time_limit_seconds_per_optimize": config.tuning_time_limit,
                    "final_time_limit_seconds_per_optimize": config.final_time_limit,
                    "n_outer_train": int(len(split["outer_train"])),
                    "n_outer_test": int(len(split["outer_test"])),
                    **final["metrics"],
                    "final_elapsed_seconds": final["elapsed_seconds"],
                    "final_training_wall_time_s": final["elapsed_seconds"],
                    "tuning_elapsed_seconds": tuning_elapsed_seconds,
                    "tuning_wall_time_s": tuning_elapsed_seconds,
                    "total_elapsed_seconds": tuning_elapsed_seconds + final["elapsed_seconds"],
                    "end_to_end_wall_time_s": tuning_elapsed_seconds + final["elapsed_seconds"],
                    "final_solver_runtime_seconds": final_solver["solver_runtime_seconds"],
                    "final_training_solver_time_s": final_solver["solver_runtime_seconds"],
                    "final_iteration_solver_time_s": final_solver["final_iteration_solver_time_s"],
                    "tuning_solver_runtime_seconds": tuning_solver_seconds,
                    "tuning_solver_time_s": tuning_solver_seconds,
                    "total_solver_runtime_seconds": tuning_solver_seconds + final_solver["solver_runtime_seconds"],
                    "end_to_end_solver_time_s": tuning_solver_seconds + final_solver["solver_runtime_seconds"],
                    "solver_status": final_solver["solver_status"],
                    "solver_status_name": final_solver["solver_status_name"],
                    "solver_solves": final_solver["solver_solves"],
                    "solver_solve_count": final_solver["solver_solves"],
                    "solution_count": final_solver["solution_count"],
                    "mip_gap": final_solver["mip_gap"],
                    "final_mip_gap": final_solver["mip_gap"],
                    "max_mip_gap": final_solver["max_mip_gap"],
                    "worst_iteration_mip_gap": final_solver["max_mip_gap"],
                    "objective_value": final_solver["objective_value"],
                    "objective_bound": final_solver["objective_bound"],
                    "objective_sense": final_solver["objective_sense"],
                    "node_count": final_solver["node_count"],
                    "time_to_first_incumbent_s": final_solver["time_to_first_incumbent_s"],
                    "time_to_best_incumbent_s": final_solver["time_to_best_incumbent_s"],
                    "final_iteration_time_to_first_incumbent_s": final_solver["final_iteration_time_to_first_incumbent_s"],
                    "final_iteration_time_to_best_incumbent_s": final_solver["final_iteration_time_to_best_incumbent_s"],
                    "actual_iterations": final["actual_iterations"],
                    "converged": final["converged"],
                    "final_diff": final["final_diff"],
                    "init_pos_prob": final.get("init_pos_prob"),
                    "tuning_candidates_planned": tuning_candidates_planned,
                    "tuning_candidates_completed": tuning_candidates_completed,
                    "tuning_candidates_failed": tuning_candidates_failed,
                }
                fold_result["paper_result_complete"] = bool(
                    fold_result["time_to_best_incumbent_s"] is not None
                    and fold_result["solution_count"] > 0
                )
                atomic_json(official, fold_result)
                _refresh_paper_results(Path(results_dir), Path(data_dir))
                fold_rows.append(fold_result)
                run_rows.append(fold_result)
                _progress(
                    f"[fold done] {dataset} {model_name} D={depth} fold={fold} "
                    f"ACC={fold_result['accuracy']:.4f} "
                    f"F1={fold_result['f1']:.4f} "
                    f"PR_AUC={fold_result.get('pr_auc', float('nan')):.4f} "
                    f"BRIER={fold_result.get('brier_score', float('nan')):.4f} "
                    f"GAP={_fmt_gap(fold_result['mip_gap'])} "
                    f"final_solver_time={fold_result['final_solver_runtime_seconds']:.2f}s "
                    f"total_solver_time={fold_result['total_solver_runtime_seconds']:.2f}s "
                    f"status={fold_result['solver_status_name']}"
                )
                completed_fold_tasks += 1
                _progress(
                    f"[progress] completed={completed_fold_tasks}/{total_fold_tasks} "
                    f"| cumulative={time.time() - run_started:.2f}s"
                )
            if len(fold_rows) == config.outer_folds:
                out_root = dataset_root / model_name / f"depth{depth}"
                fold_summary = _write_depth_outputs(
                    Path(results_dir), out_root, dataset, model_name, depth, fold_rows
                )
                shown = []
                for metric in (
                        "accuracy",
                        "f1",
                        "pr_auc",
                        "brier_score",
                ):
                    values = fold_summary.get(metric, {})
                    shown.append(
                        f"{metric.upper()}={_fmt(values.get('mean'))}±{_fmt(values.get('std'))}"
                    )
                _progress(
                    f"[summary] {dataset} {model_name} D={depth} | " + " | ".join(shown)
                )

    _refresh_paper_results(Path(results_dir), Path(data_dir))
    if run_rows:
        run_root = Path(results_dir) / "run_summaries" / f"{model_name}_{run_id}"
        run_frame = pd.DataFrame(run_rows).sort_values(["dataset", "depth", "fold"])
        atomic_csv(run_root / "fold_results.csv", run_frame)
        latest_path = Path(results_dir) / f"latest_run_summary_{model_name}.csv"
        atomic_csv(latest_path, run_frame)

        aggregate_rows = []
        for (dataset, depth), group in run_frame.groupby(["dataset", "depth"], sort=True):
            aggregate_rows.append({
                "dataset": dataset,
                "model": model_name,
                "depth": int(depth),
                "n_folds": int(len(group)),
                "accuracy_mean": float(group["accuracy"].mean()),
                "accuracy_std": float(group["accuracy"].std(ddof=1)) if len(group) > 1 else None,
                "f1_mean": float(group["f1"].mean()),
                "f1_std": float(group["f1"].std(ddof=1)) if len(group) > 1 else None,
                "tp_total": int(group["tp"].sum()),
                "fp_total": int(group["fp"].sum()),
                "fn_total": int(group["fn"].sum()),
                "tn_total": int(group["tn"].sum()),
                "mip_gap_mean": float(group["mip_gap"].dropna().mean()) if group["mip_gap"].notna().any() else None,
                "mip_gap_max": float(group["mip_gap"].dropna().max()) if group["mip_gap"].notna().any() else None,
                "final_solver_time_mean_s": float(group["final_solver_runtime_seconds"].mean()),
                "total_solver_time_sum_s": float(group["total_solver_runtime_seconds"].sum()),
                "pr_auc_mean": (
                    float(group["pr_auc"].mean())
                    if "pr_auc" in group.columns
                    else None
                ),
                "pr_auc_std": (
                    float(group["pr_auc"].std(ddof=1))
                    if "pr_auc" in group.columns and len(group) > 1
                    else None
                ),
                "brier_score_mean": (
                    float(group["brier_score"].mean())
                    if "brier_score" in group.columns
                    else None
                ),
                "brier_score_std": (
                    float(group["brier_score"].std(ddof=1))
                    if "brier_score" in group.columns and len(group) > 1
                    else None
                ),
            })
        aggregate_frame = pd.DataFrame(aggregate_rows)
        atomic_csv(run_root / "aggregate_summary.csv", aggregate_frame)
        atomic_json(run_root / "aggregate_summary.json", aggregate_rows)

        _progress("\n" + "=" * 112)
        _progress("FINAL RUN SUMMARY (outer-test results; GAP is from each fold's final model)")
        display = aggregate_frame.copy()
        for column in (
                "accuracy_mean",
                "accuracy_std",
                "f1_mean",
                "f1_std",
                "pr_auc_mean",
                "pr_auc_std",
                "brier_score_mean",
                "brier_score_std",
                "mip_gap_mean",
                "mip_gap_max",
        ):
            display[column] = display[column].map(lambda value: _fmt(value))
        for column in ("final_solver_time_mean_s", "total_solver_time_sum_s"):
            display[column] = display[column].map(lambda value: f"{value:.2f}")
        _progress(display.to_string(index=False))
        _progress(f"Total wall-clock time: {time.time() - run_started:.2f}s")
        _progress(f"Fold-level summary saved to: {latest_path}")
        _progress(f"This run's aggregate summary saved to: {run_root / 'aggregate_summary.csv'}")
        _progress(f"Paper-ready results saved to: {Path(results_dir) / 'paper_results'}")
        _progress("=" * 112)


def build_parser(default_model: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Revised PUOCT outer-five-fold experiment"
    )
    if default_model is None:
        parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--folds", nargs="+", type=int, choices=range(1, 6))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--tuning-time-limit", type=int, default=DEFAULT_CONFIG.tuning_time_limit,
        help="Per-optimize limit for each broad-search alpha (scheme A).",
    )
    parser.add_argument(
        "--final-time-limit", type=int, default=DEFAULT_CONFIG.final_time_limit,
        help="Per-optimize limit after alpha selection (scheme A).",
    )
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument(
        "--run-mode", choices=("full", "final_only", "backfill_missing_monitoring"),
        default="full",
    )
    parser.add_argument("--raw-gurobi-console", action="store_true")
    parser.add_argument("--no-concise-progress", action="store_true")
    parser.add_argument("--no-gurobi-log", action="store_true")
    parser.set_defaults(default_model=default_model)
    return parser


def cli(default_model: str | None = None) -> None:
    args = build_parser(default_model).parse_args()
    model_name = default_model or args.model
    config = ExperimentConfig(
        tuning_time_limit=args.tuning_time_limit,
        final_time_limit=args.final_time_limit,
    )
    run_experiment(
        model_name=model_name,
        datasets=args.datasets,
        depths=args.depths,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        config=config,
        folds=args.folds,
        show_solver_progress=args.raw_gurobi_console,
        show_concise_progress=not args.no_concise_progress,
        heartbeat_seconds=args.heartbeat_seconds,
        save_gurobi_log=not args.no_gurobi_log,
        run_mode=args.run_mode,
    )


if __name__ == "__main__":
    cli()
