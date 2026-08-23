from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import gurobipy as gp
from gurobipy import GRB


SOLVER_STATUS_NAMES = {
    1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
    5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT",
    9: "TIME_LIMIT", 10: "SOLUTION_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC",
    13: "SUBOPTIMAL", 14: "INPROGRESS", 15: "USER_OBJ_LIMIT",
    16: "WORK_LIMIT", 17: "MEM_LIMIT",
}


def configure_solver_logging(
    model,
    *,
    show_raw_console: bool,
    save_log: bool,
    log_file: str | Path | None,
) -> None:
    """Keep raw console output independent from complete on-disk logging."""
    log_path = Path(log_file) if log_file else None
    if save_log and log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    model.Params.OutputFlag = int(bool(show_raw_console or (save_log and log_path)))
    model.Params.LogToConsole = int(bool(show_raw_console))
    if save_log and log_path is not None:
        model.Params.LogFile = str(log_path)


def iteration_log_path(base: str | Path | None, iteration: int | None) -> str | None:
    if not base:
        return None
    path = Path(base)
    if iteration is None:
        return str(path)
    suffix = path.suffix or ".log"
    return str(path.with_name(f"{path.stem}_iter{int(iteration):02d}{suffix}"))


class SolverMonitor:
    """Record incumbent timing and print a low-frequency MIP heartbeat."""

    def __init__(
        self,
        *,
        heartbeat_seconds: float = 60.0,
        show_concise_progress: bool = True,
        context: str = "",
        iteration: int | None = None,
        max_iterations: int | None = None,
        sense: int | None = None,
    ) -> None:
        self.heartbeat_seconds = max(1.0, float(heartbeat_seconds))
        self.show_concise_progress = bool(show_concise_progress)
        self.context = context
        self.iteration = iteration
        self.max_iterations = max_iterations
        self.time_to_first_incumbent_s: float | None = None
        self.time_to_best_incumbent_s: float | None = None
        self.best_incumbent_update_count = 0
        self.best_incumbent_objective: float | None = None
        self._next_heartbeat = self.heartbeat_seconds
        self._sense: int | None = int(sense) if sense is not None else None

    def _is_better(self, value: float) -> bool:
        if self.best_incumbent_objective is None:
            return True
        if self._sense == GRB.MAXIMIZE:
            return value > self.best_incumbent_objective + 1e-10
        return value < self.best_incumbent_objective - 1e-10

    @staticmethod
    def _finite_objective(value: float) -> float | None:
        value = float(value)
        return value if math.isfinite(value) and abs(value) < GRB.INFINITY / 2 else None

    @staticmethod
    def _gap(incumbent: float | None, bound: float | None) -> float | None:
        if incumbent is None or bound is None:
            return None
        return abs(incumbent - bound) / max(abs(incumbent), 1e-10)

    @staticmethod
    def _fmt(value: float | None, digits: int = 4) -> str:
        return "NA" if value is None else f"{value:.{digits}f}"

    def __call__(self, model, where: int) -> None:
        try:
            if self._sense is None:
                self._sense = int(model.ModelSense)
            if where == GRB.Callback.MIPSOL:
                runtime = float(model.cbGet(GRB.Callback.RUNTIME))
                objective = self._finite_objective(model.cbGet(GRB.Callback.MIPSOL_OBJ))
                if objective is not None and self._is_better(objective):
                    if self.time_to_first_incumbent_s is None:
                        self.time_to_first_incumbent_s = runtime
                    self.time_to_best_incumbent_s = runtime
                    self.best_incumbent_objective = objective
                    self.best_incumbent_update_count += 1
            if where != GRB.Callback.MIP or not self.show_concise_progress:
                return
            runtime = float(model.cbGet(GRB.Callback.RUNTIME))
            if runtime + 1e-9 < self._next_heartbeat:
                return
            incumbent = self._finite_objective(model.cbGet(GRB.Callback.MIP_OBJBST))
            bound = self._finite_objective(model.cbGet(GRB.Callback.MIP_OBJBND))
            nodes = float(model.cbGet(GRB.Callback.MIP_NODCNT))
            gap = self._gap(incumbent, bound)
            em = ""
            if self.iteration is not None:
                suffix = f"/{self.max_iterations}" if self.max_iterations is not None else ""
                em = f" | EM={self.iteration}{suffix}"
            context = f" | {self.context}" if self.context else ""
            print(
                f"[solving]{em} | elapsed={runtime:.0f}s | "
                f"incumbent={self._fmt(incumbent)} | bound={self._fmt(bound)} | "
                f"gap={'NA' if gap is None else f'{100.0 * gap:.2f}%'} | "
                f"nodes={nodes:.0f}{context}",
                flush=True,
            )
            while self._next_heartbeat <= runtime:
                self._next_heartbeat += self.heartbeat_seconds
        except Exception:
            # Monitoring must never terminate a valid optimization.
            return

    def fields(self) -> dict:
        return {
            "time_to_first_incumbent_s": self.time_to_first_incumbent_s,
            "time_to_best_incumbent_s": self.time_to_best_incumbent_s,
            "best_incumbent_update_count": int(self.best_incumbent_update_count),
            "best_incumbent_objective": self.best_incumbent_objective,
        }


def combined_callback(
    monitor: SolverMonitor,
    lazy_callback: Callable | None = None,
) -> Callable:
    """Compose monitoring with an existing Flow lazy-constraint callback."""
    def callback(model, where) -> None:
        if lazy_callback is not None:
            if where == GRB.Callback.MIPSOL:
                model._lazy_cut_added = False
            lazy_callback(model, where)
            if where == GRB.Callback.MIPSOL and getattr(model, "_lazy_cut_added", False):
                return
        monitor(model, where)
    return callback


def solver_record(model, *, iteration=None, diff=None, monitor: SolverMonitor | None = None) -> dict:
    sol_count = int(model.SolCount)
    status = int(model.Status)
    record = {
        "iteration": iteration,
        "diff": diff,
        "status": status,
        "status_name": SOLVER_STATUS_NAMES.get(status, str(status)),
        "sol_count": sol_count,
        "solution_count": sol_count,
        "runtime_seconds": float(model.Runtime),
        "mip_gap": float(model.MIPGap) if sol_count > 0 else None,
        "objective_value": float(model.ObjVal) if sol_count > 0 else None,
        "objective_bound": float(model.ObjBound) if sol_count > 0 else None,
        "objective_sense": "maximize" if int(model.ModelSense) == GRB.MAXIMIZE else "minimize",
        "node_count": float(model.NodeCount),
    }
    record.update(monitor.fields() if monitor is not None else {
        "time_to_first_incumbent_s": None,
        "time_to_best_incumbent_s": None,
        "best_incumbent_update_count": None,
        "best_incumbent_objective": None,
    })
    return record
