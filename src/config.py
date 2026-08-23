
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_RESULTS_DIR =  PROJECT_DIR / "results"

DATASETS = (
    "breast_cancer_diagnostic",
    "climate_model",
    "house-votes-84",
    "ionosphere",
    "monk1",
    "monk2",
    "monk3",
    "parkinsons",
    "sonar",
    "spect",
    "tic-tac-toe",
    "wholesale",
)

MODELS = ("oct", "softpuoct", "flowoct", "flowpuoct")
PU_MODELS = frozenset(("softpuoct", "flowpuoct"))
FLOW_MODELS = frozenset(("flowoct", "flowpuoct"))


@dataclass(frozen=True)
class ExperimentConfig:
    target_col: str = "target"
    random_seed: int = 42
    outer_folds: int = 5
    inner_validation_ratio: float = 0.30
    tuning_subsample_ratio: float = 0.70
    mask_ratio: float = 0.5

    supervision_policy: str = "shared_positive_mask_all_models"
    em_max_iter: int = 6
    em_tol: float = 1e-3

    tuning_time_limit: int = 60
    final_time_limit: int = 600
    warm_start: bool = True
    alpha_broad: tuple[float, ...] = (0.0, 0.001, 0.01, 0.1, 1.0)
    positive_label: int = 1
    unlabeled_label: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = ExperimentConfig()


def dataset_path(dataset: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Expected one of: {', '.join(DATASETS)}")
    return Path(data_dir) / f"{dataset}_enc.csv"


def min_samples_5pct(n_total: int) -> int:
    """Keep the original code's exact floor-based 5% definition."""
    return max(1, int(0.05 * int(n_total)))
