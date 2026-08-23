

from src.config import  DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR
from src.config import ExperimentConfig
from src.experiment_engine import run_experiment

# breast_cancer_diagnostic, climate_model, house-votes-84, ionosphere,
# monk1, monk2, monk3, parkinsons, sonar, spect, tic-tac-toe, wholesale
DATASETS_TO_RUN = ['monk1']
DEPTHS_TO_RUN = [2]
FOLDS_TO_RUN = [1, 2, 3, 4, 5]
TUNING_TIME_LIMIT = 60
FINAL_TIME_LIMIT = 600
SHOW_CONCISE_PROGRESS = True
HEARTBEAT_SECONDS = 60
SHOW_RAW_GUROBI_CONSOLE = False
SAVE_GUROBI_LOG = False
RUN_MODE = "full"

DATA_DIR = DEFAULT_DATA_DIR
RESULTS_DIR = DEFAULT_RESULTS_DIR


if __name__ == "__main__":
    run_experiment(
        model_name="oct",
        datasets=DATASETS_TO_RUN,
        depths=DEPTHS_TO_RUN,
        folds=FOLDS_TO_RUN,
        data_dir=DATA_DIR,
        results_dir=RESULTS_DIR,
        config=ExperimentConfig(
            tuning_time_limit=TUNING_TIME_LIMIT,
            final_time_limit=FINAL_TIME_LIMIT,
        ),
        show_solver_progress=SHOW_RAW_GUROBI_CONSOLE,
        show_concise_progress=SHOW_CONCISE_PROGRESS,
        heartbeat_seconds=HEARTBEAT_SECONDS,
        save_gurobi_log=SAVE_GUROBI_LOG,
        run_mode=RUN_MODE,
    )
