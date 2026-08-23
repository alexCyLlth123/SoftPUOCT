# SoftPUOCT: Soft Positive-Unlabeled Optimal Classification Trees

This repository provides the implementation of **Soft Positive-Unlabeled Optimal Classification Trees (SoftPUOCT)**.

SoftPUOCT is an optimization-based decision tree framework designed for **positive-unlabeled (PU) classification**, where only a subset of positive samples is labeled and the remaining samples are treated as unlabeled data.

This repository contains:

- SoftPUOCT model implementation
- Optimal Classification Tree (OCT) baseline
- PU data preprocessing pipeline
- Cross-validation experiment framework
- Evaluation and result generation tools

## Environment
The experiments were conducted using:

- Python 3.12.9
- Gurobi Optimizer 13.0.2

A valid Gurobi license is required to run the optimization models.

## Repository Structure

```
SoftPUOCT/
│
├── src/
│   ├── config.py
│   ├── paper_results.py
│   ├── data_pipeline.py
│   ├── experiment_engine.py
│   ├── metrics.py
│   ├── result_io.py
│   ├── solver_monitor.py
│   ├── oct.py
│   └── softPUOCT.py
│
├── exp[data_pipeline.py](src%2Fdata_pipeline.py)eriments/
│   ├── run_oct.py
│   └── run_softpuoct.py
│
├── data/
│   └── benchmark datasets
│
├── results/
│
├── requirements.txt
└── README.md
```
## Dataset
The datasets used in the experiments are included in the `data/` directory.

The benchmark datasets include:

- breast_cancer_diagnostic
- climate_model
- house-votes-84
- ionosphere
- monk1
- monk2
- monk3
- parkinsons
- sonar
- spect
- tic-tac-toe
- wholesale

Each dataset is a binary classification dataset with a target column:
```
target
```
## Running Experiments

### Run SoftPUOCT

```bash
python experiments/run_softpuoct.py
```
### Run OCT baseline

```bash
python experiments/run_oct.py
```

The experimental pipeline includes:

1. Outer 5-fold cross-validation
2. Inner validation for parameter selection
3. Positive-unlabeled sample generation
4. Model training using Gurobi optimization
5. Evaluation on held-out test folds


## Configuration

Experiment settings can be modified in:
```
src/config.py
```

Important parameters include:

- random seed
- number of cross-validation folds
- PU masking ratio
- EM iteration number
- optimization time limits

Default settings:

- Random seed: 42
- Outer folds: 5
- PU masking ratio: 50%


## Results
The generated experimental results are saved in:

```
results/
```

The output includes:

- prediction results
- fold-level evaluation metrics
- optimization statistics
- solver information
- trained model information

## Requirements
Main dependencies:
```
numpy
pandas
scikit-learn
scipy
gurobipy
```
## License
The code is released for academic and research purposes.
