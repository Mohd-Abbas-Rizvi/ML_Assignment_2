# Machine Learning Assignment 2

## Decision Trees, Bagging, Multiclass Boosting and Bias–Variance Analysis

### Overview

This project explores decision trees, bootstrap aggregation (bagging) multiclass SAMME boosting and bias–variance decomposition using the Olivetti Faces dataset.

The dataset has 400 grayscale images from 40 people. Each person has 10 images. Every image is 64 by 64 pixels in resolution.

All experiments use the raw pixel data as features. The data is flattened into 4096 dimensions. No dimensionality reduction techniques like PCA are applied.

### Dataset and Experimental Protocol

- Dataset: Olivetti Faces (scikit-learn)

- Identities: 40

- Images per identity: 10

- Training: 6 images per identity (240 total)

- Validation: 2 images per identity (80 total)

- Test: 2 images per identity (80 total)

- seed: 26014(from my rollno ie M26CSA014 numeric value is extracted)

- Masked evaluation: central 16 × 16 region of each image is set to zero

The original split of the dataset is stored in `data/splits/olivetti_split.npz`.

Model selection is done using the validation set. The test set is used for final evaluation.

### Implemented Experiments

**Question 1. Decision Trees and Bagging**

- Hyperparameter search for decision trees

- Bagging with bootstrap resampling and majority voting

- Out-of-bag accuracy estimation

- Pairwise tree disagreement and vote-margin analysis

- Clean and masked test evaluation

**Question 2. Multiclass SAMME Boosting**

- Manual demonstration of SAMME weight updates

- Weighted decision-tree training

- Multiclass learner weight calculation

- SAMME weighted voting

- Training weight diagnostics

- Comparison between bagging and boosting

- Robustness analysis with masked images

**Question 3. Bias–Variance Analysis**

- 30 resampled training sets, each based on one identity

- Squared bias and variance calculated using 40-dimensional class-score vectors

- Comparison across tree depths

- Bagging’s effect on variance reduction

- Model stability and robustness comparison

### Installation

Create and activate a Python virtual environment:

```bash

python3 -m venv.venv

source.venv/bin/activate

python -m pip install -r requirements.txt

```
### Reproducing the Experiments

The repository provides `reproduce.sh` for running the experiments on macOS or Linux.

First, create the Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Make the script executable:

```bash
chmod +x reproduce.sh
```

Run individual experiments:

```bash
./reproduce.sh q1
./reproduce.sh q2
./reproduce.sh q3
```

Run the complete assignment:

```bash
./reproduce.sh all
```

Regenerate plots from existing results without retraining:

```bash
./reproduce.sh plots
```

The script reuses the committed dataset split and executes the experiment scripts in the required order. Q3(a) and Q3(b) are computationally expensive, so saved logs and checkpoints are also included in the repository.

For individual script commands and output descriptions, refer to the detailed reproduction sections below.
### Reproduction Commands

Run the following commands from the project root.

**Prepare the fixed dataset split ( if the supplied split is missing):**

```bash

python src/data_setup.py

```

The supplied `data/splits/olivetti_split.npz` should be reused unless it is missing.

**Question 1:**

```bash

python src/decision_tree.py

python src/bagging.py

python src/bagging_plots.py

python src/q1d_comparison.py

```

**Question 2:**

```bash

python src/q2a_samme_demo.py

python src/adaboost.py

python src/q2c_comparison.py

python src/q2d_comparison.py

```

**Question 3:**

```bash

python src/bias_variance.py

python src/q3b_depth_analysis.py

python src/q3_plots.py

```

The Q3 experiments are computationally expensive because they run model fitting 30 times on resampled training sets.

### Selected Model Results

| Model | Validation Accuracy Clean Test Accuracy | Masked Test Accuracy |

|---|---:|---:|---:|

| Single Decision Tree | 60.00% | 60.00% | 60.00% |

| Bagging (100 Trees) | 90.00% | 91.25% | 65.00% |

| SAMME Boosting (100 Rounds) | 32.50% | 31.25% | 26.25% |

### Bias–Variance Results

Measured using 30 refits on identity-wise resampled training datasets.

| Model | Squared Bias | Variance | Mean Test Accuracy

|---|---:|---:|---:|

| Single Decision Tree | 0.3550 | 0.6367 | 50.42% |

| Bagging | 0.5246 | 0.0257 | 77.83% |

| SAMME Boosting | 0.8824 | 0.0169 | 33.79% |

### Saved Outputs

- `configs/`. Selected model configurations

- `data/splits/`. Dataset. Bootstrap samples

- `results/logs/`. Evaluation tables, prediction logs and experiment checkpoints

- `results/figures/`. Figures used for analysis and reporting

- `report/`. CVPR-style report

### Reproducibility Notes

All experiments use fixed random seeds. The original dataset split is kept constant across all experiments.

The boosting algorithm is implemented manually using weighted decision-tree learners. No prebuilt AdaBoost classifier is used.

The Q3 experiments reuse the 30 resampled datasets for comparisons.

The repository includes saved experiment results so that verification can be done without rerunning every experiment.