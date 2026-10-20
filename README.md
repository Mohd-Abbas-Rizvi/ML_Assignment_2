# Machine Learning Assignment 2
## Decision Trees, Bagging, Multiclass Boosting and Bias–Variance Analysis

### Overview

This project investigates decision trees, bootstrap aggregation (bagging), multiclass SAMME boosting, and bias–variance decomposition using the Olivetti Faces dataset.

The dataset contains 400 grayscale images from 40 identities, with 10 images per identity. Each image has a resolution of 64 × 64 pixels.

The experiments use raw flattened pixel features (4096 dimensions), without PCA or other dimensionality reduction.

### Dataset and Experimental Protocol

- Dataset: Olivetti Faces (scikit-learn)
- Identities: 40
- Images per identity: 10
- Training: 6 images per identity (240 total)
- Validation: 2 images per identity (80 total)
- Test: 2 images per identity (80 total)
- Random seed: 26014
- Masked evaluation: central 16 × 16 image region set to zero

The original dataset split is stored in `data/splits/olivetti_split.npz`.

All model selection is performed using the validation set. The test set is used for final evaluation.

### Implemented Experiments

**Question 1 — Decision Trees and Bagging**

- Decision tree hyperparameter search
- Bagging with bootstrap resampling and majority voting
- Out-of-bag accuracy estimation
- Pairwise tree disagreement and vote-margin analysis
- Clean and masked test evaluation

**Question 2 — Multiclass SAMME Boosting**

- Manual SAMME weight-update demonstration
- Weighted decision-tree training
- Multiclass learner-weight calculation
- SAMME weighted voting
- Training-weight diagnostics
- Bagging versus boosting comparison
- Clean and masked robustness analysis

**Question 3 — Bias–Variance Analysis**

- 30 identity-wise resampled training sets
- Squared bias and variance calculation using 40-dimensional class-score vectors
- Comparison across tree depths
- Bagging variance-reduction analysis
- Model stability and robustness comparison

### Installation

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Reproduction Commands

Run the following commands from the project root.

**Prepare the fixed dataset split (only if the supplied split is missing):**

```bash
python src/data_setup.py
```

The supplied `data/splits/olivetti_split.npz` should normally be reused rather than regenerated.

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

Q3 experiments are computationally expensive because they repeat model fitting across 30 resampled training datasets.

### Selected Model Results

| Model | Validation Accuracy | Clean Test Accuracy | Masked Test Accuracy |
|---|---:|---:|---:|
| Single Decision Tree | 60.00% | 60.00% | 60.00% |
| Bagging (100 Trees) | 90.00% | 91.25% | 65.00% |
| SAMME Boosting (100 Rounds) | 32.50% | 31.25% | 26.25% |

### Bias–Variance Results

Measured using 30 refits on identity-wise resampled training datasets.

| Model | Squared Bias | Variance | Mean Test Accuracy |
|---|---:|---:|---:|
| Single Decision Tree | 0.3550 | 0.6367 | 50.42% |
| Bagging | 0.5246 | 0.0257 | 77.83% |
| SAMME Boosting | 0.8824 | 0.0169 | 33.79% |

### Saved Outputs

- `configs/` — Selected model configurations
- `data/splits/` — Dataset indices and bootstrap samples
- `results/logs/` — Evaluation tables, prediction logs and experiment checkpoints
- `results/figures/` — Figures used for analysis and reporting
- `report/` — Final CVPR-style report

### Reproducibility Notes

All experiments use fixed random seeds. The original dataset split is held constant across experiments.

The boosting algorithm is implemented manually using weighted decision-tree learners. No prebuilt AdaBoost classifier is used.

The Q3 experiments reuse the same 30 resampled datasets for model comparisons.

The repository includes saved experiment results to permit verification without rerunning every computationally expensive experiment.