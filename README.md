# Machine Learning Assignment 2

## Decision Trees, Bagging, Multiclass Boosting and Bias–Variance Analysis

### Overview

This project explores decision trees, bootstrap aggregation (bagging), multiclass SAMME boosting, and bias–variance analysis using the Olivetti Faces dataset.

The dataset contains 400 grayscale images of 40 individuals, with 10 images per individual. Each image has a resolution of 64 × 64 pixels.

All experiments use raw pixel intensities as features. Each image is flattened into a 4096-dimensional feature vector. No dimensionality reduction techniques, such as PCA, are applied.

### Dataset and Experimental Protocol

- Dataset: Olivetti Faces (scikit-learn)
- Identities: 40
- Images per identity: 10
- Training: 6 images per identity (240 total)
- Validation: 2 images per identity (80 total)
- Test: 2 images per identity (80 total)
- Random seed: `26014` (derived from roll number `M26CSA014`)
- Masked evaluation: The central 16 × 16 pixel region of each test image is set to zero.

The original dataset split is stored in `data/splits/olivetti_split.npz`.

Model selection is performed using the validation set. The test set is reserved for final evaluation.

### Implemented Experiments

**Question 1: Decision Trees and Bagging**

- Hyperparameter search for decision trees
- Bagging with bootstrap resampling and majority voting
- Out-of-bag (OOB) accuracy estimation
- Pairwise tree disagreement and vote-margin analysis
- Clean and masked test evaluation

**Question 2: Multiclass SAMME Boosting**

- Manual demonstration of SAMME weight updates
- Weighted decision-tree training
- Multiclass learner weight calculation
- SAMME weighted voting
- Training weight diagnostics
- Comparison between bagging and boosting
- Robustness analysis using masked images

**Question 3: Bias–Variance Analysis**

- 30 resampled training sets, each containing six bootstrap draws from every one of the 40 identities
- Squared bias and variance calculated using 40-dimensional class-score vectors
- Comparison across decision-tree depths
- Analysis of bagging's effect on variance reduction
- Model stability and robustness comparison

### Installation

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The experiments were developed using Python 3.14.7.

### Reproducing the Experiments

The repository provides `reproduce.sh` for running the experiments on macOS or Linux.

After installing the required dependencies, make the script executable:

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

The script reuses the saved dataset split and executes the experiment scripts in the required order.

The Q3 experiments are computationally expensive because they involve repeated model fitting on resampled training sets. Saved experimental logs and checkpoints are included to support result verification.

**Note:** The `plots` option regenerates the Q1(c) and Q3 plots directly from saved results. Other figures are produced by their respective experiment scripts.

For individual script commands and output descriptions, refer to the following section.

### Reproduction Commands

Run the following commands from the project root after activating the virtual environment.

**Prepare the fixed dataset split (only if the supplied split is missing):**

```bash
python src/data_setup.py
```

The supplied `data/splits/olivetti_split.npz` should be reused whenever available to maintain consistency with the original experiments.

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

The Q3 experiments are computationally expensive because the models are refitted across 30 resampled training sets.

The Q3(b) script supports saved checkpoints. When valid checkpoints are present, previously completed repetitions are reused rather than refitted. To reproduce every fit from scratch, the corresponding Q3(b) checkpoints must be removed before rerunning the script.

### Selected Model Results

The following results were obtained using models selected on the validation set and evaluated on the fixed test set.

| Model | Validation Accuracy | Clean Test Accuracy | Masked Test Accuracy |
|---|---:|---:|---:|
| Single Decision Tree | 60.00% | 60.00% | 60.00% |
| Bagging (100 Trees) | 90.00% | 91.25% | 65.00% |
| SAMME Boosting (100 Rounds) | 32.50% | 31.25% | 26.25% |

### Bias–Variance Results

Bias and variance were measured across 30 independently resampled training sets, while the evaluation test set remained fixed.

| Model | Squared Bias | Variance | Mean Test Accuracy |
|---|---:|---:|---:|
| Single Decision Tree | 0.3550 | 0.6367 | 50.42% |
| Bagging | 0.5246 | 0.0257 | 77.83% |
| SAMME Boosting | 0.8824 | 0.0169 | 33.79% |

The squared bias and variance were calculated using normalized 40-dimensional class-score vectors.

The accuracy values in this table represent the mean performance across 30 refits, whereas the selected model results above represent the frozen models trained on the original dataset split.

### Saved Outputs

- `configs/` — Selected model configurations
- `data/splits/` — Fixed dataset split, metadata, and saved bootstrap indices
- `results/logs/` — Evaluation tables, experiment logs, saved predictions, and checkpoints
- `results/figures/` — Plots and figures used for experimental analysis and reporting
- `answers/` — Detailed analyses for Questions 1, 2, and 3
- `src/` — Python implementations of the experiments

### Reproducibility Notes

All experiments use fixed random seeds. The original training, validation, and test split remains unchanged across the experiments.

Bagging uses bootstrap resampling with replacement and majority voting. In case of a tie, the smallest numerical class label is selected.

The multiclass SAMME algorithm is implemented manually using weighted decision-tree learners. No prebuilt AdaBoost classifier is used.

For Q3, the same 30 resampled training sets are reused across the relevant model comparisons to maintain a consistent experimental setup.

The depth analysis compares `max_depth=4`, `max_depth=16`, and `max_depth=None`, with `min_samples_leaf=4`. Since `None` represents unrestricted tree growth, it serves as the deepest available setting.

The repository includes saved experimental results, configurations, and checkpoints so that the reported results can be inspected without rerunning every experiment.