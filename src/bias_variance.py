
import json
import time
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from common import load_data, ROOT
from adaboost import fit_samme, get_weighted_scores


N_REPETITIONS = 30
N_CLASSES = 40
METHODS = ["Single Tree", "Bagging", "SAMME"]


def resample_by_identity(X, y, rng):
    # Draw six samples with replacement from each identity
    selected_indices = []

    for identity in range(N_CLASSES):
        class_indices = np.flatnonzero(y == identity)

        if len(class_indices) != 6:
            raise ValueError(
                f"Identity {identity} has "
                f"{len(class_indices)} training images instead of 6."
            )

        sampled = rng.choice(
            class_indices,
            size=6,
            replace=True
        )

        selected_indices.extend(sampled.tolist())

    selected_indices = np.array(selected_indices, dtype=int)

    return (
        X[selected_indices],
        y[selected_indices],
        selected_indices
    )


def aligned_tree_probabilities(model, X):
    # Align probability columns to all 40 identities
    probabilities = np.zeros(
        (len(X), N_CLASSES),
        dtype=float
    )

    probabilities[:, model.classes_.astype(int)] = (
        model.predict_proba(X)
    )

    return probabilities


def fit_single_tree(X_train, y_train, X_test, config, seed):
    # Fit the frozen decision tree
    model = DecisionTreeClassifier(
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        random_state=seed
    )

    model.fit(X_train, y_train)

    return aligned_tree_probabilities(model, X_test)


def fit_bagging(X_train, y_train, X_test, config, B, seed):
    # Build B bootstrap trees using the frozen tree settings
    rng = np.random.default_rng(seed + 1000)

    n = len(y_train)
    vote_counts = np.zeros(
        (len(X_test), N_CLASSES),
        dtype=float
    )

    for b in range(B):
        indices = rng.integers(
            0, n, size=n
        )

        model = DecisionTreeClassifier(
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            random_state=seed + 1001 + b
        )

        model.fit(
            X_train[indices],
            y_train[indices]
        )

        predictions = model.predict(X_test).astype(int)

        vote_counts[
            np.arange(len(X_test)),
            predictions
        ] += 1

    # Convert class vote counts into class vote fractions
    return vote_counts / B


def fit_boosting(X_train, y_train, X_test, config, seed):
    # Fit frozen SAMME depth and round count
    depth = int(config["max_depth"])
    requested_rounds = int(config["actual_rounds"])

    result = fit_samme(
        X_train,
        y_train,
        depth=depth,
        max_rounds=requested_rounds,
        seed=seed + depth * 1000
    )

    if len(result["models"]) == 0:
        raise RuntimeError(
            "SAMME stopped without a valid weak learner."
        )

    # Get SAMME weighted votes for all test identities
    scores = get_weighted_scores(
        result["models"],
        result["alphas"],
        X_test
    )

    total_alpha = float(np.sum(result["alphas"]))

    if not np.isfinite(total_alpha) or total_alpha <= 0:
        raise RuntimeError("Invalid total SAMME learner weight.")

    return scores / total_alpha, result["actual_rounds"]


def calculate_bias_variance(score_vectors, y_test):
    # score_vectors shape: repetitions x test images x classes
    R, N, K = score_vectors.shape

    if K != N_CLASSES or N != len(y_test):
        raise ValueError("Unexpected score vector dimensions.")

    # Calculate mean class vector across repetitions
    mean_scores = np.mean(
        score_vectors,
        axis=0
    )

    # Construct one-hot targets
    targets = np.eye(N_CLASSES)[y_test.astype(int)]

    # Calculate squared bias
    bias_squared = float(np.mean(
        np.sum(
            (mean_scores - targets) ** 2,
            axis=1
        )
    ))

    # Calculate prediction variance
    variance = float(np.mean(
        np.sum(
            (score_vectors - mean_scores[None, :, :]) ** 2,
            axis=2
        )
    ))

    # Convert each repetition's class scores to predictions
    predictions = np.argmax(
        score_vectors,
        axis=2
    )

    accuracies = np.mean(
        predictions == y_test[None, :],
        axis=1
    )

    return {
        "bias_squared": bias_squared,
        "variance": variance,
        "mean_test_accuracy": float(
            np.mean(accuracies)
        ),
        "std_test_accuracy": float(
            np.std(accuracies, ddof=1)
        ),
        "min_test_accuracy": float(
            np.min(accuracies)
        ),
        "max_test_accuracy": float(
            np.max(accuracies)
        ),
        "repetitions": int(R),
    }, accuracies


def main():
    # Load the original fixed dataset split
    data = load_data()

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    base_seed = int(data["seed"])

    # Load previously selected configurations
    with open(ROOT / "configs" / "best_tree.json") as f:
        tree_config = json.load(f)

    with open(ROOT / "configs" / "best_bagging.json") as f:
        bagging_config = json.load(f)

    with open(ROOT / "configs" / "best_adaboost.json") as f:
        boosting_config = json.load(f)

    B = int(bagging_config["selected_B"])

    # Prepare results directories
    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    all_scores = {
        method: [] for method in METHODS
    }

    all_sampled_indices = []
    all_boosting_rounds = []
    start = time.perf_counter()

    # Repeat the resampling and refitting experiment 30 times
    for repetition in range(N_REPETITIONS):
        repetition_seed = base_seed + 10000 + repetition

        rng = np.random.default_rng(repetition_seed)

        X_resampled, y_resampled, sampled_indices = (
            resample_by_identity(X_train, y_train, rng)
        )

        all_sampled_indices.append(sampled_indices)

        # Refit single decision tree
        tree_scores = fit_single_tree(
            X_resampled,
            y_resampled,
            X_test,
            tree_config,
            repetition_seed
        )

        # Refit selected bagging model
        bagging_scores = fit_bagging(
            X_resampled,
            y_resampled,
            X_test,
            tree_config,
            B,
            repetition_seed
        )

        # Refit selected SAMME boosting model
        boosting_scores, actual_rounds = fit_boosting(
            X_resampled,
            y_resampled,
            X_test,
            boosting_config,
            repetition_seed
        )

        all_scores["Single Tree"].append(tree_scores)
        all_scores["Bagging"].append(bagging_scores)
        all_scores["SAMME"].append(boosting_scores)

        all_boosting_rounds.append(actual_rounds)

        # Save intermediate results after every repetition
        np.savez_compressed(
            log_dir / "q3a_partial_results.npz",
            tree_scores=np.array(
                all_scores["Single Tree"]
            ),
            bagging_scores=np.array(
                all_scores["Bagging"]
            ),
            boosting_scores=np.array(
                all_scores["SAMME"]
            ),
            sampled_indices=np.array(
                all_sampled_indices
            ),
            actual_boosting_rounds=np.array(
                all_boosting_rounds
            )
        )

        elapsed = time.perf_counter() - start

        print(
            f"Completed repetition "
            f"{repetition + 1}/{N_REPETITIONS} "
            f"| SAMME rounds={actual_rounds} "
            f"| elapsed={elapsed:.1f}s",
            flush=True
        )

    # Calculate bias, variance and accuracy statistics
    summary_rows = []
    accuracy_rows = []

    for method in METHODS:
        score_vectors = np.array(
            all_scores[method]
        )

        metrics, accuracies = calculate_bias_variance(
            score_vectors,
            y_test
        )

        summary_rows.append({
            "model": method,
            **metrics
        })

        for repetition, accuracy in enumerate(
            accuracies, start=1
        ):
            accuracy_rows.append({
                "model": method,
                "repetition": repetition,
                "test_accuracy": float(accuracy)
            })

    summary = pd.DataFrame(summary_rows)

    # Save the final Q3(a) results
    summary.to_csv(
        log_dir / "q3a_bias_variance.csv",
        index=False
    )

    pd.DataFrame(accuracy_rows).to_csv(
        log_dir / "q3a_repetition_accuracies.csv",
        index=False
    )

    # Verify that all saved class scores sum to one
    for method in METHODS:
        scores = np.array(all_scores[method])

        if not np.allclose(
            scores.sum(axis=2),
            1.0,
            atol=1e-8
        ):
            raise RuntimeError(
                f"Class scores do not sum to one: {method}"
            )

    print("\nQ3(a) Bias-Variance Results:")
    print(summary.to_string(index=False))

    print("\nTotal time:")
    print(f"{time.perf_counter() - start:.2f} seconds")

    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()
