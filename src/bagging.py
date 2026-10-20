
import json
import time
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from common import load_data, ROOT


ENSEMBLE_SIZES = [1, 5, 10, 25, 50, 100, 200]
N_CLASSES = 40


def get_vote_counts(predictions):
    # Count the votes received by each class
    return np.stack(
        [
            np.sum(predictions == c, axis=0)
            for c in range(N_CLASSES)
        ],
        axis=1,
    )


def majority_vote(predictions):
    # Select the class with the highest vote count
    vote_counts = get_vote_counts(predictions)
    return np.argmax(vote_counts, axis=1)


def vote_statistics(predictions):
    # Calculate ensemble vote margins and tree disagreement
    n_trees = predictions.shape[0]
    vote_counts = get_vote_counts(predictions)
    vote_fractions = vote_counts / n_trees

    sorted_fractions = np.sort(vote_fractions, axis=1)

    margins = (
        sorted_fractions[:, -1]
        - sorted_fractions[:, -2]
    )

    mean_margin = float(np.mean(margins))

    if n_trees < 2:
        disagreement = float("nan")
    else:
        # Compute mean disagreement over all pairs of trees
        total_disagreement = 0.0
        num_pairs = 0

        for i in range(n_trees):
            for j in range(i + 1, n_trees):
                total_disagreement += np.mean(
                    predictions[i] != predictions[j]
                )
                num_pairs += 1

        disagreement = float(
            total_disagreement / num_pairs
        )

    return mean_margin, disagreement


def oob_accuracy(models, bootstrap_indices, X, y):
    # Predict each sample using only trees that did not train on it
    n = len(y)

    vote_counts = np.zeros(
        (n, N_CLASSES), dtype=np.int32
    )

    coverage = np.zeros(n, dtype=np.int32)

    for model, sampled_indices in zip(
        models, bootstrap_indices
    ):
        in_bag = np.zeros(n, dtype=bool)
        in_bag[np.unique(sampled_indices)] = True

        oob_indices = np.flatnonzero(~in_bag)

        if len(oob_indices) == 0:
            continue

        predictions = model.predict(X[oob_indices])

        vote_counts[oob_indices, predictions] += 1
        coverage[oob_indices] += 1

    covered = coverage > 0

    if not np.any(covered):
        return float("nan"), 0

    oob_predictions = np.argmax(
        vote_counts[covered], axis=1
    )

    accuracy = accuracy_score(
        y[covered], oob_predictions
    )

    return float(accuracy), int(covered.sum())


def main():
    # Load the fixed dataset split
    data = load_data()

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    X_masked = data["X_test_masked"]
    seed = int(data["seed"])

    # Load the best tree configuration from Q1(a)
    with open(ROOT / "configs" / "best_tree.json") as f:
        tree_config = json.load(f)

    print("\nFrozen decision tree configuration:")
    print("max_depth:", tree_config["max_depth"])
    print("min_samples_leaf:", tree_config["min_samples_leaf"])

    # Initialize random generator for bootstrap sampling
    rng = np.random.default_rng(seed + 1000)

    n = len(y_train)
    max_trees = max(ENSEMBLE_SIZES)

    # Calculate theoretical bootstrap diversity
    theoretical_unique = n * (1 - (1 - 1 / n) ** n)

    models = []
    bootstrap_indices = []
    history = []

    start_time = time.perf_counter()

    # Train 200 bootstrap decision trees
    for b in range(max_trees):
        sampled_indices = rng.integers(
            0, n, size=n
        )

        model = DecisionTreeClassifier(
            max_depth=tree_config["max_depth"],
            min_samples_leaf=tree_config["min_samples_leaf"],
            random_state=seed + 1001 + b,
        )

        model.fit(
            X_train[sampled_indices],
            y_train[sampled_indices]
        )

        models.append(model)
        bootstrap_indices.append(sampled_indices)

        B = b + 1

        if B not in ENSEMBLE_SIZES:
            continue

        # Generate predictions from the first B trees
        train_predictions = np.array([
            m.predict(X_train) for m in models
        ])

        val_predictions = np.array([
            m.predict(X_val) for m in models
        ])

        test_predictions = np.array([
            m.predict(X_test) for m in models
        ])

        # Compute majority vote predictions
        train_vote = majority_vote(train_predictions)
        val_vote = majority_vote(val_predictions)
        test_vote = majority_vote(test_predictions)

        # Calculate training, validation and test accuracy
        train_accuracy = float(
            accuracy_score(y_train, train_vote)
        )

        val_accuracy = float(
            accuracy_score(y_val, val_vote)
        )

        test_accuracy = float(
            accuracy_score(y_test, test_vote)
        )

        # Calculate mean distinct examples per bootstrap sample
        observed_unique = float(np.mean([
            len(np.unique(indices))
            for indices in bootstrap_indices
        ]))

        # Calculate statistics needed for Q1(c)
        margin, disagreement = vote_statistics(
            val_predictions
        )

        oob_score, oob_covered = oob_accuracy(
            models,
            bootstrap_indices,
            X_train,
            y_train
        )

        elapsed = time.perf_counter() - start_time

        history.append({
            "B": B,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "oob_accuracy": oob_score,
            "oob_covered_samples": oob_covered,
            "mean_distinct_bootstrap_examples": observed_unique,
            "theoretical_distinct_examples": float(
                theoretical_unique
            ),
            "mean_vote_margin": margin,
            "pairwise_disagreement": disagreement,
            "training_time_seconds": elapsed,
        })

        print(f"Completed B={B}")

    # Save results for all ensemble sizes
    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(history)

    results.to_csv(
        log_dir / "bagging_results.csv",
        index=False
    )

    # Select the best ensemble using validation accuracy
    best_row = sorted(
        history,
        key=lambda row: (
            -row["validation_accuracy"],
            row["B"]
        )
    )[0]

    selected_B = int(best_row["B"])
    selected_models = models[:selected_B]
    selected_indices = bootstrap_indices[:selected_B]

    # Evaluate the selected ensemble on clean test images
    clean_predictions = np.array([
        model.predict(X_test)
        for model in selected_models
    ])

    clean_vote = majority_vote(clean_predictions)

    # Evaluate the selected ensemble on masked test images
    masked_predictions = np.array([
        model.predict(X_masked)
        for model in selected_models
    ])

    masked_vote = majority_vote(masked_predictions)

    # Calculate clean and masked vote margins
    clean_margin, _ = vote_statistics(
        clean_predictions
    )

    masked_margin, _ = vote_statistics(
        masked_predictions
    )

    final_metrics = {
        "selected_B": selected_B,
        "selection_metric": "validation_accuracy",
        "validation_accuracy": float(
            best_row["validation_accuracy"]
        ),
        "clean_test_accuracy": float(
            accuracy_score(y_test, clean_vote)
        ),
        "masked_test_accuracy": float(
            accuracy_score(y_test, masked_vote)
        ),
        "clean_test_mean_vote_margin": clean_margin,
        "masked_test_mean_vote_margin": masked_margin,
        "tree_config": tree_config,
        "seed": seed,
    }

    # Save the selected ensemble configuration
    config_dir = ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    with open(config_dir / "best_bagging.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Save bootstrap indices for reproducibility
    np.savez(
        ROOT / "data" / "splits" / "bagging_bootstrap_indices.npz",
        **{
            f"tree_{i + 1}": indices
            for i, indices in enumerate(selected_indices)
        }
    )

    # Display all bagging results
    print("\nBagging results:")
    print(results.to_string(index=False))

    # Compare observed and theoretical bootstrap diversity
    print("\nBootstrap diversity comparison:")
    print(f"Training set size: {n}")
    print(f"Expected distinct examples: {theoretical_unique:.4f}")

    for row in history:
        print(
            f"B={row['B']:3d} | "
            f"Observed={row['mean_distinct_bootstrap_examples']:.4f} | "
            f"Expected={theoretical_unique:.4f}"
        )

    print("\nSelected bagging model:")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
