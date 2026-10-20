
import json
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from common import load_data, ROOT


def get_tree_statistics(model, n_training_samples):
    """
    Compute structural statistics for a fitted decision tree.
    """

    tree = model.tree_

    # A leaf has no left child.
    leaf_mask = tree.children_left == -1

    # Number of training samples reaching each leaf.
    leaf_sample_counts = tree.n_node_samples[leaf_mask]

    n_leaves = int(np.sum(leaf_mask))

    mean_samples_per_leaf = (
        n_training_samples / n_leaves
    )

    fraction_single_sample_leaves = (
        np.sum(leaf_sample_counts == 1) / n_leaves
    )

    return {
        "n_leaves": n_leaves,
        "actual_depth": int(model.get_depth()),
        "mean_samples_per_leaf": float(mean_samples_per_leaf),
        "fraction_single_sample_leaves": float(
            fraction_single_sample_leaves
        ),
    }


def main():

    # Load the fixed training, validation, and test splits.
    data = load_data()

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    seed = int(data["seed"])

    depths = [2, 4, 8, 16, None]
    leaf_sizes = [1, 2, 4]

    results = []

    # Hyperparameter search: 5 depths x 3 leaf sizes.
    for depth in depths:
        for leaf_size in leaf_sizes:

            model = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=leaf_size,
                random_state=seed,
            )

            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            stats = get_tree_statistics(
                model,
                len(y_train)
            )

            results.append({
                "max_depth": depth,
                "min_samples_leaf": leaf_size,
                "train_accuracy": float(
                    accuracy_score(y_train, train_pred)
                ),
                "validation_accuracy": float(
                    accuracy_score(y_val, val_pred)
                ),
                **stats,
            })

    # Select by validation accuracy only.
    # For ties, prefer shallower trees, then larger leaves.
    results.sort(
        key=lambda r: (
            -r["validation_accuracy"],
            float("inf") if r["max_depth"] is None
            else r["max_depth"],
            -r["min_samples_leaf"],
        )
    )

    best = results[0]

    # Refit the selected tree on the original training set.
    # Do not train on validation or test data.
    final_model = DecisionTreeClassifier(
        max_depth=best["max_depth"],
        min_samples_leaf=best["min_samples_leaf"],
        random_state=seed,
    )

    final_model.fit(X_train, y_train)

    # Test set is evaluated only after selecting the model.
    test_accuracy = accuracy_score(
        y_test,
        final_model.predict(X_test)
    )

    best["test_accuracy"] = float(test_accuracy)

    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save all 15 configurations.
    pd.DataFrame(results).to_csv(
        log_dir / "decision_tree_results.csv",
        index=False
    )

    config_dir = ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    with open(config_dir / "best_tree.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\nAll decision tree configurations:")
    print(pd.DataFrame(results).to_string(index=False))

    print("\nSelected Decision Tree:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
