
import os
import json
import time
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from common import load_data, ROOT


DEPTHS = [4, 16, None]
N_REPETITIONS = 30
N_CLASSES = 40
B = 100


def depth_name(depth):
    # Convert depth into a filename-safe label
    return "None" if depth is None else str(depth)


def tree_scores(model, X):
    # Align decision tree probabilities to all 40 classes
    scores = np.zeros((len(X), N_CLASSES))

    scores[:, model.classes_.astype(int)] = (
        model.predict_proba(X)
    )

    return scores


def leaf_statistics(model):
    # Count samples assigned to terminal nodes
    tree = model.tree_
    leaf_mask = tree.children_left == -1

    counts = tree.n_node_samples[leaf_mask]
    n_leaves = len(counts)

    return {
        "n_leaves": int(n_leaves),
        "mean_leaf_occupancy": float(np.mean(counts)),
        "fraction_single_sample_leaves": float(
            np.mean(counts == 1)
        ),
        "actual_depth": int(model.get_depth())
    }


def fit_tree(X_train, y_train, X_test, depth, seed):
    # Train one tree with the fixed minimum leaf size
    model = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=4,
        random_state=seed
    )

    model.fit(X_train, y_train)

    return (
        tree_scores(model, X_test),
        leaf_statistics(model)
    )


def fit_bagging(X_train, y_train, X_test, depth, seed):
    # Train 100 bootstrap trees using the selected leaf size
    rng = np.random.default_rng(seed + 1000)

    n = len(y_train)
    n_test = len(X_test)

    votes = np.zeros(
        (n_test, N_CLASSES),
        dtype=np.int32
    )

    leaf_counts = []
    occupancies = []
    single_leaf_fractions = []

    for b in range(B):
        indices = rng.integers(
            0, n, size=n
        )

        model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=4,
            random_state=seed + 1001 + b
        )

        model.fit(
            X_train[indices],
            y_train[indices]
        )

        predictions = model.predict(X_test).astype(int)

        votes[np.arange(n_test), predictions] += 1

        stats = leaf_statistics(model)

        leaf_counts.append(stats["n_leaves"])
        occupancies.append(stats["mean_leaf_occupancy"])
        single_leaf_fractions.append(
            stats["fraction_single_sample_leaves"]
        )

    # Convert votes into 40-dimensional class-score vectors
    scores = votes / B

    # Compute pairwise disagreement without enumerating every pair
    agreement = np.sum(
        votes * (votes - 1),
        axis=1
    ) / (B * (B - 1))

    disagreement = float(
        np.mean(1.0 - agreement)
    )

    stats = {
        "mean_n_leaves": float(np.mean(leaf_counts)),
        "mean_leaf_occupancy": float(
            np.mean(occupancies)
        ),
        "fraction_single_sample_leaves": float(
            np.mean(single_leaf_fractions)
        ),
        "pairwise_disagreement": disagreement
    }

    return scores, stats


def calculate_metrics(scores, y_test):
    # Calculate squared bias and variance from all 30 refits
    mean_scores = np.mean(scores, axis=0)

    targets = np.eye(N_CLASSES)[
        y_test.astype(int)
    ]

    bias_squared = np.mean(
        np.sum(
            (mean_scores - targets) ** 2,
            axis=1
        )
    )

    variance = np.mean(
        np.sum(
            (scores - mean_scores[None, :, :]) ** 2,
            axis=2
        )
    )

    predictions = np.argmax(scores, axis=2)

    accuracies = np.mean(
        predictions == y_test[None, :],
        axis=1
    )

    return {
        "bias_squared": float(bias_squared),
        "variance": float(variance),
        "mean_test_accuracy": float(
            np.mean(accuracies)
        ),
        "std_test_accuracy": float(
            np.std(accuracies, ddof=1)
        )
    }


def save_checkpoint(path, tree_array, bag_array, records):
    # Save progress atomically so interrupted runs can resume
    temporary = path.with_name(
        path.stem + ".tmp.npz"
    )

    np.savez_compressed(
        temporary,
        tree_scores=tree_array,
        bagging_scores=bag_array,
        records=np.array(
            [json.dumps(r) for r in records]
        )
    )

    os.replace(temporary, path)


def main():
    # Load the fixed training and test datasets
    data = load_data()

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    seed = int(data["seed"])

    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the saved identity-wise resamples from Q3(a)
    saved = np.load(
        log_dir / "q3a_partial_results.npz"
    )

    sampled_sets = saved["sampled_indices"]

    if sampled_sets.shape != (N_REPETITIONS, len(y_train)):
        raise ValueError("Unexpected saved resampling shape.")

    all_results = []
    all_repetition_rows = []

    for depth in DEPTHS:
        name = depth_name(depth)

        print(f"\nStarting depth={name}", flush=True)

        checkpoint = (
            log_dir / f"q3b_checkpoint_depth_{name}.npz"
        )

        tree_array = np.full(
            (N_REPETITIONS, len(y_test), N_CLASSES),
            np.nan
        )

        bag_array = np.full(
            (N_REPETITIONS, len(y_test), N_CLASSES),
            np.nan
        )

        records = []

        # Restore any previously completed repetitions
        if checkpoint.exists():
            existing = np.load(checkpoint)

            tree_array = existing["tree_scores"]
            bag_array = existing["bagging_scores"]

            records = [
                json.loads(value)
                for value in existing["records"]
            ]

            print(
                f"Restored {len(records)} completed repetitions.",
                flush=True
            )

        completed = {
            int(row["repetition"])
            for row in records
        }

        for r in range(N_REPETITIONS):
            repetition = r + 1

            if repetition in completed:
                continue

            start = time.perf_counter()

            indices = sampled_sets[r]

            X_resampled = X_train[indices]
            y_resampled = y_train[indices]

            repetition_seed = seed + 10000 + r

            # Refit a single decision tree
            single_scores, tree_stats = fit_tree(
                X_resampled,
                y_resampled,
                X_test,
                depth,
                repetition_seed
            )

            # Refit a 100-tree bagged ensemble
            bag_scores, bag_stats = fit_bagging(
                X_resampled,
                y_resampled,
                X_test,
                depth,
                repetition_seed
            )

            tree_array[r] = single_scores
            bag_array[r] = bag_scores

            tree_accuracy = float(np.mean(
                np.argmax(single_scores, axis=1)
                == y_test
            ))

            bag_accuracy = float(np.mean(
                np.argmax(bag_scores, axis=1)
                == y_test
            ))

            records.append({
                "repetition": repetition,
                "depth": name,
                "tree_accuracy": tree_accuracy,
                "bagging_accuracy": bag_accuracy,
                "accuracy_improvement": (
                    bag_accuracy - tree_accuracy
                ),
                "tree_n_leaves": tree_stats["n_leaves"],
                "tree_mean_leaf_occupancy": (
                    tree_stats["mean_leaf_occupancy"]
                ),
                "tree_single_sample_leaf_fraction": (
                    tree_stats["fraction_single_sample_leaves"]
                ),
                "tree_actual_depth": tree_stats["actual_depth"],
                "bagging_mean_n_leaves": (
                    bag_stats["mean_n_leaves"]
                ),
                "bagging_mean_leaf_occupancy": (
                    bag_stats["mean_leaf_occupancy"]
                ),
                "bagging_pairwise_disagreement": (
                    bag_stats["pairwise_disagreement"]
                )
            })

            save_checkpoint(
                checkpoint,
                tree_array,
                bag_array,
                records
            )

            elapsed = time.perf_counter() - start

            print(
                f"Depth={name} | "
                f"Repetition={repetition}/30 | "
                f"Tree={tree_accuracy:.4f} | "
                f"Bagging={bag_accuracy:.4f} | "
                f"Time={elapsed:.1f}s",
                flush=True
            )

        if np.isnan(tree_array).any() or np.isnan(bag_array).any():
            raise RuntimeError(
                f"Incomplete repetitions at depth {name}"
            )

        tree_metrics = calculate_metrics(
            tree_array,
            y_test
        )

        bag_metrics = calculate_metrics(
            bag_array,
            y_test
        )

        depth_records = pd.DataFrame(records)

        # Calculate average leaf structure and disagreement
        common = {
            "depth": name,
            "mean_tree_leaf_occupancy": float(
                depth_records[
                    "tree_mean_leaf_occupancy"
                ].mean()
            ),
            "mean_tree_n_leaves": float(
                depth_records["tree_n_leaves"].mean()
            ),
            "mean_single_leaf_fraction": float(
                depth_records[
                    "tree_single_sample_leaf_fraction"
                ].mean()
            ),
            "mean_pairwise_disagreement": float(
                depth_records[
                    "bagging_pairwise_disagreement"
                ].mean()
            )
        }

        all_results.append({
            **common,
            "method": "Single Tree",
            **tree_metrics
        })

        all_results.append({
            **common,
            "method": "Bagging",
            **bag_metrics
        })

        all_repetition_rows.extend(records)

    result_df = pd.DataFrame(all_results)

    # Quantify variance reduction and accuracy improvement
    comparison = []

    for depth in DEPTHS:
        name = depth_name(depth)

        tree_row = result_df[
            (result_df["depth"] == name)
            & (result_df["method"] == "Single Tree")
        ].iloc[0]

        bag_row = result_df[
            (result_df["depth"] == name)
            & (result_df["method"] == "Bagging")
        ].iloc[0]

        comparison.append({
            "depth": name,
            "tree_bias_squared": tree_row["bias_squared"],
            "bagging_bias_squared": bag_row["bias_squared"],
            "tree_variance": tree_row["variance"],
            "bagging_variance": bag_row["variance"],
            "variance_removed": (
                tree_row["variance"]
                - bag_row["variance"]
            ),
            "variance_reduction_percent": (
                100 * (
                    tree_row["variance"]
                    - bag_row["variance"]
                ) / tree_row["variance"]
            ),
            "tree_test_accuracy": (
                tree_row["mean_test_accuracy"]
            ),
            "bagging_test_accuracy": (
                bag_row["mean_test_accuracy"]
            ),
            "accuracy_gain": (
                bag_row["mean_test_accuracy"]
                - tree_row["mean_test_accuracy"]
            ),
            "mean_tree_leaf_occupancy": (
                tree_row["mean_tree_leaf_occupancy"]
            ),
            "mean_tree_n_leaves": (
                tree_row["mean_tree_n_leaves"]
            ),
            "mean_pairwise_disagreement": (
                tree_row["mean_pairwise_disagreement"]
            )
        })

    # Save the final tables for Q3(b) and Q3(c)
    result_df.to_csv(
        log_dir / "q3b_bias_variance_by_depth.csv",
        index=False
    )

    pd.DataFrame(comparison).to_csv(
        log_dir / "q3b_depth_comparison.csv",
        index=False
    )

    pd.DataFrame(all_repetition_rows).to_csv(
        log_dir / "q3b_repetition_details.csv",
        index=False
    )

    print("\nQ3(b) Bias-Variance Results:")
    print(result_df.to_string(index=False))

    print("\nDepth Comparison:")
    print(pd.DataFrame(comparison).to_string(index=False))

    print("\nQ3(b) finished successfully.")


if __name__ == "__main__":
    main()
