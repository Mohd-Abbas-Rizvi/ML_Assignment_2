
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from common import load_data, ROOT


N_CLASSES = 40


def get_vote_counts(predictions):
    # Count predictions for each of the 40 identities
    return np.stack(
        [
            np.sum(predictions == c, axis=0)
            for c in range(N_CLASSES)
        ],
        axis=1
    )


def evaluate_models(tree, bagged_models, X, y):
    # Generate single tree predictions and probabilities
    tree_pred = tree.predict(X)
    tree_probs = tree.predict_proba(X)
    tree_confidence = np.max(tree_probs, axis=1)

    # Generate predictions from all bagged trees
    all_predictions = np.array([
        model.predict(X)
        for model in bagged_models
    ])

    vote_counts = get_vote_counts(all_predictions)
    bagging_pred = np.argmax(vote_counts, axis=1)

    # Calculate confidence as winning vote fraction
    bagging_confidence = (
        np.max(vote_counts, axis=1) / len(bagged_models)
    )

    return {
        "tree_pred": tree_pred,
        "tree_confidence": tree_confidence,
        "bagging_pred": bagging_pred,
        "bagging_confidence": bagging_confidence,
        "vote_counts": vote_counts,
        "tree_accuracy": float(
            accuracy_score(y, tree_pred)
        ),
        "bagging_accuracy": float(
            accuracy_score(y, bagging_pred)
        ),
        "tree_mean_confidence": float(
            np.mean(tree_confidence)
        ),
        "bagging_mean_confidence": float(
            np.mean(bagging_confidence)
        ),
    }


def nonzero_votes(votes):
    # Display vote counts only for classes receiving votes
    return {
        str(i): int(count)
        for i, count in enumerate(votes)
        if count > 0
    }


def save_examples(data, clean_results, masked_results, output_dir):
    # Find cases corrected by bagging on clean test images
    y = data["y_test"]

    tree_pred = clean_results["tree_pred"]
    bagging_pred = clean_results["bagging_pred"]

    corrected = np.flatnonzero(
        (tree_pred != y) & (bagging_pred == y)
    )

    both_wrong = np.flatnonzero(
        (tree_pred != y) & (bagging_pred != y)
    )

    # Select two examples from each category
    corrected = corrected[:2]
    both_wrong = both_wrong[:2]

    selected = [
        ("Bagging corrects tree", int(i))
        for i in corrected
    ] + [
        ("Both models wrong", int(i))
        for i in both_wrong
    ]

    print("\nExample counts:")
    print("Bagging corrects tree:", len(corrected))
    print("Both models wrong:", len(both_wrong))

    if len(selected) == 0:
        print("No qualifying examples found.")
        return

    rows = []

    for category, i in selected:
        votes = clean_results["vote_counts"][i]

        row = {
            "category": category,
            "test_index": i,
            "dataset_index": int(data["test_idx"][i]),
            "true_class": int(y[i]),
            "tree_prediction": int(tree_pred[i]),
            "bagging_prediction": int(bagging_pred[i]),
            "tree_confidence": float(
                clean_results["tree_confidence"][i]
            ),
            "bagging_confidence": float(
                clean_results["bagging_confidence"][i]
            ),
            "bagging_votes": json.dumps(
                nonzero_votes(votes)
            ),
            "masked_tree_prediction": int(
                masked_results["tree_pred"][i]
            ),
            "masked_bagging_prediction": int(
                masked_results["bagging_pred"][i]
            ),
        }

        rows.append(row)

        print(f"\n{category} | Test image {i}")
        print("True identity:", row["true_class"])
        print("Tree prediction:", row["tree_prediction"])
        print("Bagging prediction:", row["bagging_prediction"])
        print("Tree confidence:", row["tree_confidence"])
        print("Bagging confidence:", row["bagging_confidence"])
        print("Bagging vote counts:", row["bagging_votes"])

    pd.DataFrame(rows).to_csv(
        output_dir / "q1d_examples.csv",
        index=False
    )

    # Plot each selected face in clean and masked form
    fig, axes = plt.subplots(
        len(selected), 2,
        figsize=(7, 3 * len(selected)),
        squeeze=False
    )

    for row_idx, (category, i) in enumerate(selected):
        clean_image = data["X_test"][i].reshape(64, 64)
        masked_image = data["X_test_masked"][i].reshape(64, 64)

        axes[row_idx, 0].imshow(
            clean_image, cmap="gray", vmin=0, vmax=1
        )

        axes[row_idx, 0].set_title(
            f"{category}\n"
            f"True={y[i]}, Tree={tree_pred[i]}, "
            f"Bagging={bagging_pred[i]}"
        )

        axes[row_idx, 1].imshow(
            masked_image, cmap="gray", vmin=0, vmax=1
        )

        axes[row_idx, 1].set_title(
            f"Masked\n"
            f"Tree={masked_results['tree_pred'][i]}, "
            f"Bagging={masked_results['bagging_pred'][i]}"
        )

        axes[row_idx, 0].axis("off")
        axes[row_idx, 1].axis("off")

    fig.tight_layout()
    fig.savefig(
        ROOT / "results" / "figures" / "q1d_examples.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


def main():
    # Load saved dataset split
    data = load_data()

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    X_masked = data["X_test_masked"]
    y_test = data["y_test"]
    seed = int(data["seed"])

    # Read configurations chosen using validation accuracy
    with open(ROOT / "configs" / "best_tree.json") as f:
        tree_config = json.load(f)

    with open(ROOT / "configs" / "best_bagging.json") as f:
        bagging_config = json.load(f)

    selected_B = int(bagging_config["selected_B"])

    # Train the selected single decision tree
    tree = DecisionTreeClassifier(
        max_depth=tree_config["max_depth"],
        min_samples_leaf=tree_config["min_samples_leaf"],
        random_state=seed
    )

    tree.fit(X_train, y_train)

    # Reload the exact bootstrap indices used in Q1(b)
    saved = np.load(
        ROOT / "data" / "splits"
        / "bagging_bootstrap_indices.npz"
    )

    bagged_models = []

    for b in range(selected_B):
        indices = saved[f"tree_{b + 1}"]

        model = DecisionTreeClassifier(
            max_depth=tree_config["max_depth"],
            min_samples_leaf=tree_config["min_samples_leaf"],
            random_state=seed + 1001 + b
        )

        model.fit(
            X_train[indices],
            y_train[indices]
        )

        bagged_models.append(model)

    print(f"Reconstructed {selected_B} bagged trees.")

    # Evaluate clean test images
    clean = evaluate_models(
        tree, bagged_models, X_test, y_test
    )

    # Evaluate masked test images
    masked = evaluate_models(
        tree, bagged_models, X_masked, y_test
    )

    # Prepare clean versus masked comparison
    comparison = pd.DataFrame([
        {
            "condition": "Clean",
            "tree_accuracy": clean["tree_accuracy"],
            "bagging_accuracy": clean["bagging_accuracy"],
            "tree_mean_confidence": clean["tree_mean_confidence"],
            "bagging_mean_confidence": clean["bagging_mean_confidence"]
        },
        {
            "condition": "Masked",
            "tree_accuracy": masked["tree_accuracy"],
            "bagging_accuracy": masked["bagging_accuracy"],
            "tree_mean_confidence": masked["tree_mean_confidence"],
            "bagging_mean_confidence": masked["bagging_mean_confidence"]
        }
    ])

    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    (ROOT / "results" / "figures").mkdir(
        parents=True, exist_ok=True
    )

    comparison.to_csv(
        log_dir / "q1d_comparison.csv",
        index=False
    )

    # Save per-image predictions for reproducibility
    prediction_table = pd.DataFrame({
        "test_index": np.arange(len(y_test)),
        "dataset_index": data["test_idx"],
        "true_class": y_test,
        "clean_tree_prediction": clean["tree_pred"],
        "clean_bagging_prediction": clean["bagging_pred"],
        "clean_tree_confidence": clean["tree_confidence"],
        "clean_bagging_confidence": clean["bagging_confidence"],
        "masked_tree_prediction": masked["tree_pred"],
        "masked_bagging_prediction": masked["bagging_pred"],
        "masked_tree_confidence": masked["tree_confidence"],
        "masked_bagging_confidence": masked["bagging_confidence"]
    })

    prediction_table.to_csv(
        log_dir / "q1d_predictions.csv",
        index=False
    )

    print("\nQ1(d) Clean vs Masked comparison:")
    print(comparison.to_string(index=False))

    # Save examples illustrating error correction and failures
    save_examples(
        data, clean, masked, log_dir
    )

    print("\nQ1(d) files saved successfully.")


if __name__ == "__main__":
    main()
