
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from common import load_data, ROOT
from adaboost import fit_samme, get_weighted_scores
from bagging import get_vote_counts


N_CLASSES = 40


def build_models(data, tree_config, bagging_config, boost_config):
    # Reconstruct the frozen single decision tree
    seed = int(data["seed"])
    X_train = data["X_train"]
    y_train = data["y_train"]

    tree = DecisionTreeClassifier(
        max_depth=tree_config["max_depth"],
        min_samples_leaf=tree_config["min_samples_leaf"],
        random_state=seed,
    )
    tree.fit(X_train, y_train)

    # Reconstruct the selected bagging ensemble
    selected_B = int(bagging_config["selected_B"])

    saved = np.load(
        ROOT / "data" / "splits" / "bagging_bootstrap_indices.npz"
    )

    bagged_models = []

    for b in range(selected_B):
        indices = saved[f"tree_{b + 1}"]

        model = DecisionTreeClassifier(
            max_depth=tree_config["max_depth"],
            min_samples_leaf=tree_config["min_samples_leaf"],
            random_state=seed + 1001 + b,
        )

        model.fit(X_train[indices], y_train[indices])
        bagged_models.append(model)

    # Reconstruct the frozen SAMME boosting ensemble
    depth = int(boost_config["max_depth"])
    rounds = int(boost_config["actual_rounds"])

    boost_run = fit_samme(
        X_train,
        y_train,
        depth=depth,
        max_rounds=rounds,
        seed=seed + depth * 1000,
    )

    if boost_run["actual_rounds"] != rounds:
        raise RuntimeError("Boosting reconstruction did not match.")

    return tree, bagged_models, boost_run


def evaluate(tree, bagged_models, boost_run, X, y):
    # Predict using the single tree
    tree_predictions = tree.predict(X)
    tree_probabilities = tree.predict_proba(X)
    tree_confidence = np.max(tree_probabilities, axis=1)

    # Predict using bagging majority votes
    bag_tree_predictions = np.array([
        model.predict(X) for model in bagged_models
    ])

    bag_votes = get_vote_counts(bag_tree_predictions)
    bag_predictions = np.argmax(bag_votes, axis=1)

    bag_confidence = (
        np.max(bag_votes, axis=1) / len(bagged_models)
    )

    # Predict using SAMME weighted votes
    boost_scores = get_weighted_scores(
        boost_run["models"],
        boost_run["alphas"],
        X,
    )

    boost_predictions = np.argmax(boost_scores, axis=1)

    # Normalize SAMME scores as specified in the assignment
    total_alpha = float(np.sum(boost_run["alphas"]))
    boost_normalized_scores = boost_scores / total_alpha

    boost_confidence = np.max(
        boost_normalized_scores, axis=1
    )

    return {
        "tree_predictions": tree_predictions,
        "bag_predictions": bag_predictions,
        "boost_predictions": boost_predictions,
        "tree_confidence": tree_confidence,
        "bag_confidence": bag_confidence,
        "boost_confidence": boost_confidence,
        "bag_votes": bag_votes,
        "bag_tree_predictions": bag_tree_predictions,
        "boost_scores": boost_scores,
        "boost_normalized_scores": boost_normalized_scores,
        "metrics": {
            "tree_accuracy": float(
                accuracy_score(y, tree_predictions)
            ),
            "bagging_accuracy": float(
                accuracy_score(y, bag_predictions)
            ),
            "boosting_accuracy": float(
                accuracy_score(y, boost_predictions)
            ),
            "tree_mean_confidence": float(
                np.mean(tree_confidence)
            ),
            "bagging_mean_confidence": float(
                np.mean(bag_confidence)
            ),
            "boosting_mean_confidence": float(
                np.mean(boost_confidence)
            ),
        },
    }


def pairwise_disagreement(votes):
    # Calculate disagreement among bagged trees on one image
    B = int(np.sum(votes))

    if B < 2:
        return float("nan")

    agreeing_pairs = np.sum(votes * (votes - 1))
    total_ordered_pairs = B * (B - 1)

    return float(1.0 - agreeing_pairs / total_ordered_pairs)


def choose_example(clean, masked, y):
    # Prefer a face bagging gets right clean but wrong masked
    bag_loses = (
        (clean["bag_predictions"] == y)
        & (masked["bag_predictions"] != y)
    )

    boost_correctness_changes = (
        (clean["boost_predictions"] == y)
        != (masked["boost_predictions"] == y)
    )

    # First prefer different correctness changes between models
    different = bag_loses & (~boost_correctness_changes)

    candidates = np.flatnonzero(different)

    if len(candidates) > 0:
        return int(candidates[0])

    # Otherwise find a face with different prediction responses
    bag_changed = (
        clean["bag_predictions"]
        != masked["bag_predictions"]
    )

    boost_changed = (
        clean["boost_predictions"]
        != masked["boost_predictions"]
    )

    candidates = np.flatnonzero(
        bag_changed != boost_changed
    )

    if len(candidates) > 0:
        return int(candidates[0])

    # Fall back to the greatest difference in confidence changes
    bag_drop = (
        clean["bag_confidence"]
        - masked["bag_confidence"]
    )

    boost_drop = (
        clean["boost_confidence"]
        - masked["boost_confidence"]
    )

    return int(np.argmax(np.abs(bag_drop - boost_drop)))


def nonzero_votes(votes):
    # Retain only identities receiving at least one vote
    return {
        str(i): int(value)
        for i, value in enumerate(votes)
        if value > 0
    }


def main():
    # Load fixed splits and frozen model configurations
    data = load_data()

    with open(ROOT / "configs" / "best_tree.json") as f:
        tree_config = json.load(f)

    with open(ROOT / "configs" / "best_bagging.json") as f:
        bagging_config = json.load(f)

    with open(ROOT / "configs" / "best_adaboost.json") as f:
        boost_config = json.load(f)

    log_dir = ROOT / "results" / "logs"
    figure_dir = ROOT / "results" / "figures"

    log_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("Reconstructing frozen models...")

    tree, bagged_models, boost_run = build_models(
        data,
        tree_config,
        bagging_config,
        boost_config,
    )

    X_test = data["X_test"]
    X_masked = data["X_test_masked"]
    y_test = data["y_test"]

    # Evaluate clean and masked test images
    clean = evaluate(
        tree, bagged_models, boost_run, X_test, y_test
    )

    masked = evaluate(
        tree, bagged_models, boost_run, X_masked, y_test
    )

    comparison = pd.DataFrame([
        {"condition": "Clean", **clean["metrics"]},
        {"condition": "Masked", **masked["metrics"]},
    ])

    comparison.to_csv(
        log_dir / "q2d_comparison.csv",
        index=False,
    )

    # Save predictions and confidence for every test face
    predictions_table = pd.DataFrame({
        "test_index": np.arange(len(y_test)),
        "dataset_index": data["test_idx"],
        "true_class": y_test,
        "clean_tree": clean["tree_predictions"],
        "masked_tree": masked["tree_predictions"],
        "clean_bagging": clean["bag_predictions"],
        "masked_bagging": masked["bag_predictions"],
        "clean_boosting": clean["boost_predictions"],
        "masked_boosting": masked["boost_predictions"],
        "clean_tree_confidence": clean["tree_confidence"],
        "masked_tree_confidence": masked["tree_confidence"],
        "clean_bagging_confidence": clean["bag_confidence"],
        "masked_bagging_confidence": masked["bag_confidence"],
        "clean_boosting_confidence": clean["boost_confidence"],
        "masked_boosting_confidence": masked["boost_confidence"],
    })

    predictions_table.to_csv(
        log_dir / "q2d_predictions.csv",
        index=False,
    )

    # Identify a face with different masking responses
    i = choose_example(clean, masked, y_test)

    clean_votes = clean["bag_votes"][i]
    masked_votes = masked["bag_votes"][i]

    # Calculate tree disagreement on the selected face
    clean_disagreement = pairwise_disagreement(clean_votes)
    masked_disagreement = pairwise_disagreement(masked_votes)

    # Inspect the boosting sample weights for the true identity
    train_y = data["y_train"]
    target_class = int(y_test[i])

    class_indices = np.flatnonzero(train_y == target_class)
    class_weights = boost_run["final_weights"][class_indices]

    # Assemble detailed information for the chosen example
    example = {
        "test_index": i,
        "dataset_index": int(data["test_idx"][i]),
        "true_class": target_class,
        "clean_tree_prediction": int(
            clean["tree_predictions"][i]
        ),
        "masked_tree_prediction": int(
            masked["tree_predictions"][i]
        ),
        "clean_bagging_prediction": int(
            clean["bag_predictions"][i]
        ),
        "masked_bagging_prediction": int(
            masked["bag_predictions"][i]
        ),
        "clean_boosting_prediction": int(
            clean["boost_predictions"][i]
        ),
        "masked_boosting_prediction": int(
            masked["boost_predictions"][i]
        ),
        "clean_bagging_confidence": float(
            clean["bag_confidence"][i]
        ),
        "masked_bagging_confidence": float(
            masked["bag_confidence"][i]
        ),
        "clean_boosting_confidence": float(
            clean["boost_confidence"][i]
        ),
        "masked_boosting_confidence": float(
            masked["boost_confidence"][i]
        ),
        "clean_tree_disagreement": clean_disagreement,
        "masked_tree_disagreement": masked_disagreement,
        "clean_bagging_votes": nonzero_votes(clean_votes),
        "masked_bagging_votes": nonzero_votes(masked_votes),
        "clean_boosting_true_class_score": float(
            clean["boost_normalized_scores"][i, target_class]
        ),
        "masked_boosting_true_class_score": float(
            masked["boost_normalized_scores"][i, target_class]
        ),
        "boosting_training_weight_for_true_class": float(
            np.sum(class_weights)
        ),
        "boosting_training_weights_for_true_class": [
            float(w) for w in class_weights
        ],
    }

    with open(
        log_dir / "q2d_selected_example.json", "w"
    ) as f:
        json.dump(example, f, indent=2)

    # Display the selected clean and masked face
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for ax, X, condition in zip(
        axes,
        [X_test, X_masked],
        ["Clean", "Masked"],
    ):
        ax.imshow(
            X[i].reshape(64, 64),
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        result = clean if condition == "Clean" else masked

        ax.set_title(
            f"{condition} | True={target_class}\n"
            f"Tree={result['tree_predictions'][i]}, "
            f"Bagging={result['bag_predictions'][i]}, "
            f"SAMME={result['boost_predictions'][i]}"
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        figure_dir / "q2d_different_response_face.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nQ2(d) Clean vs Masked Comparison:")
    print(comparison.to_string(index=False))

    print("\nSelected face with different responses:")
    print(json.dumps(example, indent=2))

    print("\nQ2(d) files saved successfully.")


if __name__ == "__main__":
    main()
