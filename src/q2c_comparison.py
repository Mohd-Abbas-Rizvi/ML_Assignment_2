
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from common import load_data, ROOT
from adaboost import fit_samme, predict_samme
from bagging import majority_vote


ENSEMBLE_SIZES = [25, 50, 100]


def reconstruct_bagging(data, tree_config):
    # Load the original bootstrap indices from Q1
    saved = np.load(
        ROOT / "data" / "splits"
        / "bagging_bootstrap_indices.npz"
    )

    models = []
    seed = int(data["seed"])

    # Reconstruct the first 100 bagged trees
    for b in range(100):
        indices = saved[f"tree_{b + 1}"]

        model = DecisionTreeClassifier(
            max_depth=tree_config["max_depth"],
            min_samples_leaf=tree_config["min_samples_leaf"],
            random_state=seed + 1001 + b
        )

        model.fit(
            data["X_train"][indices],
            data["y_train"][indices]
        )

        models.append(model)

    return models


def reconstruct_weight_history(predictions, alphas, y_train):
    # Start with uniform sample weights
    n = len(y_train)
    weights = np.full(n, 1.0 / n)

    history = [weights.copy()]

    # Apply the same SAMME update after each learner
    for predictions_round, alpha in zip(predictions, alphas):
        incorrect = predictions_round != y_train

        weights = weights * np.exp(
            alpha * incorrect.astype(float)
        )

        weights /= weights.sum()
        history.append(weights.copy())

    return np.array(history)


def main():
    # Load the fixed dataset split
    data = load_data()
    seed = int(data["seed"])

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    log_dir = ROOT / "results" / "logs"
    figure_dir = ROOT / "results" / "figures"

    log_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Load the frozen model configurations
    with open(ROOT / "configs" / "best_tree.json") as f:
        tree_config = json.load(f)

    with open(ROOT / "configs" / "best_adaboost.json") as f:
        boost_config = json.load(f)

    depth = int(boost_config["max_depth"])

    # Reconstruct bagging using the original bootstrap samples
    print("Reconstructing 100 bagged trees...")
    bagged_models = reconstruct_bagging(
        data, tree_config
    )

    # Reconstruct the selected boosting depth
    print(f"Reconstructing SAMME with depth={depth}...")

    boost_run = fit_samme(
        X_train,
        y_train,
        depth=depth,
        max_rounds=100,
        seed=seed + depth * 1000
    )

    # Read the original training times for fair reporting
    bagging_logs = pd.read_csv(
        log_dir / "bagging_results.csv"
    )

    boosting_logs = pd.read_csv(
        log_dir / "adaboost_results.csv"
    )

    comparison_rows = []

    for B in ENSEMBLE_SIZES:
        # Evaluate bagging with the first B trees
        bag_models = bagged_models[:B]

        bag_train = majority_vote(np.array([
            model.predict(X_train) for model in bag_models
        ]))

        bag_val = majority_vote(np.array([
            model.predict(X_val) for model in bag_models
        ]))

        bag_test = majority_vote(np.array([
            model.predict(X_test) for model in bag_models
        ]))

        bag_time = float(
            bagging_logs.loc[
                bagging_logs["B"] == B,
                "training_time_seconds"
            ].iloc[0]
        )

        comparison_rows.append({
            "method": "Bagging",
            "learners": B,
            "train_accuracy": accuracy_score(
                y_train, bag_train
            ),
            "validation_accuracy": accuracy_score(
                y_val, bag_val
            ),
            "test_accuracy": accuracy_score(
                y_test, bag_test
            ),
            "training_time_seconds": bag_time
        })

        # Evaluate SAMME with the first B learners
        actual_B = min(B, boost_run["actual_rounds"])

        boost_models = boost_run["models"][:actual_B]
        boost_alphas = boost_run["alphas"][:actual_B]

        boost_train = predict_samme(
            boost_models, boost_alphas, X_train
        )

        boost_val = predict_samme(
            boost_models, boost_alphas, X_val
        )

        boost_test = predict_samme(
            boost_models, boost_alphas, X_test
        )

        matching = boosting_logs.loc[
            (boosting_logs["max_depth"] == depth)
            & (boosting_logs["requested_rounds"] == B)
        ]

        boost_time = float(
            matching["training_time_seconds"].iloc[0]
        )

        comparison_rows.append({
            "method": "SAMME",
            "learners": actual_B,
            "train_accuracy": accuracy_score(
                y_train, boost_train
            ),
            "validation_accuracy": accuracy_score(
                y_val, boost_val
            ),
            "test_accuracy": accuracy_score(
                y_test, boost_test
            ),
            "training_time_seconds": boost_time
        })

    # Save the comparison table
    comparison = pd.DataFrame(comparison_rows)

    comparison.to_csv(
        log_dir / "q2c_comparison.csv",
        index=False
    )

    print("\nQ2(c) Bagging vs Boosting:")
    print(comparison.to_string(index=False))

    # Use only the learners in the selected boosting model
    selected_rounds = int(
        boost_config["actual_rounds"]
    )

    prediction_history = np.array(
        boost_run["prediction_history"][:selected_rounds]
    )

    alphas = np.array(
        boost_run["alphas"][:selected_rounds]
    )

    # Reconstruct sample weights over boosting rounds
    weight_history = reconstruct_weight_history(
        prediction_history,
        alphas,
        y_train
    )

    # Select the highest-weight image at the final round
    final_weights = weight_history[-1]
    hardest_index = int(np.argmax(final_weights))

    true_class = int(y_train[hardest_index])

    # Inspect how each weak learner classified this image
    face_predictions = prediction_history[
        :, hardest_index
    ]

    mistakes = face_predictions != true_class

    face_history = pd.DataFrame({
        "round": np.arange(1, selected_rounds + 1),
        "true_class": true_class,
        "prediction": face_predictions,
        "incorrect": mistakes.astype(int),
        "alpha": alphas,
        "weight_before": weight_history[:-1, hardest_index],
        "weight_after": weight_history[1:, hardest_index]
    })

    face_history.to_csv(
        log_dir / "q2c_high_weight_face_history.csv",
        index=False
    )

    # Plot the selected training face
    image = X_train[hardest_index].reshape(64, 64)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)

    ax.set_title(
        f"Train index {hardest_index}, "
        f"Identity {true_class}"
    )

    ax.axis("off")
    fig.tight_layout()

    fig.savefig(
        figure_dir / "q2c_high_weight_face.png",
        dpi=300
    )
    plt.close(fig)

    # Plot its sample weight across boosting rounds
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(
        np.arange(selected_rounds + 1),
        weight_history[:, hardest_index]
    )

    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Sample weight")
    ax.set_title("Weight History of a Difficult Training Face")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        figure_dir / "q2c_high_weight_face_history.png",
        dpi=300
    )
    plt.close(fig)

    print("\nHigh-weight training face:")
    print("Training index:", hardest_index)
    print("Original dataset index:", int(data["train_idx"][hardest_index]))
    print("True identity:", true_class)
    print("Final sample weight:", float(final_weights[hardest_index]))
    print("Incorrect predictions:", int(mistakes.sum()))
    print("Total boosting rounds:", selected_rounds)

    print("\nFirst 10 learner predictions:")
    print(face_history.head(10).to_string(index=False))

    print("\nLast 10 learner predictions:")
    print(face_history.tail(10).to_string(index=False))

    print("\nQ2(c) files saved successfully.")


if __name__ == "__main__":
    main()
