
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from common import load_data, ROOT
from q2a_samme_demo import samme_weight_update


TREE_DEPTHS = [1, 2, 3, 4]
BOOSTING_ROUNDS = [10, 25, 50, 100]
N_CLASSES = 40


def get_weighted_scores(models, alphas, X):
    # Calculate weighted class votes from all learners
    scores = np.zeros((len(X), N_CLASSES), dtype=float)

    for model, alpha in zip(models, alphas):
        predictions = model.predict(X)
        scores[np.arange(len(X)), predictions] += alpha

    return scores


def predict_samme(models, alphas, X):
    # Predict the identity with the highest weighted score
    if len(models) == 0:
        raise ValueError("No valid SAMME learners were trained.")

    scores = get_weighted_scores(models, alphas, X)
    return np.argmax(scores, axis=1)


def fit_samme(X_train, y_train, depth, max_rounds, seed):
    # Initialize uniform sample weights
    n_samples = len(y_train)
    weights = np.full(n_samples, 1.0 / n_samples)

    models = []
    alphas = []
    diagnostics = []
    prediction_history = []

    start_time = time.perf_counter()
    stop_reason = "maximum_rounds_reached"

    for round_index in range(max_rounds):
        # Train a tree using the current sample weights
        model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=1,
            random_state=seed + round_index,
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=weights
        )

        predictions = model.predict(X_train)
        incorrect = predictions != y_train

        # Calculate weighted training error
        weighted_error = float(
            np.sum(weights[incorrect])
        )

        random_error = 1.0 - 1.0 / N_CLASSES

        # Explicitly reject learners no better than random prediction
        if weighted_error >= random_error:
            stop_reason = "weak_learner_not_better_than_random"
            print(
                f"  Stopped at round {round_index + 1}: "
                f"weighted error={weighted_error:.6f}"
            )
            break

        # Use the exact SAMME update from Q2(a)
        error, alpha, new_weights, perfect = samme_weight_update(
            y_train,
            predictions,
            weights,
            N_CLASSES
        )

        # A perfect learner makes further boosting unnecessary
        if perfect:
            alpha = 1.0
            stop_reason = "perfect_weak_learner"

        models.append(model)
        alphas.append(float(alpha))
        prediction_history.append(predictions.copy())

        weights = new_weights

        # Calculate the largest sample weights after the update
        sorted_weights = np.sort(weights)[::-1]
        top10_total = float(
            np.sum(sorted_weights[:10])
        )

        diagnostics.append({
            "round": round_index + 1,
            "weighted_training_error": float(error),
            "alpha": float(alpha),
            "max_sample_weight": float(np.max(weights)),
            "top10_total_weight": top10_total,
            "weight_sum": float(np.sum(weights)),
            "elapsed_seconds": float(
                time.perf_counter() - start_time
            ),
        })

        if perfect:
            break

    return {
        "models": models,
        "alphas": alphas,
        "diagnostics": diagnostics,
        "prediction_history": prediction_history,
        "final_weights": weights.copy(),
        "actual_rounds": len(models),
        "stop_reason": stop_reason,
    }


def save_diagnostic_plots(diagnostics, output_dir):
    # Create four plots for the selected boosting model
    df = pd.DataFrame(diagnostics)

    plots = [
        (
            "weighted_training_error",
            "Weighted Training Error",
            "q2b_weighted_error.png"
        ),
        (
            "alpha",
            "SAMME Learner Weight",
            "q2b_learner_weight.png"
        ),
        (
            "max_sample_weight",
            "Maximum Sample Weight",
            "q2b_max_sample_weight.png"
        ),
        (
            "top10_total_weight",
            "Total Weight of 10 Heaviest Images",
            "q2b_top10_weight.png"
        ),
    ]

    for column, title, filename in plots:
        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.plot(
            df["round"],
            df[column],
            linewidth=1.5
        )

        ax.set_xlabel("Boosting Round")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            output_dir / filename,
            dpi=300
        )
        plt.close(fig)


def main():
    # Load the fixed Olivetti Faces dataset split
    data = load_data()

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    seed = int(data["seed"])

    log_dir = ROOT / "results" / "logs"
    figure_dir = ROOT / "results" / "figures"
    config_dir = ROOT / "configs"

    log_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    results = []
    trained_models = {}

    # Train each depth up to 100 rounds
    for depth in TREE_DEPTHS:
        print(f"\nTraining SAMME with depth={depth}")

        result = fit_samme(
            X_train,
            y_train,
            depth=depth,
            max_rounds=max(BOOSTING_ROUNDS),
            seed=seed + depth * 1000
        )

        trained_models[depth] = result

        if result["actual_rounds"] == 0:
            print(
                f"Depth {depth} produced no valid weak learners."
            )
            continue

        # Evaluate prefixes of 10, 25, 50 and 100 learners
        for requested_rounds in BOOSTING_ROUNDS:
            actual_rounds = min(
                requested_rounds,
                result["actual_rounds"]
            )

            models = result["models"][:actual_rounds]
            alphas = result["alphas"][:actual_rounds]

            train_predictions = predict_samme(
                models, alphas, X_train
            )

            val_predictions = predict_samme(
                models, alphas, X_val
            )

            train_accuracy = float(
                accuracy_score(y_train, train_predictions)
            )

            val_accuracy = float(
                accuracy_score(y_val, val_predictions)
            )

            training_time = float(
                result["diagnostics"][actual_rounds - 1][
                    "elapsed_seconds"
                ]
            )

            results.append({
                "max_depth": depth,
                "requested_rounds": requested_rounds,
                "actual_rounds": actual_rounds,
                "train_accuracy": train_accuracy,
                "validation_accuracy": val_accuracy,
                "training_time_seconds": training_time,
                "stop_reason": result["stop_reason"]
                if actual_rounds < requested_rounds
                else "requested_rounds_reached",
            })

            print(
                f"Depth={depth}, "
                f"Requested={requested_rounds}, "
                f"Actual={actual_rounds}, "
                f"Train={train_accuracy:.4f}, "
                f"Val={val_accuracy:.4f}"
            )

    if len(results) == 0:
        raise RuntimeError(
            "No valid SAMME model could be trained."
        )

    # Select model using clean validation accuracy only
    results.sort(
        key=lambda row: (
            -row["validation_accuracy"],
            row["actual_rounds"],
            row["max_depth"],
            row["requested_rounds"]
        )
    )

    best_row = results[0]

    best_depth = int(best_row["max_depth"])
    best_requested_rounds = int(
        best_row["requested_rounds"]
    )
    best_actual_rounds = int(best_row["actual_rounds"])

    best_run = trained_models[best_depth]

    selected_models = best_run["models"][:best_actual_rounds]
    selected_alphas = best_run["alphas"][:best_actual_rounds]

    # Evaluate test data after model selection
    test_predictions = predict_samme(
        selected_models,
        selected_alphas,
        X_test
    )

    test_accuracy = float(
        accuracy_score(y_test, test_predictions)
    )

    # Save all 16 requested configurations
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        log_dir / "adaboost_results.csv",
        index=False
    )

    # Save diagnostics for the selected boosting model
    selected_diagnostics = best_run["diagnostics"][
        :best_actual_rounds
    ]

    pd.DataFrame(selected_diagnostics).to_csv(
        log_dir / "adaboost_selected_round_diagnostics.csv",
        index=False
    )

    # Save the four required diagnostic plots
    save_diagnostic_plots(
        selected_diagnostics,
        figure_dir
    )

    # Record the selected hyperparameters and evaluation metrics
    best_config = {
        "algorithm": "SAMME",
        "max_depth": best_depth,
        "requested_rounds": best_requested_rounds,
        "actual_rounds": best_actual_rounds,
        "train_accuracy": float(
            best_row["train_accuracy"]
        ),
        "validation_accuracy": float(
            best_row["validation_accuracy"]
        ),
        "clean_test_accuracy": test_accuracy,
        "training_time_seconds": float(
            best_row["training_time_seconds"]
        ),
        "stop_reason": best_row["stop_reason"],
        "seed": seed,
    }

    with open(
        config_dir / "best_adaboost.json", "w"
    ) as f:
        json.dump(best_config, f, indent=2)

    # Save predictions and weights for Q2(c)
    np.savez(
        log_dir / "adaboost_selected_diagnostics.npz",
        final_weights=best_run["final_weights"],
        prediction_history=np.array(
            best_run["prediction_history"][:best_actual_rounds]
        ),
        alphas=np.array(selected_alphas),
        y_train=y_train,
    )

    print("\nAll SAMME configurations:")
    print(results_df.to_string(index=False))

    print("\nSelected boosting model:")
    print(json.dumps(best_config, indent=2))

    print("\nSaved diagnostic plots:")
    for name in [
        "q2b_weighted_error.png",
        "q2b_learner_weight.png",
        "q2b_max_sample_weight.png",
        "q2b_top10_weight.png",
    ]:
        print(name)


if __name__ == "__main__":
    main()
