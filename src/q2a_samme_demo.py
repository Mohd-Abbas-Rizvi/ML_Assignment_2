
import json
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from common import ROOT


def samme_weight_update(y_true, y_pred, weights, n_classes):
    # Identify incorrectly classified samples
    incorrect = y_pred != y_true

    # Calculate the weighted classification error
    error = float(np.sum(weights[incorrect]))

    # Reject learners that are no better than random guessing
    random_error = 1.0 - 1.0 / n_classes

    if error >= random_error:
        raise ValueError(
            "Weak learner is not better than random multiclass prediction."
        )

    # A perfect learner should terminate boosting
    if error <= 1e-12:
        return error, float("inf"), weights.copy(), True

    # Calculate the SAMME learner weight
    alpha = (
        np.log((1.0 - error) / error)
        + np.log(n_classes - 1)
    )

    # Increase weights of incorrectly classified samples
    updated_weights = weights * np.exp(
        alpha * incorrect.astype(float)
    )

    # Normalize weights to sum to one
    updated_weights /= np.sum(updated_weights)

    return error, float(alpha), updated_weights, False


def main():
    # Create a small dataset with 9 samples and 3 classes
    X = np.arange(9, dtype=float).reshape(-1, 1)

    y = np.array([
        0, 0, 0,
        1, 1, 1,
        2, 2, 2
    ])

    n_samples = len(y)
    n_classes = len(np.unique(y))

    # Initialize equal sample weights
    initial_weights = np.full(
        n_samples,
        1.0 / n_samples
    )

    # Train one depth-1 weak learner
    model = DecisionTreeClassifier(
        max_depth=1,
        random_state=26014
    )

    model.fit(
        X,
        y,
        sample_weight=initial_weights
    )

    # Predict the training samples
    predictions = model.predict(X)

    # Perform one complete SAMME update
    error, alpha, updated_weights, perfect = samme_weight_update(
        y,
        predictions,
        initial_weights,
        n_classes
    )

    # Calculate whether each prediction was incorrect
    incorrect = (predictions != y).astype(int)

    # Create a table containing the complete SAMME round
    results = pd.DataFrame({
        "sample": np.arange(1, n_samples + 1),
        "feature": X.ravel(),
        "true_class": y,
        "prediction": predictions,
        "incorrect": incorrect,
        "initial_weight": initial_weights,
        "updated_weight": updated_weights
    })

    # Save the demonstration results
    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        log_dir / "q2a_samme_demo.csv",
        index=False
    )

    summary = {
        "n_samples": int(n_samples),
        "n_classes": int(n_classes),
        "weak_learner_depth": 1,
        "weighted_error": float(error),
        "alpha": None if perfect else float(alpha),
        "perfect_learner": bool(perfect),
        "initial_weight_sum": float(
            initial_weights.sum()
        ),
        "updated_weight_sum": float(
            updated_weights.sum()
        ),
        "number_of_mistakes": int(
            incorrect.sum()
        )
    }

    with open(log_dir / "q2a_samme_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Display the complete SAMME round
    print("\nQ2(a): Three-class SAMME demonstration")
    print(results.to_string(index=False))

    print("\nSAMME round summary:")
    print(json.dumps(summary, indent=2))

    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()
