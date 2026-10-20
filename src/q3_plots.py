
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import ROOT


def save_figure(fig, output_dir, name):
    # Save figures in PNG and PDF formats
    fig.savefig(
        output_dir / f"{name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        output_dir / f"{name}.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_bias_variance(log_dir, output_dir):
    # Read results while preserving the None depth label
    df = pd.read_csv(
        log_dir / "q3b_bias_variance_by_depth.csv",
        keep_default_na=False
    )

    df["depth"] = df["depth"].astype(str)
    df["depth"] = df["depth"].replace({
        "4": "4",
        "16": "16",
        "None": "Unrestricted"
    })

    depth_order = ["4", "16", "Unrestricted"]

    tree = df[df["method"] == "Single Tree"]
    bagging = df[df["method"] == "Bagging"]

    tree = tree.set_index("depth").loc[depth_order]
    bagging = bagging.set_index("depth").loc[depth_order]

    x = np.arange(len(depth_order))

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 4)
    )

    # Plot squared bias
    axes[0].plot(
        x,
        tree["bias_squared"],
        marker="o",
        label="Single Tree"
    )

    axes[0].plot(
        x,
        bagging["bias_squared"],
        marker="s",
        label="Bagging"
    )

    axes[0].set_title("Squared Bias")
    axes[0].set_ylabel("Bias squared")

    # Plot variance
    axes[1].plot(
        x,
        tree["variance"],
        marker="o",
        label="Single Tree"
    )

    axes[1].plot(
        x,
        bagging["variance"],
        marker="s",
        label="Bagging"
    )

    axes[1].set_title("Prediction Variance")
    axes[1].set_ylabel("Variance")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(depth_order)
        ax.set_xlabel("Maximum Tree Depth")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "q3_bias_variance_vs_depth"
    )


def plot_bagging_improvement(log_dir, output_dir):
    # Load improvement measurements for each depth
    df = pd.read_csv(
        log_dir / "q3b_depth_comparison.csv",
        keep_default_na=False
    )

    df["depth"] = df["depth"].astype(str)

    order = ["4", "16", "None"]

    df = df.set_index("depth").loc[order]

    labels = ["4", "16", "Unrestricted"]

    x = np.arange(len(labels))

    # Convert accuracy improvement to percentage points
    gains = df["accuracy_gain"].to_numpy() * 100

    fig, ax = plt.subplots(figsize=(6.5, 4))

    bars = ax.bar(x, gains)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_xlabel("Maximum Tree Depth")
    ax.set_ylabel("Accuracy Gain (percentage points)")
    ax.set_title("Test Accuracy Improvement from Bagging")

    ax.bar_label(
        bars,
        fmt="%.2f",
        padding=3
    )

    ax.set_ylim(0, max(gains) * 1.2)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "q3_bagging_accuracy_improvement"
    )


def plot_clean_masked_comparison(log_dir, output_dir):
    # Load frozen model accuracy results from Q2(d)
    df = pd.read_csv(
        log_dir / "q2d_comparison.csv"
    )

    clean = df.loc[
        df["condition"] == "Clean"
    ].iloc[0]

    masked = df.loc[
        df["condition"] == "Masked"
    ].iloc[0]

    models = [
        "Single Tree",
        "Bagging",
        "SAMME"
    ]

    columns = [
        "tree_accuracy",
        "bagging_accuracy",
        "boosting_accuracy"
    ]

    clean_acc = np.array([
        clean[column] for column in columns
    ]) * 100

    masked_acc = np.array([
        masked[column] for column in columns
    ]) * 100

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))

    clean_bars = ax.bar(
        x - width / 2,
        clean_acc,
        width,
        label="Clean"
    )

    masked_bars = ax.bar(
        x + width / 2,
        masked_acc,
        width,
        label="Masked"
    )

    ax.bar_label(
        clean_bars,
        fmt="%.2f",
        padding=3,
        fontsize=8
    )

    ax.bar_label(
        masked_bars,
        fmt="%.2f",
        padding=3,
        fontsize=8
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models)

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Clean vs Masked Test Accuracy")
    ax.set_ylim(0, 105)

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "q3_clean_masked_accuracy"
    )


def main():
    # Locate existing experiment results
    log_dir = ROOT / "results" / "logs"
    output_dir = ROOT / "results" / "figures"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Generate the final Q3 figures
    plot_bias_variance(log_dir, output_dir)
    plot_bagging_improvement(log_dir, output_dir)
    plot_clean_masked_comparison(log_dir, output_dir)

    print("\nQ3 figures generated successfully:")

    for name in [
        "q3_bias_variance_vs_depth",
        "q3_bagging_accuracy_improvement",
        "q3_clean_masked_accuracy"
    ]:
        print(f"{name}.png")
        print(f"{name}.pdf")


if __name__ == "__main__":
    main()
