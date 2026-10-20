
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import ROOT


def main():
    # Load previously saved bagging results
    csv_path = ROOT / "results" / "logs" / "bagging_results.csv"
    df = pd.read_csv(csv_path)

    # Create output folder
    output_dir = ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    B = df["B"].to_numpy()

    # Plot OOB and validation accuracy
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        B, df["oob_accuracy"],
        marker="o", label="OOB accuracy"
    )
    ax.plot(
        B, df["validation_accuracy"],
        marker="s", label="Validation accuracy"
    )
    ax.set_xscale("log")
    ax.set_xticks(B)
    ax.set_xticklabels(B)
    ax.set_xlabel("Number of trees (B)")
    ax.set_ylabel("Accuracy")
    ax.set_title("OOB and Validation Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "q1c_oob_accuracy.png", dpi=300
    )
    plt.close(fig)

    # Plot pairwise disagreement
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        B, df["pairwise_disagreement"],
        marker="o"
    )
    ax.set_xscale("log")
    ax.set_xticks(B)
    ax.set_xticklabels(B)
    ax.set_xlabel("Number of trees (B)")
    ax.set_ylabel("Mean pairwise disagreement")
    ax.set_title("Pairwise Tree Disagreement")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        output_dir / "q1c_disagreement.png", dpi=300
    )
    plt.close(fig)

    # Plot average vote margin
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        B, df["mean_vote_margin"],
        marker="o"
    )
    ax.set_xscale("log")
    ax.set_xticks(B)
    ax.set_xticklabels(B)
    ax.set_xlabel("Number of trees (B)")
    ax.set_ylabel("Mean vote margin")
    ax.set_title("Ensemble Vote Margin")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        output_dir / "q1c_vote_margin.png", dpi=300
    )
    plt.close(fig)

    print("Q1(c) plots saved to:", output_dir)

    for filename in [
        "q1c_oob_accuracy.png",
        "q1c_disagreement.png",
        "q1c_vote_margin.png",
    ]:
        print(filename)


if __name__ == "__main__":
    main()
