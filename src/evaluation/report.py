import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_zero_shot_comparison(csv_path="outputs/reports/zero_shot_results.csv",
                               save_path="outputs/figures/zero_shot_comparison.png"):
    """Bar chart comparing prompt template performance."""
    df = pd.read_csv(csv_path)
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, df["top1"], width, label="Top-1", color="steelblue")
    bars2 = ax.bar(x + width / 2, df["top5"], width, label="Top-5", color="coral")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Zero-shot CLIP Performance by Prompt Template")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(len(df))])
    ax.legend()
    ax.set_ylim(0, 100)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {save_path}")


def plot_comparison_table(zero_shot_csv="outputs/reports/zero_shot_results.csv",
                           lp_report_path="outputs/reports/linear_probe_results.csv",
                           save_path="outputs/figures/comparison_table.png"):
    """Table comparing zero-shot vs Linear Probe performance."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    data = [
        ["Method", "Seen Top-1", "Seen Top-5", "Unseen Top-1", "Unseen Top-5"],
    ]

    # Zero-shot best
    zs_df = pd.read_csv(zero_shot_csv)
    best_zs = zs_df.loc[zs_df["top1"].idxmax()]
    data.append([
        "Zero-shot (best template)",
        f"{best_zs['top1']:.2f}%", f"{best_zs['top5']:.2f}%",
        "N/A", "N/A"
    ])

    # Linear probe
    try:
        lp_df = pd.read_csv(lp_report_path)
        data.append([
            "Linear Probe",
            f"{lp_df['seen_top1'].iloc[0]:.2f}%", f"{lp_df['seen_top5'].iloc[0]:.2f}%",
            "N/A", "N/A"
        ])
    except (FileNotFoundError, KeyError):
        data.append(["Linear Probe", "N/A", "N/A", "N/A", "N/A"])

    table = ax.table(cellText=data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison table saved to {save_path}")


def plot_retrieval_results(query_paths, result_paths, save_path="outputs/figures/retrieval_examples.png"):
    """Side-by-side visualization of retrieval queries and their top results."""
    n = len(query_paths)
    fig, axes = plt.subplots(n, 6, figsize=(15, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        # Query image
        from PIL import Image
        query_img = Image.open(query_paths[i])
        axes[i, 0].imshow(query_img)
        axes[i, 0].set_title("Query")
        axes[i, 0].axis("off")

        # Top-5 results
        for j in range(5):
            idx = i * 5 + j
            if idx < len(result_paths):
                res_img = Image.open(result_paths[idx])
                axes[i, j + 1].imshow(res_img)
                axes[i, j + 1].set_title(f"Result {j + 1}")
                axes[i, j + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Retrieval examples saved to {save_path}")
