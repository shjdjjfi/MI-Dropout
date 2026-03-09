from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_plots(eval_csv: str, out_dir: str) -> None:
    df = pd.read_csv(eval_csv)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="compression_ratio", y="f1", hue="baseline", alpha=0.7)
    plt.title("Faithfulness/Accuracy vs Compression")
    plt.tight_layout()
    plt.savefig(out / "accuracy_vs_compression.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="latency_ms", y="f1", hue="baseline", alpha=0.7)
    plt.title("Latency vs Accuracy")
    plt.tight_layout()
    plt.savefig(out / "latency_vs_accuracy.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="compressed_nodes", bins=20)
    plt.title("Compressed Graph Size Histogram")
    plt.tight_layout()
    plt.savefig(out / "graph_size_hist.png", dpi=200)
    plt.close()
