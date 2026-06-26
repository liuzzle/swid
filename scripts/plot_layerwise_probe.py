#!/usr/bin/env python3
"""
Step 13 plotting — speaker-ID performance vs Whisper layer.

Reads the CSV written by ``run_layerwise_probe.py`` (``layer_metrics.csv``) and
renders:

  accuracy_vs_layer.png    top-1 accuracy vs layer index, one panel per duration,
                           one line per classifier; the pooled-mean baseline is
                           drawn as a dashed horizontal reference per classifier
  macro_f1_vs_layer.png    same layout for macro-averaged F1
  layer_accuracy_heatmap.png  heatmap (rows=layer, cols=duration) of accuracy for
                           the best overall classifier — the headline "where is
                           speaker signal strongest" figure

Run standalone after a sweep, or let the runner invoke it:
  python scripts/plot_layerwise_probe.py --results-dir voxceleb_data/processed_test/results/layerwise
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLF_ORDER = ["cosine", "knn", "svm"]
CLF_COLORS = {"cosine": "#1f77b4", "knn": "#ff7f0e", "svm": "#2ca02c"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot step-13 layerwise probe results")
    p.add_argument("--results-dir", type=Path,
                   default=Path("voxceleb_data/processed_test/results/layerwise"))
    return p.parse_args()


def _layer_plot(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    """One panel per duration; x=layer index, y=metric, line per classifier.

    'mean' rows are not plotted as a point on the layer axis; instead they are
    drawn as a dashed horizontal reference line per classifier.
    """
    durations = sorted(df["duration_sec"].unique())
    single = df[df["layer"] != "mean"].copy()
    single["layer_index"] = single["layer_index"].astype(int)
    layers = sorted(single["layer_index"].unique())

    fig, axes = plt.subplots(1, len(durations), figsize=(4.2 * len(durations), 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, dsec in zip(axes, durations):
        sub = single[single["duration_sec"] == dsec]
        meansub = df[(df["layer"] == "mean") & (df["duration_sec"] == dsec)]
        for clf in CLF_ORDER:
            s = sub[sub["classifier"] == clf].sort_values("layer_index")
            if not s.empty:
                ax.plot(s["layer_index"], s[metric] * 100, marker="o",
                        label=clf, color=CLF_COLORS.get(clf))
            m = meansub[meansub["classifier"] == clf]
            if not m.empty:
                ax.axhline(float(m[metric].iloc[0]) * 100, ls="--", lw=1,
                           color=CLF_COLORS.get(clf), alpha=0.6)
        ax.set_xticks(layers)
        ax.set_xlabel("Whisper layer index")
        ax.set_title(f"{dsec:g}s")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel(f"{title} (%)")
    axes[-1].legend(title="classifier (— = mean)", loc="upper left", fontsize=8)
    fig.suptitle(f"{title} vs Whisper layer", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Accuracy heatmap (rows=layer, cols=duration) for the best overall classifier."""
    single = df[df["layer"] != "mean"].copy()
    single["layer_index"] = single["layer_index"].astype(int)
    # best classifier = highest mean accuracy across all single-layer rows
    best = single.groupby("classifier")["accuracy"].mean().idxmax()
    sub = single[single["classifier"] == best]

    durations = sorted(sub["duration_sec"].unique())
    layers = sorted(sub["layer_index"].unique())
    mat = np.full((len(layers), len(durations)), np.nan)
    for i, L in enumerate(layers):
        for j, d in enumerate(durations):
            cell = sub[(sub["layer_index"] == L) & (sub["duration_sec"] == d)]
            if not cell.empty:
                mat[i, j] = cell["accuracy"].iloc[0] * 100

    fig, ax = plt.subplots(figsize=(1.4 * len(durations) + 2, 0.7 * len(layers) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="lower",
                   vmin=0, vmax=100)
    ax.set_xticks(range(len(durations)))
    ax.set_xticklabels([f"{d:g}s" for d in durations])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(L) for L in layers])
    ax.set_xlabel("Clip duration")
    ax.set_ylabel("Whisper layer index")
    ax.set_title(f"Accuracy (%) by layer × duration — {best}")
    for i in range(len(layers)):
        for j in range(len(durations)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                        color="white" if mat[i, j] < 60 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    args = parse_args()
    rd = args.results_dir
    df = pd.read_csv(rd / "layer_metrics.csv")

    print(f"Plotting from {rd}/ ...")
    _layer_plot(df, "accuracy", "Top-1 accuracy", rd / "accuracy_vs_layer.png")
    _layer_plot(df, "macro_f1", "Macro-F1", rd / "macro_f1_vs_layer.png")
    _heatmap(df, rd / "layer_accuracy_heatmap.png")


if __name__ == "__main__":
    main()
