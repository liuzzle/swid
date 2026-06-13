#!/usr/bin/env python3
"""
Step 12 plotting — accuracy/F1 vs duration, plus per-speaker variability.

Reads the CSVs written by ``run_duration_ablation.py``
(``global_metrics.csv`` and ``per_class_metrics.csv``) and renders:

  accuracy_vs_duration.png    line plot of top-1 accuracy vs clip duration,
                              one panel per embedding type, one line per classifier
  macro_f1_vs_duration.png    same layout for macro-averaged F1
  per_speaker_f1_box.png      per-speaker F1 spread vs duration (best classifier
                              per embedding type) — shows class-level variability

Run standalone after a sweep, or let the runner invoke it:
  python scripts/plot_duration_ablation.py --results-dir voxceleb_data/processed_test/results/ablation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import pandas as pd

EMB_ORDER = ["xvector", "whisper_last", "whisper_mean"]
CLF_ORDER = ["cosine", "knn", "svm"]
CLF_COLORS = {"cosine": "#1f77b4", "knn": "#ff7f0e", "svm": "#2ca02c"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot step-12 duration ablation results")
    p.add_argument("--results-dir", type=Path,
                   default=Path("voxceleb_data/processed_test/results/ablation"))
    return p.parse_args()


def _line_plot(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    """One panel per embedding type; x=duration (log), y=metric, line per classifier."""
    embs = [e for e in EMB_ORDER if e in df["emb_type"].unique()]
    fig, axes = plt.subplots(1, len(embs), figsize=(5 * len(embs), 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    durations = sorted(df["duration_sec"].unique())

    for ax, emb in zip(axes, embs):
        sub = df[df["emb_type"] == emb]
        for clf in CLF_ORDER:
            s = sub[sub["classifier"] == clf].sort_values("duration_sec")
            if s.empty:
                continue
            ax.plot(s["duration_sec"], s[metric] * 100, marker="o",
                    label=clf, color=CLF_COLORS.get(clf))
        ax.set_xscale("log")
        ax.set_xticks(durations)
        ax.set_xticklabels([f"{d:g}s" for d in durations])
        ax.set_xlabel("Clip duration")
        ax.set_title(emb)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel(f"{title} (%)")
    axes[-1].legend(title="classifier", loc="lower right")
    fig.suptitle(f"{title} vs clip duration", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _per_speaker_box(global_df: pd.DataFrame, pc_df: pd.DataFrame, out_path: Path) -> None:
    """Per-speaker F1 distribution vs duration for each embedding's best classifier."""
    embs = [e for e in EMB_ORDER if e in pc_df["emb_type"].unique()]
    fig, axes = plt.subplots(1, len(embs), figsize=(5 * len(embs), 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    durations = sorted(pc_df["duration_sec"].unique())

    for ax, emb in zip(axes, embs):
        # pick the classifier with the highest mean accuracy for this embedding
        best = (global_df[global_df["emb_type"] == emb]
                .groupby("classifier")["accuracy"].mean().idxmax())
        sub = pc_df[(pc_df["emb_type"] == emb) & (pc_df["classifier"] == best)]
        data = [sub[sub["duration_sec"] == d]["f1"].values * 100 for d in durations]
        ax.boxplot(data, tick_labels=[f"{d:g}s" for d in durations], showmeans=True)
        ax.set_xlabel("Clip duration")
        ax.set_title(f"{emb}  ({best})")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Per-speaker F1 (%)")
    fig.suptitle("Per-speaker F1 spread vs clip duration (best classifier per embedding)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    args = parse_args()
    rd = args.results_dir
    global_df = pd.read_csv(rd / "global_metrics.csv")
    pc_df = pd.read_csv(rd / "per_class_metrics.csv")

    print(f"Plotting from {rd}/ ...")
    _line_plot(global_df, "accuracy", "Top-1 accuracy", rd / "accuracy_vs_duration.png")
    _line_plot(global_df, "macro_f1", "Macro-F1", rd / "macro_f1_vs_duration.png")
    _per_speaker_box(global_df, pc_df, rd / "per_speaker_f1_box.png")


if __name__ == "__main__":
    main()
