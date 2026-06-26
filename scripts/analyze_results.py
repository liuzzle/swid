#!/usr/bin/env python3
"""
Step 14 — Analyze results and create paper figures.

Consolidates the step-11/12/13 experiment outputs into publication-ready figures
and tables.  Three deliverables, matching ``detailed_todo.md`` step 14
("confusion matrices, accuracy-vs-duration plots, and per-speaker analysis"):

  1. CONFUSION MATRICES (new)
     For the best classifier of each embedding type (auto-selected by mean
     accuracy from the ablation), recompute predictions and render a row of
     recall-normalised 40×40 confusion matrices, one panel per duration.
     Raw matrices are also saved as .npy + a tidy CSV for the supplement.

  2. HEAD-TO-HEAD ACCURACY vs DURATION (consolidated)
     A single clean figure comparing x-vector vs whisper_mean (best classifier
     each) vs whisper best-single-layer, for the paper's main results figure.

  3. PER-SPEAKER ANALYSIS (consolidated)
     Per-speaker recall heatmap (speakers × duration) for each embedding's best
     classifier, plus a ``hard_speakers.csv`` ranking speakers by mean recall.

Confusion-matrix predictions are produced by re-running the exact same classifier
functions used in steps 11–13 (imported from ``classify_speakers``) on the same
manifests, so they are consistent with the reported accuracies.  The other two
deliverables read the already-written ablation CSVs.

Outputs under <output-root>/results/analysis/ :
  confusion_<emb>.png            1×4 panel of normalised confusion matrices
  confusion_matrices.csv         tidy long form (emb, clf, duration, true, pred, count, recall)
  cm_<emb>_<dur>.npy             raw count matrices (supplement)
  headtohead_accuracy.png        consolidated accuracy-vs-duration figure
  per_speaker_recall_<emb>.png   per-speaker recall heatmap
  hard_speakers.csv              speakers ranked by mean recall (per emb)

Example:
  python scripts/analyze_results.py --output-root voxceleb_data/processed_test
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from classify_speakers import (  # noqa: E402
    load_embeddings, cosine_centroid, knn, svm,
)

DURATIONS = ["0.5s", "1s", "3s", "5s"]
DURATION_SECONDS = {"0.5s": 0.5, "1s": 1.0, "3s": 3.0, "5s": 5.0}

# Embedding configs to analyse (label -> how to load it)
EMB_SPECS = [
    {"label": "xvector",      "emb_type": "xvector", "whisper_layer": "-1"},
    {"label": "whisper_mean", "emb_type": "whisper", "whisper_layer": "mean"},
]

CLF_FUNCS = {"cosine": cosine_centroid, "knn": knn, "svm": svm}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step-14 results analysis / paper figures")
    p.add_argument("--output-root", type=Path, default=Path("voxceleb_data/processed_test"))
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Output dir (default: <output-root>/results/analysis)")
    p.add_argument("--ablation-dir", type=Path, default=None,
                   help="Step-12 ablation results dir (default: <output-root>/results/ablation)")
    return p.parse_args()


def best_classifier_per_emb(global_df: pd.DataFrame) -> dict:
    """For each emb_type label, the classifier with the highest mean accuracy."""
    best = {}
    for emb in global_df["emb_type"].unique():
        sub = global_df[global_df["emb_type"] == emb]
        best[emb] = sub.groupby("classifier")["accuracy"].mean().idxmax()
    return best


# ---------------------------------------------------------------------------
# 1. Confusion matrices
# ---------------------------------------------------------------------------

def _predict(train_csv, test_csv, emb_type, whisper_layer, clf_name):
    X_tr, y_tr = load_embeddings(train_csv, emb_type, whisper_layer)
    X_te, y_te = load_embeddings(test_csv, emb_type, whisper_layer)
    res = CLF_FUNCS[clf_name](X_tr, y_tr, X_te, y_te)
    return y_te, res["predictions"], res["accuracy"]


def confusion_matrices(metadata_root: Path, best: dict, results_dir: Path):
    """Build + plot recall-normalised confusion matrices per emb × duration."""
    long_rows = []
    for spec in EMB_SPECS:
        label = spec["label"]
        clf = best.get(label, "knn")
        fig, axes = plt.subplots(1, len(DURATIONS), figsize=(4.0 * len(DURATIONS), 4.2),
                                 squeeze=False)
        axes = axes[0]
        im = None
        for ax, dur in zip(axes, DURATIONS):
            train_csv = metadata_root / f"dur_{dur}" / f"{spec['emb_type']}_train.csv"
            test_csv = metadata_root / f"dur_{dur}" / f"{spec['emb_type']}_test.csv"
            y_true, y_pred, acc = _predict(
                train_csv, test_csv, spec["emb_type"], spec["whisper_layer"], clf)

            labels = sorted(np.unique(y_true))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            np.save(results_dir / f"cm_{label}_{dur}.npy", cm)

            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = cm / np.maximum(row_sums, 1)

            for i, t in enumerate(labels):
                for j, pcol in enumerate(labels):
                    if cm[i, j]:
                        long_rows.append({
                            "emb_type": label, "classifier": clf, "duration": dur,
                            "true": t, "pred": pcol,
                            "count": int(cm[i, j]),
                            "recall": round(float(cm_norm[i, j]), 6),
                        })

            im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1, aspect="auto")
            ax.set_title(f"{dur}  (acc {acc*100:.1f}%)", fontsize=10)
            ax.set_xlabel("Predicted speaker")
            ax.set_xticks([]); ax.set_yticks([])
        axes[0].set_ylabel("True speaker")
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01, label="Recall (row-normalised)")
        fig.suptitle(f"Confusion matrices — {label} / {clf}  (40 speakers)", fontsize=13)
        out = results_dir / f"confusion_{label}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")

    _write_csv(results_dir / "confusion_matrices.csv", long_rows,
               ["emb_type", "classifier", "duration", "true", "pred", "count", "recall"])
    print(f"  wrote {results_dir / 'confusion_matrices.csv'} ({len(long_rows)} rows)")


# ---------------------------------------------------------------------------
# 2. Head-to-head accuracy vs duration
# ---------------------------------------------------------------------------

def headtohead(global_df: pd.DataFrame, layer_df: pd.DataFrame, best: dict,
               results_dir: Path):
    """One consolidated figure: best-per-embedding accuracy vs duration."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    durations = sorted(global_df["duration_sec"].unique())

    styles = {
        "xvector": ("#d62728", "o", "-"),
        "whisper_mean": ("#1f77b4", "s", "-"),
    }
    for emb, (color, marker, ls) in styles.items():
        clf = best.get(emb, "knn")
        s = (global_df[(global_df["emb_type"] == emb) & (global_df["classifier"] == clf)]
             .sort_values("duration_sec"))
        ax.plot(s["duration_sec"], s["accuracy"] * 100, marker=marker, ls=ls,
                color=color, label=f"{emb} / {clf}")

    # whisper best single layer (per duration, over all classifiers)
    if layer_df is not None:
        single = layer_df[layer_df["layer"] != "mean"]
        best_layer = (single.loc[single.groupby("duration_sec")["accuracy"].idxmax()]
                      .sort_values("duration_sec"))
        ax.plot(best_layer["duration_sec"], best_layer["accuracy"] * 100,
                marker="^", ls="--", color="#2ca02c",
                label="whisper best single layer")
        for _, r in best_layer.iterrows():
            ax.annotate(f"L{int(r['layer_index'])}",
                        (r["duration_sec"], r["accuracy"] * 100),
                        textcoords="offset points", xytext=(0, -12),
                        fontsize=8, ha="center", color="#2ca02c")

    ax.set_xscale("log")
    ax.set_xticks(durations)
    ax.set_xticklabels([f"{d:g}s" for d in durations])
    ax.set_xlabel("Clip duration")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.set_title("Speaker-ID accuracy vs clip duration\n(x-vector vs Whisper, best classifier each)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = results_dir / "headtohead_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# 3. Per-speaker analysis
# ---------------------------------------------------------------------------

def per_speaker_analysis(pc_df: pd.DataFrame, best: dict, results_dir: Path):
    """Per-speaker recall heatmaps + a hardest-speakers ranking CSV."""
    hard_rows = []
    for emb in [s["label"] for s in EMB_SPECS]:
        if emb not in pc_df["emb_type"].unique():
            continue
        clf = best.get(emb, "knn")
        sub = pc_df[(pc_df["emb_type"] == emb) & (pc_df["classifier"] == clf)]
        pivot = sub.pivot_table(index="speaker_id", columns="duration_sec",
                                values="recall")
        durations = sorted(pivot.columns)
        pivot = pivot[durations]
        # order speakers by mean recall (hardest at top)
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

        fig, ax = plt.subplots(figsize=(1.2 * len(durations) + 3,
                                        0.22 * len(pivot) + 2))
        im = ax.imshow(pivot.values * 100, cmap="RdYlGn", vmin=0, vmax=100,
                       aspect="auto")
        ax.set_xticks(range(len(durations)))
        ax.set_xticklabels([f"{d:g}s" for d in durations])
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=6)
        ax.set_xlabel("Clip duration")
        ax.set_ylabel("Speaker (sorted: hardest → easiest)")
        ax.set_title(f"Per-speaker recall (%) — {emb} / {clf}")
        fig.colorbar(im, ax=ax, label="Recall (%)")
        fig.tight_layout()
        out = results_dir / f"per_speaker_recall_{emb}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  wrote {out}")

        means = pivot.mean(axis=1)
        for spk, mr in means.items():
            hard_rows.append({
                "emb_type": emb, "classifier": clf, "speaker_id": spk,
                "mean_recall": round(float(mr), 6),
                **{f"recall_{d:g}s": round(float(pivot.loc[spk, d]), 6) for d in durations},
            })

    hard_rows.sort(key=lambda r: (r["emb_type"], r["mean_recall"]))
    fields = ["emb_type", "classifier", "speaker_id", "mean_recall"] + \
             [k for k in hard_rows[0] if k.startswith("recall_")] if hard_rows else []
    _write_csv(results_dir / "hard_speakers.csv", hard_rows, fields)
    print(f"  wrote {results_dir / 'hard_speakers.csv'} ({len(hard_rows)} rows)")

    # quick textual summary of the 5 hardest speakers per embedding
    for emb in {r["emb_type"] for r in hard_rows}:
        rows = [r for r in hard_rows if r["emb_type"] == emb][:5]
        print(f"\n  Hardest speakers — {emb}:")
        for r in rows:
            print(f"    {r['speaker_id']}  mean_recall={r['mean_recall']*100:5.1f}%")


def _write_csv(path: Path, rows, fieldnames) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    metadata_root = args.output_root / "metadata"
    ablation_dir = args.ablation_dir or (args.output_root / "results" / "ablation")
    layerwise_dir = args.output_root / "results" / "layerwise"
    results_dir = args.results_dir or (args.output_root / "results" / "analysis")
    results_dir.mkdir(parents=True, exist_ok=True)

    global_df = pd.read_csv(ablation_dir / "global_metrics.csv")
    pc_df = pd.read_csv(ablation_dir / "per_class_metrics.csv")
    layer_csv = layerwise_dir / "layer_metrics.csv"
    layer_df = pd.read_csv(layer_csv) if layer_csv.exists() else None

    best = best_classifier_per_emb(global_df)
    print("Best classifier per embedding (by mean accuracy):")
    for emb in [s["label"] for s in EMB_SPECS]:
        print(f"  {emb}: {best.get(emb)}")

    print("\n[1/3] Confusion matrices ...")
    confusion_matrices(metadata_root, best, results_dir)

    print("\n[2/3] Head-to-head accuracy vs duration ...")
    headtohead(global_df, layer_df, best, results_dir)

    print("\n[3/3] Per-speaker analysis ...")
    per_speaker_analysis(pc_df, best, results_dir)

    print(f"\n{'='*60}\nAnalysis figures written to {results_dir}/")


if __name__ == "__main__":
    main()
