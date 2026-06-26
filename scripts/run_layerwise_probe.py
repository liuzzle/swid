#!/usr/bin/env python3
"""
Step 13 — Layerwise Whisper probing.

Probes **which Whisper encoder layer encodes the most speaker information** by
running the same three classifiers (cosine nearest-centroid, kNN, LinearSVC) on
each hidden state of the pooled Whisper embeddings, independently, at every clip
duration.

Background: the pooled Whisper-base embeddings have shape (n_layers, hidden) =
(7, 512).  Index 0 is the convolutional/embedding output that feeds the encoder;
indices 1..6 are the outputs of the 6 encoder transformer blocks.  Steps 11/12
only ever used the *last* layer (-1) or the *mean* across layers; this step opens
that up and evaluates every layer on its own so we can rank them.

Sweep grid:
  durations    : 0.5s, 1s, 3s, 5s
  layers       : 0 .. n_layers-1 (auto-detected from a sample embedding)
  classifiers  : cosine, knn, svm

For comparison, the ``mean`` (average over all layers) configuration is also
evaluated at each duration and recorded with layer label ``mean`` (layer_index
= -1) so the per-layer numbers can be read against the pooled-mean baseline used
in step 12.

Outputs under <output-root>/results/layerwise/ :
  dur_<d>_layer<L>.json     per-combination results (incl. macro/weighted P/R/F1)
  layer_metrics.csv         one row per (duration, layer, classifier)
  layer_ranking.csv         layers ranked by mean accuracy across durations,
                            per classifier and overall
  layerwise_summary.json    machine-readable copy of layer_metrics rows + ranking

Plots are produced by ``plot_layerwise_probe.py`` (invoked automatically unless
``--no-plots`` is given).

Example:
  python scripts/run_layerwise_probe.py --output-root voxceleb_data/processed_test
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).parent))
from classify_speakers import evaluate  # noqa: E402

DURATIONS = ["0.5s", "1s", "3s", "5s"]
CLASSIFIERS = ["cosine", "knn", "svm"]
DURATION_SECONDS = {"0.5s": 0.5, "1s": 1.0, "3s": 3.0, "5s": 5.0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run step-13 layerwise Whisper probe")
    p.add_argument("--output-root", type=Path, default=Path("voxceleb_data/processed_test"))
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Directory for output files (default: <output-root>/results/layerwise)")
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--svm-c", type=float, default=1.0)
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    return p.parse_args()


def detect_n_layers(manifest_csv: Path) -> int:
    """Read the first embedding referenced by a manifest and return its layer count."""
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    arr = np.load(row["embedding_path"])
    return int(arr.shape[0])


def main() -> None:
    args = parse_args()
    metadata_root = args.output_root / "metadata"
    results_dir = args.results_dir or (args.output_root / "results" / "layerwise")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Detect number of Whisper hidden states from a sample manifest.
    sample = metadata_root / "dur_1s" / "whisper_train.csv"
    n_layers = detect_n_layers(sample)
    # layer_specs: (label, layer_index, whisper_layer_arg)
    layer_specs = [(str(i), i, str(i)) for i in range(n_layers)]
    layer_specs.append(("mean", -1, "mean"))
    print(f"Detected {n_layers} Whisper hidden states; probing layers "
          f"{list(range(n_layers))} + mean", flush=True)

    metric_rows = []   # one row per (duration, layer, classifier)

    for dur in DURATIONS:
        train_csv = metadata_root / f"dur_{dur}" / "whisper_train.csv"
        test_csv = metadata_root / f"dur_{dur}" / "whisper_test.csv"
        if not train_csv.exists() or not test_csv.exists():
            print(f"  SKIP {dur} — whisper manifest not found")
            continue

        for label, layer_idx, layer_arg in layer_specs:
            combo_label = f"dur_{dur}_layer{label}"
            print(f"\n=== {combo_label} ===", flush=True)

            results = evaluate(
                train_csv=train_csv,
                test_csv=test_csv,
                emb_type="whisper",
                whisper_layer=layer_arg,
                classifiers=CLASSIFIERS,
                knn_k=args.knn_k,
                svm_c=args.svm_c,
                per_class=True,
            )

            with (results_dir / f"{combo_label}.json").open("w") as f:
                json.dump(results, f, indent=2)

            for clf_name, clf in results["classifiers"].items():
                metric_rows.append({
                    "duration": dur,
                    "duration_sec": DURATION_SECONDS[dur],
                    "layer": label,
                    "layer_index": layer_idx,
                    "classifier": clf_name,
                    "accuracy": round(clf["accuracy"], 6),
                    "macro_precision": round(clf["macro_precision"], 6),
                    "macro_recall": round(clf["macro_recall"], 6),
                    "macro_f1": round(clf["macro_f1"], 6),
                    "weighted_f1": round(clf["weighted_f1"], 6),
                    "n_train": results["n_train"],
                    "n_test": results["n_test"],
                    "n_speakers": results["n_speakers"],
                    "elapsed_sec": round(clf["elapsed_sec"], 4),
                })
                print(f"  {clf_name:10s}  acc={clf['accuracy']*100:6.2f}%  "
                      f"macro_f1={clf['macro_f1']*100:6.2f}%", flush=True)

    _write_csv(results_dir / "layer_metrics.csv", metric_rows, [
        "duration", "duration_sec", "layer", "layer_index", "classifier",
        "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
        "n_train", "n_test", "n_speakers", "elapsed_sec",
    ])

    ranking_rows = _build_ranking(metric_rows, n_layers)
    _write_csv(results_dir / "layer_ranking.csv", ranking_rows, [
        "classifier", "layer", "layer_index", "mean_accuracy", "mean_macro_f1", "rank",
    ])

    with (results_dir / "layerwise_summary.json").open("w") as f:
        json.dump({"metrics": metric_rows, "ranking": ranking_rows}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results written to {results_dir}/")
    print(f"  layer_metrics.csv  ({len(metric_rows)} rows)")
    print(f"  layer_ranking.csv  ({len(ranking_rows)} rows)")

    _print_pivot(metric_rows, n_layers)
    _print_ranking(ranking_rows)

    if not args.no_plots:
        _make_plots(results_dir)


def _build_ranking(rows, n_layers):
    """Rank single layers (excludes 'mean') by mean accuracy across durations.

    Produces, per classifier and for an 'overall' pseudo-classifier (averaged
    across classifiers too), the per-layer mean accuracy / macro-F1 and a rank
    (1 = best).  'mean' is excluded so the ranking is strictly about individual
    encoder layers.
    """
    single = [r for r in rows if r["layer"] != "mean"]
    ranking = []

    def _rank_group(group_rows, clf_label):
        agg = {}
        for r in group_rows:
            key = (r["layer"], r["layer_index"])
            agg.setdefault(key, {"acc": [], "f1": []})
            agg[key]["acc"].append(r["accuracy"])
            agg[key]["f1"].append(r["macro_f1"])
        entries = [
            {
                "classifier": clf_label,
                "layer": layer,
                "layer_index": idx,
                "mean_accuracy": round(float(np.mean(v["acc"])), 6),
                "mean_macro_f1": round(float(np.mean(v["f1"])), 6),
            }
            for (layer, idx), v in agg.items()
        ]
        entries.sort(key=lambda e: e["mean_accuracy"], reverse=True)
        for i, e in enumerate(entries, start=1):
            e["rank"] = i
        return entries

    for clf in CLASSIFIERS:
        ranking.extend(_rank_group([r for r in single if r["classifier"] == clf], clf))
    ranking.extend(_rank_group(single, "overall"))
    return ranking


def _write_csv(path: Path, rows, fieldnames) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_pivot(rows, n_layers) -> None:
    """Print accuracy pivot: rows=layer, cols=duration, for each classifier."""
    layer_order = [str(i) for i in range(n_layers)] + ["mean"]
    for clf in CLASSIFIERS:
        data = {}
        for r in rows:
            if r["classifier"] != clf:
                continue
            data.setdefault(r["layer"], {})[r["duration"]] = r["accuracy"] * 100
        print(f"\n{'='*60}")
        print(f"Accuracy (%) — {clf} — rows=layer, cols=duration")
        print(f"{'layer':<8}" + "".join(f"{d:>10}" for d in DURATIONS))
        for layer in layer_order:
            if layer not in data:
                continue
            print(f"{layer:<8}" + "".join(
                f"{data[layer].get(d, float('nan')):>10.2f}" for d in DURATIONS))


def _print_ranking(ranking_rows) -> None:
    overall = [r for r in ranking_rows if r["classifier"] == "overall"]
    overall.sort(key=lambda e: e["rank"])
    print(f"\n{'='*60}")
    print("Layer ranking (overall, mean accuracy across durations & classifiers)")
    print(f"{'rank':<6}{'layer':<8}{'mean_acc%':>12}{'mean_macroF1%':>16}")
    for r in overall:
        print(f"{r['rank']:<6}{r['layer']:<8}"
              f"{r['mean_accuracy']*100:>12.2f}{r['mean_macro_f1']*100:>16.2f}")


def _make_plots(results_dir: Path) -> None:
    plot_script = Path(__file__).parent / "plot_layerwise_probe.py"
    print(f"\nGenerating plots via {plot_script.name} ...", flush=True)
    rc = subprocess.call([sys.executable, str(plot_script), "--results-dir", str(results_dir)])
    if rc != 0:
        print(f"  WARNING: plotting exited with code {rc}", flush=True)


if __name__ == "__main__":
    main()
