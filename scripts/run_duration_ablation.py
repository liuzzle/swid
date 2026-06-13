#!/usr/bin/env python3
"""
Step 12 — Duration ablation experiments.

Systematically evaluates each embedding type at each clip duration and records
**both per-class (per-speaker) and global metrics**.  This extends the step-11
baseline sweep (which logged global accuracy only) with:

  * macro / weighted precision, recall and F1 (global, label-balanced view), and
  * per-speaker precision / recall / F1 / support (per-class view).

Sweep grid (identical to the baseline so results are directly comparable):
  durations    : 0.5s, 1s, 3s, 5s
  embeddings   : xvector, whisper_last (layer -1), whisper_mean (mean of layers)
  classifiers  : cosine (nearest-centroid), knn (cosine), svm (LinearSVC)

Outputs under <output-root>/results/ablation/ :
  dur_<d>_<emb>.json        per-combination results incl. per-class block
  global_metrics.csv        one row per (duration, emb, classifier): acc + macro/weighted P/R/F1
  per_class_metrics.csv     long form: one row per (duration, emb, classifier, speaker)
  ablation_summary.json     machine-readable copy of global_metrics rows

Plots are produced by ``plot_duration_ablation.py`` (invoked automatically unless
``--no-plots`` is given).

Example:
  python scripts/run_duration_ablation.py --output-root voxceleb_data/processed_test
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).parent))
from classify_speakers import evaluate  # noqa: E402

DURATIONS = ["0.5s", "1s", "3s", "5s"]
EMB_CONFIGS = [
    {"emb_type": "xvector", "whisper_layer": "-1",   "label": "xvector"},
    {"emb_type": "whisper", "whisper_layer": "-1",   "label": "whisper_last"},
    {"emb_type": "whisper", "whisper_layer": "mean", "label": "whisper_mean"},
]
CLASSIFIERS = ["cosine", "knn", "svm"]

# Numeric seconds for plotting / sorting
DURATION_SECONDS = {"0.5s": 0.5, "1s": 1.0, "3s": 3.0, "5s": 5.0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run step-12 duration ablation sweep")
    p.add_argument("--output-root", type=Path, default=Path("voxceleb_data/processed_test"))
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Directory for output files (default: <output-root>/results/ablation)")
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--svm-c", type=float, default=1.0)
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metadata_root = args.output_root / "metadata"
    results_dir = args.results_dir or (args.output_root / "results" / "ablation")
    results_dir.mkdir(parents=True, exist_ok=True)

    global_rows = []      # one row per (duration, emb, classifier)
    per_class_rows = []   # one row per (duration, emb, classifier, speaker)

    for dur in DURATIONS:
        for cfg in EMB_CONFIGS:
            emb_type = cfg["emb_type"]
            label = cfg["label"]
            whisper_layer = cfg["whisper_layer"]

            train_csv = metadata_root / f"dur_{dur}" / f"{emb_type}_train.csv"
            test_csv = metadata_root / f"dur_{dur}" / f"{emb_type}_test.csv"

            if not train_csv.exists() or not test_csv.exists():
                print(f"  SKIP {dur} {label} — manifest not found")
                continue

            combo_label = f"dur_{dur}_{label}"
            print(f"\n=== {combo_label} ===", flush=True)

            results = evaluate(
                train_csv=train_csv,
                test_csv=test_csv,
                emb_type=emb_type,
                whisper_layer=whisper_layer,
                classifiers=CLASSIFIERS,
                knn_k=args.knn_k,
                svm_c=args.svm_c,
                per_class=True,
            )

            # Save per-combination JSON (includes per-class blocks)
            with (results_dir / f"{combo_label}.json").open("w") as f:
                json.dump(results, f, indent=2)

            for clf_name, clf in results["classifiers"].items():
                global_rows.append({
                    "duration": dur,
                    "duration_sec": DURATION_SECONDS[dur],
                    "emb_type": label,
                    "classifier": clf_name,
                    "accuracy": round(clf["accuracy"], 6),
                    "macro_precision": round(clf["macro_precision"], 6),
                    "macro_recall": round(clf["macro_recall"], 6),
                    "macro_f1": round(clf["macro_f1"], 6),
                    "weighted_precision": round(clf["weighted_precision"], 6),
                    "weighted_recall": round(clf["weighted_recall"], 6),
                    "weighted_f1": round(clf["weighted_f1"], 6),
                    "n_train": results["n_train"],
                    "n_test": results["n_test"],
                    "n_speakers": results["n_speakers"],
                    "elapsed_sec": round(clf["elapsed_sec"], 4),
                })

                for spk, m in clf["per_class"].items():
                    per_class_rows.append({
                        "duration": dur,
                        "duration_sec": DURATION_SECONDS[dur],
                        "emb_type": label,
                        "classifier": clf_name,
                        "speaker_id": spk,
                        "precision": round(m["precision"], 6),
                        "recall": round(m["recall"], 6),
                        "f1": round(m["f1"], 6),
                        "support": m["support"],
                    })

                print(f"  {clf_name:10s}  acc={clf['accuracy']*100:6.2f}%  "
                      f"macro_f1={clf['macro_f1']*100:6.2f}%", flush=True)

    _write_csv(results_dir / "global_metrics.csv", global_rows, [
        "duration", "duration_sec", "emb_type", "classifier", "accuracy",
        "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1",
        "n_train", "n_test", "n_speakers", "elapsed_sec",
    ])
    _write_csv(results_dir / "per_class_metrics.csv", per_class_rows, [
        "duration", "duration_sec", "emb_type", "classifier", "speaker_id",
        "precision", "recall", "f1", "support",
    ])
    with (results_dir / "ablation_summary.json").open("w") as f:
        json.dump(global_rows, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results written to {results_dir}/")
    print(f"  global_metrics.csv    ({len(global_rows)} rows)")
    print(f"  per_class_metrics.csv ({len(per_class_rows)} rows)")

    _print_pivot(global_rows)

    if not args.no_plots:
        _make_plots(results_dir)


def _write_csv(path: Path, rows, fieldnames) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_pivot(rows) -> None:
    """Print accuracy and macro-F1 pivots: rows=duration, cols=emb/classifier."""
    cols = sorted({f"{r['emb_type']}/{r['classifier']}" for r in rows})
    for metric, title in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
        data = {}
        for r in rows:
            data.setdefault(r["duration"], {})[f"{r['emb_type']}/{r['classifier']}"] = r[metric] * 100
        print(f"\n{'='*60}")
        print(f"{title} (%) — duration ablation")
        print(f"{'dur':<8}" + "".join(f"{c:>20}" for c in cols))
        for dur in DURATIONS:
            if dur not in data:
                continue
            print(f"{dur:<8}" + "".join(f"{data[dur].get(c, float('nan')):>20.2f}" for c in cols))


def _make_plots(results_dir: Path) -> None:
    plot_script = Path(__file__).parent / "plot_duration_ablation.py"
    print(f"\nGenerating plots via {plot_script.name} ...", flush=True)
    rc = subprocess.call([sys.executable, str(plot_script), "--results-dir", str(results_dir)])
    if rc != 0:
        print(f"  WARNING: plotting exited with code {rc}", flush=True)


if __name__ == "__main__":
    main()
