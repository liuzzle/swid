#!/usr/bin/env python3
"""
Run the full baseline sweep: all durations × both embedding types × all classifiers.

Imports ``classify_speakers.evaluate()`` and iterates over the 4 duration
variants (0.5s, 1s, 3s, 5s) for both x-vector and Whisper (last layer),
writing per-combination JSON results and a consolidated CSV summary table.

Output layout under <results-dir>/ (default: voxceleb_data/processed_test/results/):

  baseline/
    dur_0.5s_xvector.json
    dur_0.5s_whisper_last.json
    ...
    summary.csv          ← one row per (duration, emb_type, classifier)
    summary.json         ← same data as JSON for programmatic use

Example:
  python scripts/run_baseline.py \\
    --output-root voxceleb_data/processed_test
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).parent))
from classify_speakers import evaluate  # noqa: E402

DURATIONS = ["0.5s", "1s", "3s", "5s"]
EMB_CONFIGS = [
    {"emb_type": "xvector",  "whisper_layer": "-1",   "label": "xvector"},
    {"emb_type": "whisper",  "whisper_layer": "-1",   "label": "whisper_last"},
    {"emb_type": "whisper",  "whisper_layer": "mean", "label": "whisper_mean"},
]
CLASSIFIERS = ["cosine", "knn", "svm"]


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Run full baseline classifier sweep")
    p.add_argument("--output-root", type=Path, default=Path("voxceleb_data/processed_test"))
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Directory for output files (default: <output-root>/results/baseline)")
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--svm-c", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metadata_root = args.output_root / "metadata"
    results_dir = args.results_dir or (args.output_root / "results" / "baseline")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for dur in DURATIONS:
        for cfg in EMB_CONFIGS:
            emb_type = cfg["emb_type"]
            label = cfg["label"]
            whisper_layer = cfg["whisper_layer"]

            train_csv = metadata_root / f"dur_{dur}" / f"{emb_type}_train.csv"
            test_csv  = metadata_root / f"dur_{dur}" / f"{emb_type}_test.csv"

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
            )

            # Save per-combination JSON
            out_path = results_dir / f"{combo_label}.json"
            with out_path.open("w") as f:
                json.dump(results, f, indent=2)

            # Collect summary rows
            for clf_name, clf_results in results["classifiers"].items():
                row = {
                    "duration": dur,
                    "emb_type": label,
                    "classifier": clf_name,
                    "accuracy": f"{clf_results['accuracy']:.4f}",
                    "accuracy_pct": f"{clf_results['accuracy'] * 100:.2f}",
                    "n_train": results["n_train"],
                    "n_test": results["n_test"],
                    "n_speakers": results["n_speakers"],
                    "elapsed_sec": f"{clf_results['elapsed_sec']:.2f}",
                }
                summary_rows.append(row)
                print(f"  {clf_name:10s}  acc={clf_results['accuracy']*100:.2f}%", flush=True)

    # Write CSV summary
    summary_csv = results_dir / "summary.csv"
    fieldnames = ["duration", "emb_type", "classifier", "accuracy", "accuracy_pct",
                  "n_train", "n_test", "n_speakers", "elapsed_sec"]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Write JSON summary
    summary_json = results_dir / "summary.json"
    with summary_json.open("w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results written to {results_dir}/")
    print(f"Summary table: {summary_csv}")

    # Print quick pivot table
    _print_pivot(summary_rows)


def _print_pivot(rows) -> None:
    """Print accuracy as a readable pivot: rows=duration, cols=emb_type×classifier."""
    from collections import defaultdict
    data = defaultdict(dict)
    for r in rows:
        col = f"{r['emb_type']}/{r['classifier']}"
        data[r["duration"]][col] = float(r["accuracy"]) * 100

    cols = sorted({f"{r['emb_type']}/{r['classifier']}" for r in rows})
    header = f"{'dur':<8}" + "".join(f"{c:>25}" for c in cols)
    print(f"\n{'='*60}")
    print("Accuracy (%) — baseline results table")
    print(header)
    for dur in ["0.5s", "1s", "3s", "5s"]:
        if dur not in data:
            continue
        line = f"{dur:<8}" + "".join(f"{data[dur].get(c, float('nan')):>25.2f}" for c in cols)
        print(line)


if __name__ == "__main__":
    main()
