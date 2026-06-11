#!/usr/bin/env python3
"""
Slice the all-duration manifests into per-duration train/val/test manifests.

Step 10 of the project plan. The clip generation (step 5) and embedding
extraction (steps 7/9) already produce data for all four durations in a single
CSV each. This script slices those CSVs by ``clip_duration_sec`` so that
downstream classifier experiments (step 11/12) can load a single clean manifest
per duration rather than filtering on the fly.

Output layout under <output-root>/metadata/:

  dur_0.5s/
    xvector_train.csv  xvector_val.csv  xvector_test.csv
    whisper_train.csv  whisper_val.csv  whisper_test.csv
  dur_1s/
    ...
  dur_3s/
    ...
  dur_5s/
    ...
  duration_manifests_summary.json

Schema of each output file matches the all-duration manifests:
  split, speaker_id, video_id, clip_path, embedding_path, clip_duration_sec

Example:
  python scripts/build_duration_manifests.py \\
    --output-root voxceleb_data/processed_test
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


DURATION_LABELS = {
    "0.500": "0.5s",
    "0.5":   "0.5s",
    "1.000": "1s",
    "1.0":   "1s",
    "1":     "1s",
    "3.000": "3s",
    "3.0":   "3s",
    "3":     "3s",
    "5.000": "5s",
    "5.0":   "5s",
    "5":     "5s",
}

FIELDNAMES = ["split", "speaker_id", "video_id", "clip_path", "embedding_path", "clip_duration_sec"]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_all_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def normalize_duration(raw: str) -> str | None:
    """Return canonical label (e.g. '0.5s') or None if unrecognized."""
    return DURATION_LABELS.get(raw.strip())


def slice_by_duration(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, List]]:
    """Return {dur_label: {split: [rows]}}."""
    result: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        label = normalize_duration(row.get("clip_duration_sec", ""))
        if label is None:
            continue
        result[label][row["split"]].append(row)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-duration manifests for ablation experiments")
    parser.add_argument("--output-root", type=Path, default=Path("voxceleb_data/processed_test"))
    parser.add_argument("--xvector-all", type=Path, default=None,
                        help="Path to xvector_all.csv (default: <output-root>/metadata/xvector_all.csv)")
    parser.add_argument("--whisper-all", type=Path, default=None,
                        help="Path to whisper_all.csv (default: <output-root>/metadata/whisper_all.csv)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_root = args.output_root / "metadata"
    xvector_all = args.xvector_all or (metadata_root / "xvector_all.csv")
    whisper_all = args.whisper_all or (metadata_root / "whisper_all.csv")

    if not xvector_all.exists():
        raise FileNotFoundError(f"Missing: {xvector_all}")
    if not whisper_all.exists():
        raise FileNotFoundError(f"Missing: {whisper_all}")

    xvec_rows = load_all_csv(xvector_all)
    whis_rows = load_all_csv(whisper_all)

    xvec_by_dur = slice_by_duration(xvec_rows)
    whis_by_dur = slice_by_duration(whis_rows)

    all_labels = sorted(set(xvec_by_dur) | set(whis_by_dur))
    summary: Dict = {"durations": {}}

    for label in all_labels:
        dur_dir = metadata_root / f"dur_{label}"
        ensure_dir(dur_dir)

        dur_summary: Dict = {}

        for emb_name, by_dur in [("xvector", xvec_by_dur), ("whisper", whis_by_dur)]:
            splits = by_dur.get(label, {})
            for split in ("train", "val", "test"):
                rows = splits.get(split, [])
                write_csv(dur_dir / f"{emb_name}_{split}.csv", rows)

            split_counts = {s: len(splits.get(s, [])) for s in ("train", "val", "test")}
            all_rows_this_dur = [r for rows in splits.values() for r in rows]
            speakers = {r["speaker_id"] for r in all_rows_this_dur}
            dur_summary[emb_name] = {
                "split_counts": split_counts,
                "total": sum(split_counts.values()),
                "num_speakers": len(speakers),
            }

        summary["durations"][label] = dur_summary
        print(f"  dur_{label}: xvec total={dur_summary['xvector']['total']}  whisper total={dur_summary['whisper']['total']}")

    summary_path = metadata_root / "duration_manifests_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote duration manifests under {metadata_root}/dur_*/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
