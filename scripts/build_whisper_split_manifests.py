#!/usr/bin/env python3
"""
Build split-specific Whisper manifests for classifier input.

This is the Whisper counterpart to ``build_xvector_split_manifests.py``. It
joins the per-file manifest produced by ``extract_whisper_embeddings.py``
(``wav_path -> embedding_path``) with the clip metadata (``clips.csv``) to emit
split-aware CSV manifests:

  - metadata/whisper_train.csv
  - metadata/whisper_val.csv
  - metadata/whisper_test.csv
  - metadata/whisper_all.csv
  - metadata/whisper_summary.json

Each output row carries the fields downstream classifier training expects, with
the same schema as the x-vector manifests so the two are drop-in comparable:

  split, speaker_id, video_id, clip_path, embedding_path, clip_duration_sec

Example:
  python scripts/build_whisper_split_manifests.py \
    --output-root voxceleb_data/processed_test
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_clip_path(clip_path: str) -> Dict[str, str]:
    """Parse a clip path of the form clips/<duration>/<split>/speaker/video/file.wav."""
    parts = Path(clip_path).parts
    try:
        clips_idx = parts.index("clips")
        duration = parts[clips_idx + 1]
        split = parts[clips_idx + 2]
        speaker_id = parts[clips_idx + 3]
        video_id = parts[clips_idx + 4]
        utterance_file = parts[clips_idx + 5]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Unrecognized clip path format: {clip_path}") from exc

    return {
        "duration": duration,
        "split": split,
        "speaker_id": speaker_id,
        "video_id": video_id,
        "utterance_file": utterance_file,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build split manifests from Whisper metadata")
    parser.add_argument("--output-root", type=Path, default=Path("voxceleb_data/processed_test"))
    parser.add_argument(
        "--whisper-manifest",
        type=Path,
        default=None,
        help="Per-file manifest written by extract_whisper_embeddings.py "
        "(defaults to <output-root>/embeddings/whisper_clips/whisper_manifest.csv)",
    )
    parser.add_argument("--clips-metadata", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root: Path = args.output_root
    metadata_root = output_root / "metadata"
    whisper_manifest = args.whisper_manifest or (
        output_root / "embeddings" / "whisper_clips" / "whisper_manifest.csv"
    )
    clips_metadata = args.clips_metadata or (metadata_root / "clips.csv")

    if not whisper_manifest.exists():
        raise FileNotFoundError(f"Missing Whisper manifest: {whisper_manifest}")
    if not clips_metadata.exists():
        raise FileNotFoundError(f"Missing clips metadata: {clips_metadata}")

    # Load clip metadata so we can enrich each row with split/speaker/duration.
    clip_info: Dict[str, Dict[str, str]] = {}
    with clips_metadata.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_info[row["clip_path"]] = {
                "split": row["split"],
                "speaker_id": row["speaker_id"],
                "video_id": row["video_id"],
                "utterance_id": row["utterance_id"],
                "clip_duration_sec": row["clip_duration_sec"],
            }

    rows_by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    all_rows: List[dict] = []
    speaker_ids = set()
    duration_counts: Counter = Counter()
    status_counts: Counter = Counter()
    skipped_non_ok = 0

    with whisper_manifest.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("status") or "").strip()
            status_counts[status or "(blank)"] += 1
            # Only successfully extracted clips get a usable embedding path.
            if status != "ok":
                skipped_non_ok += 1
                continue

            clip_path = row["wav_path"]
            embedding_path = row["embedding_path"]
            if clip_path not in clip_info:
                parsed = parse_clip_path(clip_path)
                split = parsed["split"]
                speaker_id = parsed["speaker_id"]
                video_id = parsed["video_id"]
                duration = parsed["duration"].rstrip("s")
            else:
                info = clip_info[clip_path]
                split = info["split"]
                speaker_id = info["speaker_id"]
                video_id = info["video_id"]
                duration = info["clip_duration_sec"]

            record = {
                "split": split,
                "speaker_id": speaker_id,
                "video_id": video_id,
                "clip_path": clip_path,
                "embedding_path": embedding_path,
                "clip_duration_sec": duration,
            }
            rows_by_split[split].append(record)
            all_rows.append(record)
            speaker_ids.add(speaker_id)
            duration_counts[str(duration)] += 1

    ensure_dir(metadata_root)
    fieldnames = ["split", "speaker_id", "video_id", "clip_path", "embedding_path", "clip_duration_sec"]

    def write_csv_file(path: Path, rows: List[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_csv_file(metadata_root / "whisper_all.csv", all_rows)
    for split_name, rows in rows_by_split.items():
        write_csv_file(metadata_root / f"whisper_{split_name}.csv", rows)

    summary = {
        "output_root": str(output_root),
        "whisper_manifest": str(whisper_manifest),
        "clips_metadata": str(clips_metadata),
        "total_rows": len(all_rows),
        "split_counts": {split_name: len(rows) for split_name, rows in rows_by_split.items()},
        "num_speakers": len(speaker_ids),
        "duration_counts": dict(duration_counts),
        "manifest_status_counts": dict(status_counts),
        "skipped_non_ok": skipped_non_ok,
    }
    with (metadata_root / "whisper_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
