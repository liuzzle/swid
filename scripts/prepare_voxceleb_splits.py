#!/usr/bin/env python3
"""
Prepare VoxCeleb splits and preprocessing artifacts.

What this script does:
1. Discovers utterance files under VoxCeleb-style layout: speaker/video/utterance.wav
2. Optionally preprocesses each utterance (mono, resample, trim silence)
3. Creates reproducible per-speaker train/val/test utterance splits
4. Generates deterministic short-clip metadata for requested durations
5. Optionally writes cropped/padded clip WAV files to disk

Example:
  python scripts/prepare_voxceleb_splits.py \
      --input-root voxceleb_data/vox1/wav \
      --output-root voxceleb_data/processed \
      --durations 0.5 1.0 3.0 5.0 \
      --clips-per-utterance 2 \
      --write-clips
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf


SUPPORTED_AUDIO_SUFFIX = {".wav"}


@dataclass
class UtteranceRecord:
    speaker_id: str
    video_id: str
    utterance_id: str
    split: str
    input_path: Path
    preprocessed_path: Path
    sample_rate: int
    num_samples: int
    duration_sec: float


def discover_utterances(input_root: Path) -> List[Tuple[str, str, str, Path]]:
    rows: List[Tuple[str, str, str, Path]] = []
    for path in sorted(input_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIX:
            continue

        rel = path.relative_to(input_root)
        parts = rel.parts
        if len(parts) < 3:
            continue

        speaker_id = parts[0]
        video_id = parts[1]
        stem = Path(parts[-1]).stem
        rows.append((speaker_id, video_id, stem, path))

    return rows


def read_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path)
    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), int(sr)


def preprocess_audio(
    wav: np.ndarray,
    sr: int,
    target_sr: int,
    trim_silence: bool,
    trim_top_db: float,
) -> Tuple[np.ndarray, int]:
    out = wav
    if sr != target_sr:
        out = librosa.resample(out, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if trim_silence and out.size > 0:
        trimmed, _ = librosa.effects.trim(out, top_db=trim_top_db)
        if trimmed.size > 0:
            out = trimmed

    if out.size == 0:
        out = np.zeros((1,), dtype=np.float32)

    return out.astype(np.float32), sr


def stable_int_seed(seed: int, key: str) -> int:
    msg = f"{seed}:{key}".encode("utf-8")
    digest = hashlib.blake2b(msg, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def split_per_speaker(
    items: Sequence[Tuple[str, str, str, Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_utterances_per_speaker: int,
    seed: int,
) -> Tuple[Dict[str, List[Tuple[str, str, str, Path]]], List[str]]:
    by_speaker: Dict[str, List[Tuple[str, str, str, Path]]] = {}
    for row in items:
        by_speaker.setdefault(row[0], []).append(row)

    splits: Dict[str, List[Tuple[str, str, str, Path]]] = {"train": [], "val": [], "test": []}
    dropped_speakers: List[str] = []

    for speaker_id, rows in sorted(by_speaker.items()):
        n = len(rows)
        if n < min_utterances_per_speaker:
            dropped_speakers.append(speaker_id)
            continue

        rng = np.random.default_rng(stable_int_seed(seed, speaker_id))
        perm = rng.permutation(n)
        ordered = [rows[i] for i in perm]

        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        n_train = n - n_val - n_test

        if n_train < 1:
            # Keep at least one train item for speaker-supervised models.
            deficit = 1 - n_train
            if n_val >= n_test:
                take = min(deficit, n_val - 1)
                n_val -= take
                deficit -= take
            if deficit > 0:
                take = min(deficit, n_test - 1)
                n_test -= take
                deficit -= take
            n_train = n - n_val - n_test

        train_rows = ordered[:n_train]
        val_rows = ordered[n_train : n_train + n_val]
        test_rows = ordered[n_train + n_val : n_train + n_val + n_test]

        if not train_rows:
            # Defensive fallback, should not happen with constraints above.
            train_rows = ordered[:1]
            val_rows = ordered[1:2]
            test_rows = ordered[2:]

        splits["train"].extend(train_rows)
        splits["val"].extend(val_rows)
        splits["test"].extend(test_rows)

    return splits, dropped_speakers


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare VoxCeleb splits and preprocessing artifacts")
    parser.add_argument("--input-root", type=Path, default=Path("voxceleb_data/vox1/wav"))
    parser.add_argument("--output-root", type=Path, default=Path("voxceleb_data/processed"))
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--trim-silence", action="store_true")
    parser.add_argument("--trim-top-db", type=float, default=30.0)
    parser.add_argument("--durations", type=float, nargs="+", default=[0.5, 1.0, 3.0, 5.0])
    parser.add_argument("--clips-per-utterance", type=int, default=1)
    parser.add_argument("--write-clips", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--min-utterances-per-speaker", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = create_parser().parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")
    if args.clips_per_utterance < 1:
        raise ValueError("--clips-per-utterance must be >= 1")

    input_root: Path = args.input_root
    output_root: Path = args.output_root
    preprocessed_root = output_root / "preprocessed_wav"
    metadata_root = output_root / "metadata"
    clips_root = output_root / "clips"

    ensure_dir(preprocessed_root)
    ensure_dir(metadata_root)
    if args.write_clips:
        ensure_dir(clips_root)

    discovered = discover_utterances(input_root)
    if args.max_files > 0:
        discovered = discovered[: args.max_files]

    if not discovered:
        raise FileNotFoundError(f"No supported audio files found under {input_root}")

    splits, dropped_speakers = split_per_speaker(
        discovered,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_utterances_per_speaker=args.min_utterances_per_speaker,
        seed=args.seed,
    )

    utterance_records: List[UtteranceRecord] = []
    for split_name, rows in splits.items():
        for speaker_id, video_id, utt_id, in_path in rows:
            rel = in_path.relative_to(input_root)
            out_path = preprocessed_root / rel.with_suffix(".wav")
            ensure_dir(out_path.parent)

            if out_path.exists() and not args.overwrite:
                wav, sr = read_audio_mono(out_path)
            else:
                wav, sr = read_audio_mono(in_path)
                wav, sr = preprocess_audio(
                    wav,
                    sr,
                    target_sr=args.target_sr,
                    trim_silence=args.trim_silence,
                    trim_top_db=args.trim_top_db,
                )
                sf.write(out_path, wav, sr)

            duration_sec = float(len(wav) / sr)
            utterance_records.append(
                UtteranceRecord(
                    speaker_id=speaker_id,
                    video_id=video_id,
                    utterance_id=utt_id,
                    split=split_name,
                    input_path=in_path,
                    preprocessed_path=out_path,
                    sample_rate=sr,
                    num_samples=len(wav),
                    duration_sec=duration_sec,
                )
            )

    utterance_csv_rows = []
    for r in utterance_records:
        utterance_csv_rows.append(
            {
                "split": r.split,
                "speaker_id": r.speaker_id,
                "video_id": r.video_id,
                "utterance_id": r.utterance_id,
                "input_path": str(r.input_path),
                "preprocessed_path": str(r.preprocessed_path),
                "sample_rate": r.sample_rate,
                "num_samples": r.num_samples,
                "duration_sec": f"{r.duration_sec:.6f}",
            }
        )
    write_csv(
        metadata_root / "utterances.csv",
        [
            "split",
            "speaker_id",
            "video_id",
            "utterance_id",
            "input_path",
            "preprocessed_path",
            "sample_rate",
            "num_samples",
            "duration_sec",
        ],
        utterance_csv_rows,
    )

    clip_rows = []
    for r in utterance_records:
        wav, sr = read_audio_mono(r.preprocessed_path)
        for duration in args.durations:
            n_target = int(round(duration * sr))
            key = f"{r.speaker_id}/{r.video_id}/{r.utterance_id}/{duration}"
            rng = np.random.default_rng(stable_int_seed(args.seed, key))

            for clip_idx in range(args.clips_per_utterance):
                needs_padding = len(wav) < n_target
                if needs_padding:
                    start = 0
                else:
                    max_start = len(wav) - n_target
                    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
                end = start + n_target

                rel_utt = r.preprocessed_path.relative_to(preprocessed_root)
                duration_tag = f"{duration:g}s"
                clip_rel = Path(
                    duration_tag,
                    r.split,
                    rel_utt.parent,
                    f"{rel_utt.stem}__clip{clip_idx:02d}.wav",
                )
                clip_abs = clips_root / clip_rel

                if args.write_clips:
                    ensure_dir(clip_abs.parent)
                    crop = wav[start:end]
                    if len(crop) < n_target:
                        crop = np.pad(crop, (0, n_target - len(crop)), mode="constant")
                    if args.overwrite or not clip_abs.exists():
                        sf.write(clip_abs, crop.astype(np.float32), sr)

                clip_rows.append(
                    {
                        "split": r.split,
                        "speaker_id": r.speaker_id,
                        "video_id": r.video_id,
                        "utterance_id": r.utterance_id,
                        "source_path": str(r.preprocessed_path),
                        "clip_duration_sec": f"{duration:.3f}",
                        "clip_index": clip_idx,
                        "start_sample": start,
                        "end_sample": end,
                        "start_sec": f"{start / sr:.6f}",
                        "end_sec": f"{end / sr:.6f}",
                        "needs_padding": int(needs_padding),
                        "clip_path": str(clip_abs if args.write_clips else clip_rel),
                    }
                )

    write_csv(
        metadata_root / "clips.csv",
        [
            "split",
            "speaker_id",
            "video_id",
            "utterance_id",
            "source_path",
            "clip_duration_sec",
            "clip_index",
            "start_sample",
            "end_sample",
            "start_sec",
            "end_sec",
            "needs_padding",
            "clip_path",
        ],
        clip_rows,
    )

    split_counts = {
        split_name: sum(1 for r in utterance_records if r.split == split_name)
        for split_name in ("train", "val", "test")
    }
    speaker_counts = {
        split_name: len({r.speaker_id for r in utterance_records if r.split == split_name})
        for split_name in ("train", "val", "test")
    }

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "target_sr": args.target_sr,
        "trim_silence": bool(args.trim_silence),
        "trim_top_db": args.trim_top_db,
        "durations": list(args.durations),
        "clips_per_utterance": args.clips_per_utterance,
        "write_clips": bool(args.write_clips),
        "num_discovered_files": len(discovered),
        "num_kept_utterances": len(utterance_records),
        "split_utterance_counts": split_counts,
        "split_speaker_counts": speaker_counts,
        "dropped_speakers": dropped_speakers,
    }
    with (metadata_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Preparation complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
