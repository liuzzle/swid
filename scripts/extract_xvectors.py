#!/usr/bin/env python3
"""
Extract x-vector embeddings for clips using SpeechBrain pretrained model.

Saves one .npy embedding per clip under <output_root>/embeddings/xvector/<duration>/<split>/...
Also writes a CSV metadata file mapping clip_path -> embedding_path.

Example:
  python scripts/extract_xvectors.py \
    --clips-root voxceleb_data/processed_test/clips \
    --output-root voxceleb_data/processed_test \
    --batch-size 16 \
    --max-files 100
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch

from speechbrain.inference.classifiers import EncoderClassifier


def iter_wavs(clips_root: Path):
    for p in sorted(clips_root.rglob("*.wav")):
        yield p


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_wav_resample(path: Path, target_sr: int = 16000):
    wav, sr = sf.read(path)
    wav = wav.astype('float32')
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return wav, sr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clips-root", type=Path, default=Path("voxceleb_data/processed_test/clips"))
    p.add_argument("--output-root", type=Path, default=Path("voxceleb_data/processed_test"))
    p.add_argument("--model-src", type=str, default="speechbrain/spkrec-xvect-voxceleb")
    p.add_argument("--savedir", type=Path, default=Path("pretrained_models/spkrec_xvect"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    clips_root: Path = args.clips_root
    out_root: Path = args.output_root
    embeddings_root = out_root / "embeddings" / "xvector"
    ensure_dir(embeddings_root)

    print("Loading x-vector model (may download weights)...")
    classifier = EncoderClassifier.from_hparams(source=args.model_src, savedir=str(args.savedir))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    classifier.to(device)

    wav_paths: List[Path] = list(iter_wavs(clips_root))
    if args.max_files > 0:
        wav_paths = wav_paths[: args.max_files]

    meta_rows = []
    batch = []
    batch_paths = []

    def flush_batch():
        nonlocal batch, batch_paths, meta_rows
        if not batch:
            return
        # pad to same length by zero-pad to max
        maxlen = max(x.shape[0] for x in batch)
        toks = []
        for x in batch:
            if x.shape[0] < maxlen:
                pad = np.zeros(maxlen - x.shape[0], dtype=np.float32)
                x = np.concatenate([x, pad])
            toks.append(torch.from_numpy(x).float())
        wav_batch = torch.stack(toks).to(device)
        with torch.no_grad():
            emb = classifier.encode_batch(wav_batch)
        emb = emb.cpu().numpy()

        for i, p in enumerate(batch_paths):
            rel = p.relative_to(clips_root)
            emb_out = embeddings_root / rel.with_suffix('.npy')
            ensure_dir(emb_out.parent)
            if emb_out.exists() and not args.overwrite:
                pass
            else:
                np.save(str(emb_out), emb[i])
            meta_rows.append({"clip_path": str(p), "embedding_path": str(emb_out)})

        batch = []
        batch_paths = []

    for p in wav_paths:
        wav, sr = load_wav_resample(p)
        batch.append(wav)
        batch_paths.append(p)
        if len(batch) >= args.batch_size:
            flush_batch()

    flush_batch()

    # write metadata CSV
    meta_csv = out_root / "metadata" / "xvector_embeddings.csv"
    ensure_dir(meta_csv.parent)
    with meta_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["clip_path", "embedding_path"])
        writer.writeheader()
        for row in meta_rows:
            writer.writerow(row)

    print(f"Wrote {len(meta_rows)} embeddings and metadata to {meta_csv}")


if __name__ == "__main__":
    main()
