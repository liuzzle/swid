#!/usr/bin/env python3
"""Extract encoder hidden states (or pooled vectors) from a Whisper model.

Saves per-file outputs as .npz containing per-layer pooled vectors (or raw states).

Usage examples:
  python scripts/extract_whisper_embeddings.py --input-wav example.wav --out-dir embeddings/
  python scripts/extract_whisper_embeddings.py --manifest clips.csv --out-dir embeddings/ --pool mean
"""
from pathlib import Path
import argparse
import torch
import numpy as np
import soundfile as sf
import csv
import json
import time
from transformers import WhisperProcessor, WhisperModel


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_audio(path: Path, sr: int = 16000):
    data, samplerate = sf.read(str(path))
    if samplerate != sr:
        import librosa

        data = librosa.resample(data.astype(float), orig_sr=samplerate, target_sr=sr)
        samplerate = sr
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), samplerate


def extract_for_file(path: Path, model, processor, device, pool: str, out_dir: Path):
    audio, sr = load_audio(path, sr=processor.feature_extractor.sampling_rate)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        # call the encoder directly to avoid decoder-related kwargs conflicts
        outputs = model.encoder(input_features, output_hidden_states=True, return_dict=True)

    # encoder returns hidden states in `hidden_states`
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Couldn't find encoder hidden states in model output")

    # hidden_states is a tuple/list of tensors: (layer_0, layer_1, ..., layer_N)
    # Each tensor shape: (batch=1, seq_len, hidden)
    per_layer = []
    for hs in hidden_states:
        arr = hs.cpu().numpy()[0]
        if pool == "none":
            per_layer.append(arr)
        elif pool == "mean":
            per_layer.append(arr.mean(axis=0))
        elif pool == "max":
            per_layer.append(arr.max(axis=0))
        else:
            raise ValueError(f"Unknown pool: {pool}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    if pool == "none":
        # save raw per-layer arrays into an npz
        np.savez_compressed(out_dir / f"{stem}_whisper_layers.npz", *per_layer)
    else:
        # stack into (num_layers, hidden)
        stacked = np.stack(per_layer, axis=0)
        np.save(out_dir / f"{stem}_whisper_pooled.npy", stacked)


def main():
    parser = argparse.ArgumentParser(description="Extract Whisper encoder hidden states or pooled vectors")
    parser.add_argument("--input-wav", type=Path, help="Single input wav file")
    parser.add_argument("--manifest", type=Path, help="CSV with a column 'wav_path' listing files to process")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write embeddings")
    parser.add_argument("--model-name", default="openai/whisper-base", help="Hugging Face model name")
    parser.add_argument("--pool", choices=["mean", "max", "none"], default="mean", help="Pooling over time")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of files to process in a batch (for CPU/GPU)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files with existing embeddings in out-dir")
    parser.add_argument("--manifest-out", type=Path, default=None, help="Path to write per-file manifest CSV (defaults to <out-dir>/whisper_manifest.csv)")
    parser.add_argument("--summary-out", type=Path, default=None, help="Path to write summary JSON (defaults to <out-dir>/whisper_summary.json)")
    parser.add_argument("--device", default="auto", help="Device: 'auto', 'cpu' or 'cuda'")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading processor+model: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperModel.from_pretrained(args.model_name).to(device)
    model.eval()

    files = []
    if args.input_wav:
        files.append(args.input_wav)
    if args.manifest:
        import csv

        with open(args.manifest, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if "wav_path" in row and row["wav_path"]:
                    files.append(Path(row["wav_path"]).expanduser())

    if not files:
        raise SystemExit("No input files specified")

    # process files in batches to leverage batching on CPU/GPU
    batch_size = max(1, int(args.batch_size))
    def chunked(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i : i + n]

    manifest_rows = []

    def extract_batch(paths):
        # load audios
        audios = []
        paths_to_process = []
        for p in paths:
            stem = Path(p).stem
            if args.pool == "none":
                out_name = f"{stem}_whisper_layers.npz"
            else:
                out_name = f"{stem}_whisper_pooled.npy"
            out_path = Path(args.out_dir) / out_name
            if args.skip_existing and out_path.exists():
                manifest_rows.append({"wav_path": str(p), "embedding_path": str(out_path), "status": "skipped"})
                continue
            try:
                a, _ = load_audio(p, sr=processor.feature_extractor.sampling_rate)
            except Exception as e:
                manifest_rows.append({"wav_path": str(p), "embedding_path": None, "status": "error", "error": str(e)})
                continue
            audios.append(a)
            paths_to_process.append((p, out_path))

        if not paths_to_process:
            return

        inputs = processor(audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
        input_features = inputs.input_features.to(device)
        # Whisper encoder expects a fixed mel-frame length; pad/truncate to expected length
        conv1_stride = model.encoder.conv1.stride[0]
        conv2_stride = model.encoder.conv2.stride[0]
        expected_seq_length = int(model.config.max_source_positions * conv1_stride * conv2_stride)
        cur_len = input_features.size(-1)
        if cur_len < expected_seq_length:
            pad_amt = expected_seq_length - cur_len
            pad_tensor = torch.zeros((*input_features.shape[:-1], pad_amt), dtype=input_features.dtype, device=input_features.device)
            input_features = torch.cat([input_features, pad_tensor], dim=-1)
        elif cur_len > expected_seq_length:
            input_features = input_features[..., :expected_seq_length]

        with torch.no_grad():
            outputs = model.encoder(input_features, output_hidden_states=True, return_dict=True)

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Couldn't find encoder hidden states in model output")

        # hidden_states: tuple of (layer) tensors with shape (batch, seq_len, hidden)
        batch_size_local = input_features.size(0)
        per_file_per_layer = [ [] for _ in range(batch_size_local) ]

        for hs in hidden_states:
            # hs: (B, S, H)
            if args.pool == "none":
                arr = hs.cpu().numpy()
                for i in range(batch_size_local):
                    per_file_per_layer[i].append(arr[i])
            elif args.pool == "mean":
                pooled = hs.mean(dim=1).cpu().numpy()  # (B, H)
                for i in range(batch_size_local):
                    per_file_per_layer[i].append(pooled[i])
            elif args.pool == "max":
                pooled = hs.max(dim=1).values.cpu().numpy()
                for i in range(batch_size_local):
                    per_file_per_layer[i].append(pooled[i])
            else:
                raise ValueError(f"Unknown pool: {args.pool}")

        # write outputs per file
        for (path, out_path), layers in zip(paths_to_process, per_file_per_layer):
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if args.pool == "none":
                    np.savez_compressed(out_path, *layers)
                    layer_count = len(layers)
                else:
                    stacked = np.stack(layers, axis=0)
                    np.save(out_path, stacked)
                    layer_count = stacked.shape[0]
                manifest_rows.append({"wav_path": str(path), "embedding_path": str(out_path), "status": "ok", "layer_count": int(layer_count)})
            except Exception as e:
                manifest_rows.append({"wav_path": str(path), "embedding_path": str(out_path), "status": "error", "error": str(e)})

    start_time = time.time()
    for batch in chunked(files, batch_size):
        print(f"Processing batch of {len(batch)} files, example: {batch[0]}")
        extract_batch(batch)
    elapsed = time.time() - start_time

    # write manifest and summary
    manifest_out = args.manifest_out or (Path(args.out_dir) / "whisper_manifest.csv")
    summary_out = args.summary_out or (Path(args.out_dir) / "whisper_summary.json")
    ensure_dir(Path(manifest_out).parent)
    with open(manifest_out, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["wav_path", "embedding_path", "status", "layer_count", "error"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary = {
        "model": args.model_name,
        "pool": args.pool,
        "batch_size": batch_size,
        "skip_existing": bool(args.skip_existing),
        "n_files_requested": len(files),
        "n_processed": sum(1 for r in manifest_rows if r.get("status") == "ok"),
        "n_skipped": sum(1 for r in manifest_rows if r.get("status") == "skipped"),
        "n_errors": sum(1 for r in manifest_rows if r.get("status") == "error"),
        "elapsed_seconds": elapsed,
    }
    with open(summary_out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote manifest: {manifest_out}")
    print(f"Wrote summary: {summary_out}")


if __name__ == "__main__":
    main()