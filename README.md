# SWID — Short-utterance Whisper IDentification

Can a frozen **ASR encoder** (OpenAI Whisper), trained only to transcribe *what* is said,
also tell us *who* is speaking, and how does it hold up when only a fraction of a second of
speech is available?

SWID benchmarks a frozen `whisper-base` encoder against a dedicated speaker encoder
(**x-vectors**) on closed-set speaker identification over VoxCeleb1, across four utterance
durations (0.5 / 1 / 3 / 5 s), three simple back-end classifiers, and a layer-by-layer probe
of the Whisper encoder.

## Key findings

- **Whisper does encode speaker identity.** Mean-pooled kNN on full-encoder embeddings reaches
  **88.3% top-1** accuracy at 5 s — far above the 2.5% chance level (1/40 speakers).
- **But x-vectors win at every duration**, and the gap widens as utterances shorten: x-vectors
  hit **90.3% at 1 s** and **76.1% at 0.5 s**, where Whisper drops below 35%.
- **Speaker information follows an inverted-U over encoder depth**, peaking at an *intermediate*
  layer (layer 3, ~59.6% mean accuracy) rather than the final transcription layer.

> Repurposing a frozen ASR encoder for speaker ID is promising for longer utterances but
> rather limited for short ones.

## Repository layout

```
scripts/                 # the full experimental pipeline (see below)
voxceleb_data/           # VoxCeleb1 splits, clips, embeddings, results (data gitignored)
pretrained_models/       # SpeechBrain x-vector checkpoint
requirements.txt         # Python dependencies
proposal-revised.md      # original project proposal
```

## Pipeline

Each step is a standalone, documented script under `scripts/`. They are designed to run in
sequence; x-vector and Whisper branches share an identical manifest schema so the two are
drop-in comparable.

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `prepare_voxceleb_splits.py`     | Build VoxCeleb1 train/val/test splits + preprocessing artifacts |
| 2 | `extract_xvectors.py`            | Extract x-vector embeddings (SpeechBrain ECAPA/TDNN) |
| 2 | `extract_whisper_embeddings.py`  | Extract per-layer pooled Whisper encoder hidden states |
| 3 | `build_xvector_split_manifests.py` / `build_whisper_split_manifests.py` | Join clip + embedding metadata into split-aware manifests |
| 4 | `build_duration_manifests.py`    | Slice manifests into per-duration (0.5/1/3/5 s) train/val/test sets |
| 5 | `classify_speakers.py`           | Core back-ends: cosine nearest-centroid, kNN, LinearSVC |
| 6 | `run_baseline.py`                | Full sweep: 4 durations × {x-vector, Whisper} × 3 classifiers |
| 7 | `run_duration_ablation.py` + `plot_duration_ablation.py` | Accuracy/F1 vs duration + per-speaker variability |
| 8 | `run_layerwise_probe.py` + `plot_layerwise_probe.py` | Rank Whisper encoder layers by speaker information |
| 9 | `analyze_results.py`             | Consolidate outputs into publication-ready figures and tables |

All scripts expose `--help`. Results are written under `voxceleb_data/processed_test/results/`.

## Setup

```bash
python -m venv myenv && source myenv/bin/activate
pip install -r requirements.txt
```

You will need the [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) test
partition. The frozen models are pulled automatically:
[`openai/whisper-base`](https://huggingface.co/openai/whisper-base) via `transformers`, and the
SpeechBrain x-vector extractor (checkpoint vendored under `pretrained_models/spkrec_xvect/`).

## Quick start

```bash
# 1. Prepare data and embeddings
python scripts/prepare_voxceleb_splits.py     --help
python scripts/extract_xvectors.py            --help
python scripts/extract_whisper_embeddings.py  --help

# 2. Build manifests, then run the headline sweep
python scripts/build_duration_manifests.py    --help
python scripts/run_baseline.py

# 3. The two analyses behind the paper's claims
python scripts/run_duration_ablation.py
python scripts/run_layerwise_probe.py
python scripts/analyze_results.py
```

## Experimental setup

- **Dataset:** VoxCeleb1 test partition — closed-set identification over 40 speakers (chance = 2.5%).
- **Speaker encoder (baseline):** frozen x-vectors (SpeechBrain).
- **ASR encoder (under test):** frozen `whisper-base`; embeddings as last layer, mean-pooled
  across layers, or per individual encoder layer.
- **Durations:** 0.5, 1, 3, 5 s.
- **Back-ends:** cosine nearest-centroid, k-nearest-neighbours, LinearSVC. All embeddings are
  L2-normalised before classification.
