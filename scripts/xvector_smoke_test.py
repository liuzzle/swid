#!/usr/bin/env python3
"""
Small smoke test for SpeechBrain x-vector extractor.

Usage:
  python scripts/xvector_smoke_test.py --audio path/to/audio.wav

If no audio is provided, a 1s 440Hz sine wave is generated and used.
"""
import argparse
import tempfile
import numpy as np
import soundfile as sf
import torch

from speechbrain.inference.classifiers import EncoderClassifier


def make_sine(path, duration=1.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), False)
    x = 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, x, sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", help="Path to audio file", default=None)
    args = parser.parse_args()

    if args.audio is None:
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        make_sine(tf.name)
        audio_path = tf.name
        print("No audio provided; generated:", audio_path)
    else:
        audio_path = args.audio

    print("Loading x-vector model (this will download weights the first time)...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec_xvect",
    )

    print("Extracting embedding for:", audio_path)
    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav).float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # shape -> (1, N)

    emb = classifier.encode_batch(wav)

    print("Embedding shape:", wav.shape)
    # print first few values
    flat = emb.flatten().cpu().numpy()
    print("First 5 values:", flat[:5].tolist())


if __name__ == "__main__":
    main()
