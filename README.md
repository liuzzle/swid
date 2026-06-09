# swid
SWID: Short-Utterance Whisper Identification

## Step 5: VoxCeleb splits and preprocessing

Use the script below to generate reproducible train/val/test splits,
preprocessed utterances, and short-clip metadata.

```bash
source myenv/bin/activate
python scripts/prepare_voxceleb_splits.py \
	--input-root voxceleb_data/vox1/wav \
	--output-root voxceleb_data/processed \
	--trim-silence \
	--durations 0.5 1.0 3.0 5.0 \
	--clips-per-utterance 1
```

### Artifacts

- `voxceleb_data/processed/preprocessed_wav/...`: preprocessed mono WAV files.
- `voxceleb_data/processed/metadata/utterances.csv`: utterance-level split metadata.
- `voxceleb_data/processed/metadata/clips.csv`: deterministic short-clip metadata.
- `voxceleb_data/processed/metadata/summary.json`: run configuration and counts.

### Optional flags

- `--write-clips`: also materialize clip WAV files under `voxceleb_data/processed/clips/...`.
- `--overwrite`: rebuild existing preprocessed and clip files.
- `--max-files N`: run on a subset (useful for smoke testing).
