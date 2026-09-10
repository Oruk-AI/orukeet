# Reproducing the OpenWhispr ONNX qualification

`qualify-sherpa.cjs` is the portable harness. It imports the selected OpenWhispr checkout's actual managers and runs under Electron. `score-paired-sherpa.py` scores every selected output with Orukeet's existing English metric. The recorded source hashes identify the implementation used for each run.

Use an isolated output directory. The harness writes raw transcripts there; the published scores retain only aggregate and per-clip edit counts.

## Inputs

- An OpenWhispr checkout containing the Orukeet registry entry, with dependencies installed and `npm run download:sherpa-onnx` completed.
- The Orukeet repository, including its `demos` directory and `evaluation/unseen/metrics.py`.
- Extracted Orukeet and stock Parakeet directories, each containing `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx` and `tokens.txt`.
- A local manifest containing the selected audio files, exact reference text, duration and SHA-256. The scoring script requires an explicit `--manifest`; no dataset path is inferred.

The published count rows use the fixed 640-clip selection recorded by `manifest_sha256`. The audio selection contains 64 clips per corpus and does not depend on either model's outputs. Changing its audio, references or selection produces a different comparison.

## Run

Set these paths to local checkouts and extracted models:

```bash
APP=/path/to/openwhispr
MODEL_REPO=/path/to/orukeet
ORUKEET_ONNX=/path/to/sherpa-onnx-orukeet-v0.1.0-int8
PARAKEET_ONNX=/path/to/parakeet-tdt-0.6b-v3
MANIFEST=/path/to/eval-manifest.json
RESULTS=/path/to/private-validation
```

From the OpenWhispr directory, run:

```bash
cd "$APP"
npx --no-install electron "$MODEL_REPO/evidence/onnx-r3-20260910/qualify-sherpa.cjs" \
  --checkout "$APP" \
  --fixtures "$MODEL_REPO/demos" \
  --model orukeet-v0.1.0-q8 \
  --model-dir "$ORUKEET_ONNX" \
  --stock-dir "$PARAKEET_ONNX" \
  --output-root "$RESULTS" \
  --run paired-640 \
  --mode benchmark \
  --manifest "$MANIFEST"
```

The `orukeet-v0.1.0-q8` value is the app's existing internal model identifier. This registry entry loads the ONNX INT8 package through the offline sherpa runtime. `--binary` can explicitly select an unmodified shipped sherpa WebSocket binary; otherwise the app resolves its bundled binary normally. On a headless Linux runner, prefix the Electron command with `xvfb-run --auto-servernum` and pass Electron `--no-sandbox`.

Omit `--mode benchmark` and `--manifest` to run lifecycle checks: loading, repeated transcription, multilingual normalization, simultaneous requests, silence, cancellation, 44-second segmentation, preview-sized prefixes and restart. Choose a new `--run` for each invocation so prior evidence stays intact.

The timing run keeps one warm server per model. Calls execute sequentially and alternate the first model for each clip. Loading is recorded separately. Warm timings include the app's audio normalization, 15-second segmentation, WebSocket exchange and recognition. They measure file transcription; microphone and saved-history timing uses a separate UI check.

## Score

Use a Python environment with `numpy`, `rapidfuzz` and `whisper-normalizer` installed. The recorded comparison uses package versions listed in `scores.json`.

```bash
python "$MODEL_REPO/evidence/onnx-r3-20260910/score-paired-sherpa.py" \
  "$RESULTS/paired-640" \
  --manifest "$MANIFEST" \
  --repo "$MODEL_REPO"
```

This writes `scores.json` and `scores.md`. The scorer verifies complete selection order, each audio hash, each reference, matching denominators and the raw-result hash. Empty API results are scored as empty transcripts. Runtime errors remain visible and prevent a passing scoring status.

`scores.json` includes per-clip word/character edit counts without reference or hypothesis text. Its aggregate WER and stratified bootstrap can be recomputed directly from these rows. The exact executed scripts are preserved separately as text evidence when the portable script's path options differ from the recorded local invocation.

## Manifest shape

```json
{
  "selection": "Description of the fixed selection made before inference",
  "clips": [
    {
      "id": "corpus:source-index",
      "dataset": "corpus",
      "path": "/local/audio/clip.wav",
      "sha256": "SHA-256 of the original encoded audio bytes",
      "reference": "Exact reference transcript",
      "audio_seconds": 7.5
    }
  ]
}
```
