# Paired English16 Core ML diagnostic

This runs the exact published Parakeet v2 inference bundle and the published
Orukeet greedy portable bundle through FluidAudio 0.15.5 on the same 16 English
recordings. It is a **tiny reused regression diagnostic**, not an English release
benchmark, a basis to change OpenWhispr's English default, or an iPhone accuracy,
latency, memory, energy, or portability qualification. CI timings are recorded for
debugging; do not rank device performance from them.

The v2 pin contains compiled `.mlmodelc` assets only. Loading them successfully on
this Mac runner does not establish portable source compatibility on iOS. A failed
load is retained as a failure, with no download fallback or replacement model.

## Pins and inference policy

- FluidAudio: `0.15.5`, revision `19600a485baa4998812e4654b70d2bab8f2c9949`.
- [FluidInference's v2 conversion](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-coreml/tree/ee09c569f73759e6d44c9bd16766f477b2b36d39):
  revision `ee09c569f73759e6d44c9bd16766f477b2b36d39`. `v2-assets.json`
  enumerates exactly 21 files (464,413,247 bytes). LFS files use published SHA-256;
  non-LFS MIL graphs use their published Git blob SHA-1. Staging records actual
  SHA-256 for every downloaded file. The encoder is LUT6/FP16, as declared by the
  selected component's metadata. Blank ID is 1024, regardless of vocabulary count.
- [Orukeet portable inference bundle](https://huggingface.co/oruk/orukeet/tree/43142dd1897f9ddadcd70173fcb5ff45c08aa951/coreml):
  revision `43142dd1897f9ddadcd70173fcb5ff45c08aa951`, greedy ZIP SHA-256
  `beccdc6f18c4b10527a764f6e3ab12e3e11b969220c0cee175b3bb7eaa94290e`.
  The runner independently verifies the archive and every extracted source byte,
  then compiles fresh on the executing Mac.
- `fixtures.json` is the unchanged sealed FLEURS English16 manifest, SHA-256
  `46d5db4c0a5b92557bf378bfeb9a7b150029714d449dc462d1a1b945475428d2`.
  [Google FLEURS](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd),
  revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, English test split.
  The original selection skipped the first 16 eligible archive records and sealed
  the next 16 before earlier inference. These files have already been used in
  prior Orukeet diagnostics; they are not a fresh holdout. Total: 150.28 seconds.

Both models receive unchanged mono 16 kHz Float32 recordings. Every clip has a
new decoder state. Both managers use batch inference, chunk concurrency 1,
CPU preprocessor, CPU/ANE decoder and joint, and the same encoder compute policy
(`ane` by default; `cpu` is available for a separate labeled diagnostic).
The actual component policies are checked after loading. Neither loader invokes
ModelHub or any inference-time network/download recovery.

## Run on GitHub-hosted macOS CI

Model/audio staging and the runner reject local execution. Run them only from an
explicit GitHub-hosted macOS job (`GITHUB_ACTIONS=true`,
`RUNNER_ENVIRONMENT=github-hosted`, `RUNNER_OS=macOS`). Do not set these variables
to bypass the guard on a workstation. No training checkpoints are used.

From the repository root, after the existing bundle-verifier workflow has staged
and authenticated `greedy.zip` and extracted `orukeet-r3-coreml-greedy`:

```sh
python3 export/coreml/english-comparison/stage.py \
  --destination "$ORUKEET_COMPARISON_ROOT"
python3 export/coreml/english-comparison/run.py \
  --v2-models "$ORUKEET_COMPARISON_ROOT/v2" \
  --audio-root "$ORUKEET_COMPARISON_ROOT/audio" \
  --orukeet-packages "$ORUKEET_COMPARISON_ROOT/orukeet-r3-coreml-greedy" \
  --orukeet-archive "$ORUKEET_COMPARISON_ROOT/greedy.zip" \
  --output "$ORUKEET_COMPARISON_EVIDENCE/results" \
  --encoder-units ane
```

`--output` must be a new directory. It contains JSON and logs only. The temporary
compiled cache lives beside the portable packages and is cleaned even on failure.
`stage.py` streams the pinned FLEURS archive, writes only the 16 selected members
without altering their bytes, and stops once all selected members are found.
Existing staged files are fully reverified before reuse.

The runner writes `raw.json` with all 32 model/clip results, raw transcripts,
audio hashes, asset identities, runtime/configuration, diagnostic timings and
explicit failures. It writes `scored.json` only after complete successful
inference and strict pair/hash validation. Each failure returns a nonzero status;
available evidence remains in `failure.json`, `raw.json` and logs. Upload only
JSON/log evidence, never model or audio directories.

Scoring uses each sealed `normalized_reference.split()` unchanged, giving the
same **343 reference words per model**. Hypothesis normalization and deterministic
edit alignment are documented in `score.py` and included in the score output.
Per-clip substitutions, deletions, insertions, raw text, normalized hypotheses,
micro-WER and paired deltas remain inspectable. Apostrophes and hyphens are
preserved. This fixed scoring convention is specific to this diagnostic.

## Local checks without model downloads

```sh
python3 -m unittest discover -s export/coreml/english-comparison -p 'test_*.py' -v
swift build --package-path export/coreml/english-comparison -c release --jobs 4
```

These checks use synthetic data and Swift source dependencies only. They do not
stage or transcribe model assets. Imports do not trigger downloads.

FLEURS audio/reference text: Google, CC-BY-4.0. Parakeet v2: NVIDIA;
Core ML conversion: FluidInference, CC-BY-4.0. Orukeet attribution and licenses
remain in the authenticated portable bundle. See linked pinned source cards.
