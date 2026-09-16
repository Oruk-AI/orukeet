# Orukeet Core ML for TapTalk

Orukeet r3 running through **FluidAudio 0.15.5**, the version pinned by TapTalk.
Includes conversion, a local Swift engine, paired batch/streaming benchmarks,
component validation, and two optional optimization profiles.

## What is identical

The baseline exporter preserves the upstream `Encoder`, `Decoder`, and
`JointDecisionv3` **model.mlmodel files byte for byte**. It changes the weight
blob payloads to Orukeet's r3 parameters. The preprocessor and vocabulary are
copied from the pinned reference. Orukeet's model weights and transcripts are
different from NVIDIA Parakeet's; the complete bundles cannot be byte-identical.

The exporter verifies the r3 checkpoint SHA-256, the three upstream graph
checksums, every mapped tensor's shape, and coverage of all 699 inference
tensors. It reconstructs folded batch normalization, relative-position
projections, and the LSTM gate permutation. It adds no Gabor-specific inference
operations.

**Precision matters:** the file called `Encoder.mlmodelc` by FluidAudio's
`.int8` setting actually contains 6-bit LUT palettes plus FP16 tensors. The
reference repository now documents this historical naming mismatch. This
converter matches the artifact used by TapTalk 0.15.5.

## Profiles

| Profile | Changes | Intended use |
|---|---|---|
| `baseline` | Orukeet weights in byte-identical Parakeet graphs | Exact deployment baseline; retains top-64 outputs |
| `greedy` | Removes unused top-64 outputs and their dependencies from the joint | TapTalk's ordinary decoding; same token, probability, and duration outputs |
| `precision` | Greedy joint plus FP16 position projections and temporal filters | Optional accuracy experiment; changes transcripts and encoder graph |

Use **greedy** for the measured speed improvement with exact output parity to
the baseline. Keep **baseline** when adding language hints or top-K vocabulary
reranking. The precision experiment improved the first regression sample but
did not reduce word-error counts on the held-out sample, so it is not the default.

See [measured results](../../evidence/coreml-taptalk-20260915/README.md).

## Download the preview

Portable model bundles are available on
[Hugging Face](https://huggingface.co/oruk/orukeet/tree/coreml-taptalk-preview-20260915/coreml)
and attached to the
[Core ML TapTalk preview release](https://github.com/Oruk-AI/orukeet/releases/tag/coreml-taptalk-preview-20260915):

- `orukeet-r3-coreml-greedy.zip` — recommended for ordinary greedy decoding.
- `orukeet-r3-coreml-baseline.zip` — preserves the reference graphs and top-64 outputs.
- `SHA256SUMS.txt` — archive checksums; each bundle also includes a per-file manifest.

Both hosts serve identical archives. The [Hugging Face guide](huggingface/README.md)
includes direct URLs and a hash-checked Python download example. Use the
`coreml-taptalk-preview-20260915` release tag when downloading from `oruk/orukeet`;
the [publication receipt](../../evidence/coreml-taptalk-20260915/huggingface-publication.json)
records the exact Hub commit for applications that pin revisions:
`43142dd1897f9ddadcd70173fcb5ff45c08aa951`.

These archives contain `.mlpackage` files. Compile them on the destination Mac
using `OrukeetLocalModels.compilePackages` below. The preview is separate from
the stable Metal release; the precision experiment remains available through
the conversion scripts and is not included in these downloads.

## Reproduce conversion

Run from the repository root on Apple Silicon, macOS 14 or later. These commands
use Python 3.11. The conversion itself needs no NeMo installation or GPU.

```sh
uv venv --python 3.11 export/coreml/.venv
uv pip install --python export/coreml/.venv/bin/python -r export/coreml/requirements.txt

export/coreml/.venv/bin/python export/coreml/download_reference.py \
  --output artifacts/coreml/parakeet

# Obtain the released source using the repository's pinned artifact catalog.
export/coreml/.venv/bin/python -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download("oruk/orukeet", "orukeet-v0.1.0.nemo", revision="555136b50265a132d4cea0d35560c26fc4f657ab", local_dir="artifacts/coreml/source"))'

export/coreml/.venv/bin/python export/coreml/convert.py \
  --source artifacts/coreml/source/orukeet-v0.1.0.nemo \
  --reference artifacts/coreml/parakeet --output artifacts/coreml/baseline

export/coreml/.venv/bin/python export/coreml/optimize_joint.py \
  --models artifacts/coreml/baseline --output artifacts/coreml/greedy
```

Outputs must be new directories. The converter compiles with the Core ML runtime
API, so a full Xcode installation or `coremlcompiler` executable is unnecessary.
The optimized working directory symlinks unchanged models to the baseline.
Use `package.py` below to create a self-contained distributable bundle.

## TapTalk integration

The local Swift package at `export/coreml/benchmark` exports the
`OrukeetCoreML` library and the `CoreMLBenchmark` executable. Its FluidAudio
dependency is pinned to 0.15.5 in both the manifest and resolved revision.

```swift
import OrukeetCoreML

// Run during installation. Compile packages on the destination Mac; a compiled
// cache created on macOS 26 is not a compatibility test for macOS 14.
try OrukeetLocalModels.compilePackages(from: downloadedBundle, to: installedCache)

let engine = OrukeetEngine(modelDirectory: installedCache)
try await engine.ensureLoaded()
let result = try await engine.transcribe(samples: mono16kSamples)
print(result.text)

// FluidAudio's sliding-window TDT interface, using the same loaded models.
let streaming = try await engine.makeStreamingManager()
// Consume transcriptionUpdates, startStreaming(), streamAudio(), then finish().
```

The engine mirrors TapTalk's actor lifecycle and creates fresh decoder state
per recording. It loads explicit local components through `AsrModels` rather
than the NVIDIA downloader, which otherwise assumes Parakeet's cache identity.
TapTalk can keep its existing capture, resampling, VAD, and paste paths.

The default encoder uses `.cpuAndNeuralEngine`; preprocessing uses `.cpuOnly`.
Decoder and joint retain FluidAudio's `.cpuAndNeuralEngine` setting. An explicit
`.cpuAndGPU` encoder override is available for plugged-in throughput workloads.
Measure that option on the target device; this Mac's GPU advantage does not
establish a universal default for older chips, iPhones, or battery use.

## Benchmark

```sh
swift build -c release --package-path export/coreml/benchmark
export/coreml/benchmark/.build/release/CoreMLBenchmark \
  --models parakeet=artifacts/coreml/parakeet \
  --models orukeet=artifacts/coreml/baseline \
  --models orukeet-greedy=artifacts/coreml/greedy \
  --audio demos/fixtures/jfk.wav --repeat 20 --warmup 3 \
  --output artifacts/coreml/batch.json

python3 export/coreml/summarize.py artifacts/coreml/batch.json \
  --baseline orukeet --require-parity orukeet-greedy \
  --output artifacts/coreml/batch-summary.json
```

Repeat `--audio` for each fixture. Inputs must be nonempty mono 16 kHz files.
Use `--mode streaming` for unpaced sliding-window throughput and add `--paced`
to feed real 100 ms packets against a monotonic clock. Paced wall time includes
recording duration; first-text and finalization latency are separate fields.
`--encoder-units ane|gpu|cpu|all` selects the encoder backend, and
`--concurrency N` controls the batch chunk concurrency.

The benchmark excludes audio loading and model loading from request timing,
records model initialization separately, warms each fixture/model, and rotates
the execution order. It records raw timings, exact transcripts/token IDs,
hardware, OS, and thermal state. Initialization timing reflects the OS caches
that already exist; it is not a cache-cleared cold-start measurement.

**Streaming scope:** this is FluidAudio's overlapping-window TDT path. The
default `.streaming` configuration has an 11-second center, 2 seconds of left
context and 2 seconds of lookahead. Its first update is consequently dominated
by buffering. TapTalk's optional live-typing model is the separate Parakeet EOU
120M, with different weights and a different streaming architecture. This
conversion does not turn Orukeet into that model. Applications may pass a custom
`SlidingWindowAsrConfig` to trade context, update frequency, and accuracy.

## Validate accuracy and exact outputs

```sh
uv pip install --python export/coreml/.venv/bin/python -r export/coreml/requirements-validation.txt
export/coreml/.venv/bin/python -m unittest discover -s export/coreml -p 'test_*.py'

export/coreml/.venv/bin/python export/coreml/validate.py \
  --source artifacts/coreml/source/orukeet-v0.1.0.nemo \
  --models artifacts/coreml/baseline --audio demos/fixtures/jfk.wav \
  --output artifacts/coreml/numerical.json

export/coreml/.venv/bin/python export/coreml/audit_joint.py \
  --baseline artifacts/coreml/baseline --candidate artifacts/coreml/greedy \
  --audio demos/fixtures/jfk.wav --output artifacts/coreml/joint-parity.json
```

The numerical validator reports FP32-source drift separately from conversion
error against the **same exported weights** executed in NeMo. This distinguishes
6-bit compression loss from incorrect conversion. NeMo transcription restores
some modules to training mode, so the validator explicitly restores `eval()`.

`prepare_fleurs.py` seals real FLEURS test samples before inference.
`--skip-per-language 8` selects the next eight eligible recordings for a held-out
check. `evaluate.py` runs all Core ML variants and the NeMo source, then applies
the repository's existing multilingual/compound-aware WER scorer. The included
128-recording experiment is a regression check, not a full 25-language release
evaluation. Speaker/sentence independence between the two samples is not
guaranteed; the selection guarantees different recording files.

## Package

```sh
export/coreml/.venv/bin/python export/coreml/package.py \
  --models artifacts/coreml/greedy --output artifacts/coreml/orukeet-greedy \
  --profile greedy
```

This resolves symlinks, includes all four `.mlpackage` components, the vocabulary,
weight license, attribution, conversion receipts and per-file SHA-256 checksums.
Add `--include-compiled` for a ready-to-run local cache. Compile the packages on
other OS versions during installation. The public preview archives omit
compiled caches. TapTalk and FluidAudio source are unchanged by this release.

## Sources and attribution

- [TapTalk's pinned dependency](https://github.com/vakharwalad23/tap-talk/blob/main/project.yml)
- [FluidAudio v0.15.5](https://github.com/FluidInference/FluidAudio/tree/v0.15.5), Apache-2.0
- [Pinned Core ML artifacts](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/tree/7dd20fe6b1797d35f5e3307e8b1732d9a178edfe)
- [Fluid Inference conversion research](https://github.com/FluidInference/mobius/tree/main/models/stt/parakeet-tdt-v3-0.6b/coreml)
- [Orukeet weight attribution](../../NOTICE.md) and [CC BY-SA 4.0 license](../../LICENSE-WEIGHTS)

The conversion and integration code here is MIT. NVIDIA and Fluid Inference
attribution is retained in distributable bundles.
