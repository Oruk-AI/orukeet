# Orukeet Core ML: TapTalk parity and optimization

The converted Orukeet r3 baseline matches Parakeet's Core ML deployment and
measured latency on this Mac. The recommended **greedy** profile removes unused
top-64 calculations, reducing batch latency by **9.5%** in the final paired run
while preserving tested decoding outputs. Both profiles are ready as local
Core ML bundles with a Swift integration library.

[Conversion and integration guide](../../export/coreml/README.md) ·
[Swift engine](../../export/coreml/benchmark/Sources/OrukeetCoreML/OrukeetEngine.swift)

## Identity and deployment

- Source: Orukeet r3, SHA-256
  `031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`.
- Runtime: FluidAudio **0.15.5**, revision
  `19600a485baa4998812e4654b70d2bab8f2c9949`, matching the inspected TapTalk dependency.
- Reference: [pinned Parakeet artifacts](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/tree/7dd20fe6b1797d35f5e3307e8b1732d9a178edfe).
- Baseline `Encoder`, `Decoder`, and `JointDecisionv3` graph files are
  **byte-identical** to the reference. All 699 inference tensors were mapped;
  Orukeet's weights replace the original weight payloads. Complete models and
  transcripts are therefore different from Parakeet.
- The actual reference encoder uses **6-bit LUT palettes plus FP16**, despite
  FluidAudio's historical `.int8` setting name. This conversion matches that
  artifact, including folded batch normalization and position projections.
- Preprocessing and vocabulary are shared. Local loading avoids the NVIDIA
  downloader/cache identity; model initialization is outside request timing.

See [conversion receipt](conversion.json). Its “validation required” status is
the historical export-time status; subsequent validation is recorded below.

## Batch latency

Measured on **Apple M5 Max, 18 CPUs, 128 GiB RAM, macOS 26.4.1 (25E253)**.
Default encoder: CPU/Neural Engine; preprocessing: CPU; decoder and joint:
CPU/Neural Engine. Thermal state was nominal in every recorded measurement.

Final confirmation: four fixtures, 20 repetitions per model/fixture, three
warmups, rotating and reversing model order, batch chunk concurrency four.
Numbers below are median end-to-end transcription request milliseconds,
excluding file and model loading.

| Audio | Duration | Parakeet | Orukeet baseline | Orukeet greedy |
|---|---:|---:|---:|---:|
| Latvian | 7.14 s | 61.09 | 62.26 | 56.82 |
| French | 9.42 s | 76.32 | 76.55 | 69.04 |
| Spanish | 7.68 s | 85.31 | 83.87 | 72.99 |
| JFK English | 11.00 s | 65.26 | 65.48 | 60.11 |

Across 80 matched runs, the median Parakeet/Orukeet latency ratio was **0.9967**
(bootstrap 95% interval 0.9928–1.0009): baseline performance was within **0.3%**.
Greedy/Orukeet was **0.9050** (0.9002–0.9111), or **9.5% lower latency**.
Greedy transcripts and token IDs matched the baseline in all 80 pairs.

[Final raw measurements](final-batch-ane.json) ·
[Final summary, including p95](final-batch-summary.json)

An earlier independent run measured a **7.7%** greedy reduction with the same
output parity. Absolute times differed across sessions; the comparisons above
use matched runs from one session. Bootstrap intervals describe repeat timing
variation on these fixtures, not uncertainty across devices or all speech.
[Earlier raw measurements](batch-ane.json) · [Earlier summary](batch-ane-summary.json)

A **29.95075-second** recording exercised chunked batch inference: medians were
216.77 ms Parakeet, 222.72 ms baseline, and 198.15 ms greedy. The paired greedy
reduction was 7.9%, with a wider interval (ratio 0.8671–0.9519). The upstream
filename says “first-minute,” but the file contains only 29.95075 seconds.
[Long-audio provenance](long-audio-provenance.json) ·
[Measurements](long-batch-ane.json) · [Summary](long-batch-summary.json)

## Streaming

The same optimization works through FluidAudio's sliding-window TDT manager.
Unpaced short-recording throughput improved **7.5%**, with exact final transcript
parity in 40 matched runs. Those short inputs mostly exercise finish-time
flushing; they do not establish early partial-update performance.
[Measurements](streaming-unpaced-ane.json) · [Summary](streaming-unpaced-summary.json)

A separate paced replay fed the 29.95075-second recording in real 100 ms packets,
twice per model. This exercised updates during speech and finalization:

| Median latency | Parakeet | Orukeet baseline | Orukeet greedy |
|---|---:|---:|---:|
| First text after recording starts | 13,142.95 ms | 13,160.93 ms | 13,147.99 ms |
| Finalization after end of audio | 107.80 ms | 90.90 ms | 86.80 ms |

Greedy and baseline final transcripts matched. Two repetitions are a smoke
check, not a robust streaming latency distribution. Paced wall time includes
the recording duration and is not reported as inference throughput.
[Measurements and update events](streaming-paced-ane.json) ·
[Summary](streaming-paced-summary.json)

**The default first-update delay remains about 13 seconds.** FluidAudio's
`.streaming` policy uses an 11-second center plus 2 seconds of lookahead, with
2 seconds of left context on subsequent windows. Compute optimization does not
remove that buffering. The library accepts a custom `SlidingWindowAsrConfig`,
but shorter-context accuracy has not been qualified here. TapTalk's optional
live-typing Parakeet EOU 120M is a separate model and streaming architecture;
this work converts the 0.6B TDT model.

## Accuracy and numerical checks

We sealed 64 FLEURS test recordings before evaluation, then selected the next
64 different files for a second check: eight recordings per sample in each of
English, French, Spanish, Latvian, German, Italian, Ukrainian, and Russian.
Each recording is at most 15 seconds. Dataset revision, sample selection,
audio hashes, references, and hypotheses are retained in the manifests and logs.

| Model | First 64 WER | Next 64 WER | Combined 128 WER | Combined word errors |
|---|---:|---:|---:|---:|
| Parakeet Core ML | 8.81% | 8.31% | 8.55% | 217 |
| Orukeet baseline | 8.15% | 7.18% | 7.64% | 194 |
| Orukeet greedy | 8.15% | 7.18% | 7.64% | 194 |
| Orukeet source, NeMo | 7.57% | 7.03% | 7.29% | 185 |

Greedy text matched baseline text on **all 128 recordings**. The compressed
Core ML deployment retains a small accuracy gap versus the source NeMo model:
9 additional word errors in this sample. It has fewer errors than Parakeet here.

Scores use the repository's existing multilingual, compound-aware scorer;
reference word counts can vary slightly with its compound normalization
(2,539 for Parakeet; 2,538 for baseline/greedy/source). This is a regression
sample, not the full 25-language evaluation or a statistical superiority claim.
The second sample guarantees different recording files, not distinct speakers
or sentences, and is not asserted to be a training-data holdout.

[First corpus](fleurs/corpus.json) · [First scores](fleurs/accuracy.json) ·
[Second corpus](holdout/corpus.json) · [Second scores](holdout/accuracy.json)

Additional checks:

- **2,658 byte comparisons passed** for greedy token ID, token probability,
  and duration outputs against the original joint, using evolving decoder
  states from four recordings on both CPU-only and CPU/Neural Engine.
  [Joint byte audit](joint-byte-parity.json)
- The independent NeMo encoder with the **same exported quantized weights**
  had relative L2 differences of 0.0093–0.0181 and cosine similarity at least
  0.999837 against Core ML. Position construction matched exactly; 96 joint
  token/duration decisions and decoder-state numeric checks passed.
  The larger difference from original FP32 weights is reported separately.
  [Numerical validation](numerical-validation.json)
- Recompiling the distributable preprocessor package preserved byte-identical
  outputs versus the reference compiled preprocessor on two recordings.
  [Preprocessor check](preprocessor-recompile.json)
- Ten Python unit tests passed, and the release Swift package built successfully.
  Tests cover packed palette indices, quantization, LSTM gate order, blob write
  boundaries, paired benchmark completeness, and paced-time reporting.

## Other experiments and portability

The GPU encoder option reduced absolute latency on this M5 Max, and removing
the unused joint calculations still saved **9.7%** against the GPU baseline.
The default remains CPU/Neural Engine: these measurements establish neither
older-device speed nor battery/energy behavior.
[GPU measurements](batch-gpu.json) · [GPU summary](batch-gpu-summary.json)

The optional **precision** profile retains 24 position projections and 24
temporal filters in FP16, adding about 19 MB per encoder representation. It
improved the first sample from 99 to 94 word errors, but the next sample had
95 errors for both precision and baseline. It changes transcripts and graphs
and is **experimental, not the recommended profile**. Its exploratory latency
run was similar to greedy; dataset preparation was active during that run.
[Precision change](precision.json) · [First precision scores](fleurs/precision-accuracy.json) ·
[Second scores](holdout/accuracy.json) · [Timing](precision-batch-summary.json)

The greedy graph reduction does not rely on M5-specific APIs. Older Apple
Silicon devices, macOS versions, power use, a full multilingual evaluation, and
shorter streaming contexts remain unmeasured. Compile the `.mlpackage` files
on the destination OS during installation.

## Deliverables

Local self-contained bundles are under `artifacts/coreml-release-20260915/`:
`greedy` (recommended), `baseline`, and `precision-experimental`. Each includes
four model packages, this Mac's compiled cache, vocabulary, source receipts,
attribution, weight license, and a per-file SHA-256 manifest. Baseline and greedy
are approximately 966 MB each including both portable packages and compiled
caches. Portable packages alone avoid that duplication.

[Bundle checksum verification](package-verification.json) ·
[Standalone-bundle inference smoke check](package-smoke.json)

The conversion, validation, packaging, benchmarks, and Swift integration are
local on branch `codex/taptalk-coreml-parity`. TapTalk and FluidAudio have not
been changed remotely, and no model bundles have been uploaded.
