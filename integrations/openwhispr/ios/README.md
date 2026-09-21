# OpenWhispr iOS: Orukeet batch transcription

This integration preserves OpenWhispr's record-then-transcribe flow: mono
16 kHz floating-point PCM goes into FluidAudio/Core ML on the device. Orukeet r3
is a v3 multilingual model, including English. Start with it as an optional
multilingual selection; retain Parakeet v2 for English until a paired evaluation
on OpenWhispr recordings justifies changing that selection.

**Status: integration candidate, not an iPhone-qualified release.** The portable
weights already exist. This change adds iOS 17 to the Swift package, a bounded
batch engine and installation/input lifecycle checks. Local macOS inference is
separate from iOS build and physical-device qualification. See the
[validation record](../../../evidence/coreml-openwhispr-20260920/README.md).

## Version and model contract

| Item | Candidate |
|---|---|
| Swift package | Local package `export/coreml/benchmark`; library product `OrukeetCoreML` |
| Runtime | FluidAudio **0.15.5**, revision `19600a485baa4998812e4654b70d2bab8f2c9949` |
| Platform declaration | iOS 17+, macOS 14+; Swift 6 |
| Model lineage | Orukeet r3, Parakeet TDT 0.6B v3 architecture |
| Audio | 16,000 Hz, one channel, normalized finite `[Float]` PCM (approximately −1…1); at least 300 ms |
| Decoder | v3; 8,192 vocabulary entries; fresh decoder state per recording |
| Default placement | Preprocessor CPU; encoder/decoder/joint CPU + Neural Engine |
| Batch concurrency | One chunk at a time by default; qualify memory before raising it |
| Portable graphs | Core ML specification 8 / CoreML7; fixed 15-second model windows |
| Long recordings | FluidAudio's batch chunking; do not truncate to one model window |

The runtime version is the existing tested reference, **not a claim about
OpenWhispr's current pin**. Its iOS repository/version was not available during
preparation. Reconcile its `Package.resolved` before merging an app integration.
Do not add a second incompatible FluidAudio version to an existing application.

## Download and verify

Use the immutable Hugging Face revision
`43142dd1897f9ddadcd70173fcb5ff45c08aa951`:

| Profile | Download bytes | SHA-256 |
|---|---:|---|
| [Greedy](https://huggingface.co/oruk/orukeet/resolve/43142dd1897f9ddadcd70173fcb5ff45c08aa951/coreml/orukeet-r3-coreml-greedy.zip?download=true) | 466,579,943 | `beccdc6f18c4b10527a764f6e3ab12e3e11b969220c0cee175b3bb7eaa94290e` |
| [Baseline](https://huggingface.co/oruk/orukeet/resolve/43142dd1897f9ddadcd70173fcb5ff45c08aa951/coreml/orukeet-r3-coreml-baseline.zip?download=true) | 466,579,851 | `b2a6efc4ed3280c860f29b3e2e2ea242ade14c6482c94f1c8d3e8551d5edb626` |

For ordinary unconditioned batch decoding use **greedy**. Use **baseline** if the
application needs top-K outputs for language hints or vocabulary reranking.
The wrapper's batch API performs unconditioned decoding. Both archives contain
the same Orukeet encoder weights and full v3 vocabulary; they are not separate
English and multilingual models.

Verify the archive before extraction (run from the repository root):

```sh
python3 export/coreml/verify_bundle.py \
  --archive /path/to/orukeet-r3-coreml-greedy.zip --profile greedy
```

The verifier authenticates the archive against the pinned SHA-256, checks every
manifest payload, rejects unsafe/unlisted entries, and validates the vocabulary.
It reads the archive without extracting another weight copy. In the app, pin the
same archive hash in the download manager and verify it before extraction. Treat
`bundle.json` as integrity metadata, not a substitute for the trusted archive hash.

Each extracted archive has a single `orukeet-r3-coreml-<profile>/` root containing:

```text
Preprocessor.mlpackage/
Encoder.mlpackage/
Decoder.mlpackage/
JointDecisionv3.mlpackage/
parakeet_vocab.json
bundle.json
LICENSE-WEIGHTS
NOTICE.md
COREML-NOTICE.txt
```

Compile portable `.mlpackage` files on the destination device. Do not ship this
Mac's `.mlmodelc` cache to iOS. Keep model downloads and compilation outside the
recording/transcription timer, and retain the manifest and attribution alongside
the installed cache. Allow disk space for the download, extracted packages and
compiled cache during installation; the ZIP size is not peak disk or RAM usage.

## App integration

Add the repository's `export/coreml/benchmark` directory as a local Swift package
in Xcode and link the `OrukeetCoreML` library to the iOS target. The repository
root is a Python/native package, so adding its root URL as a Swift package will
not work. Preserve the package's pinned FluidAudio dependency while validating.

```swift
import Foundation
import OrukeetCoreML

// Installer context, off the main actor. `downloadedBundle` is the verified,
// extracted archive root. Use a NEW revision-specific destination directory.
try OrukeetLocalModels.compilePackages(
    from: downloadedBundle,
    to: installedCache
)

// Keep one engine alive between completed recordings.
let engine = OrukeetEngine(modelDirectory: installedCache)
try await engine.ensureLoaded()

// The existing capture/resampler must supply normalized 16 kHz mono PCM.
// Divide Int16 samples by 32768; merely casting Int16 to Float is not normalization.
let output = try await engine.transcribe(samples: mono16kSamples)
print(output.text)

// Release the model when switching away or handling memory pressure.
await engine.unload()
```

`compilePackages` publishes a complete new cache directory or fails without
leaving a partial installation. It refuses an existing destination. To upgrade,
install to a new revision-specific path and switch the app's saved selection
after success. Keep the previous installed version until the switch succeeds.

The engine rejects invalid input before loading models. It creates fresh decoder
state for each recording and rejects overlapping transcriptions with a busy
error. Serialize completed recordings in the app's queue; don't silently drop
one. Cancellation checks prevent returning a cancelled result, but an in-flight
Core ML operation is not guaranteed to stop immediately. Wait for completion
before starting the next request. There is no network request in the engine.

Newer FluidAudio versions provide `AsrModels.loadLocal(from:version:)`; upstream
documents [Orukeet local loading](https://github.com/FluidInference/FluidAudio/blob/5343241cd8a7576890e50925dec666bafc89d324/Documentation/Orukeet.md).
If OpenWhispr uses that API, follow its exact version's contract. The standard
NVIDIA download/load helpers can select Parakeet cache directories; do not use
them to load Orukeet or overwrite an existing NVIDIA model cache.

## Precision and English selection

FluidAudio's historical `.int8` setting refers to `Encoder.mlmodelc`, whose
published v3 weights actually use mixed 6-bit LUT palettes and FP16. Orukeet's
published preview matches that conversion. Current upstream also has a distinct
`.int8V2` / `Encoder_v2.mlmodelc` linear INT8 option. These are different artifact
contracts; confirm which OpenWhispr loads before describing them as identical.

A true symmetric INT8 Orukeet encoder exists as an experimental candidate.
Existing subset tests show mixed per-language results, including English and
wrong-script regressions. It must pass broader accuracy and target-device checks
before promotion. It is not part of these downloads, and its results do not
establish superiority to English Parakeet v2.

There is no separately trained/released v2-derived English Orukeet model in this
handoff. Keep OpenWhispr's English/multilingual selector and compare the current
v2, v3 and Orukeet on the same English recordings with the same normalization.
Published benchmark averages from different runtimes are not that comparison.

A [paired English16 diagnostic](../../../evidence/coreml-openwhispr-20260920/english16/README.md)
completed with the same FluidAudio version and identical audio: Parakeet v2 had
16 errors / 343 reference words (4.66% WER), versus Orukeet's 20 / 343 (5.83%).
These reused clips are too small a sample to establish general English accuracy,
but support retaining the separate English selection while qualifying Orukeet.

## Before enabling by default on iPhone

Run the [device qualification protocol](device-qualification.md) against the
actual app dependency version and supported devices. It covers successful and
interrupted installs, short and long recordings, repeated/cancelled requests,
multilingual/script regressions, peak memory, thermal behavior and English-v2
accuracy. Build success alone does not establish acceptable on-device memory,
latency, battery use or recognition quality.

Modified weights retain CC BY-SA 4.0 and NVIDIA attribution. The bundle includes
`LICENSE-WEIGHTS`, `NOTICE.md` and `COREML-NOTICE.txt`; distribute them with the
models. The integration code is MIT and FluidAudio is Apache-2.0.
