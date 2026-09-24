# Orukeet in OpenWhispr iOS

Use **one Orukeet r3 model for English and multilingual recordings**, with an
INT8 encoder and greedy TDT decoding. This preserves Chad's existing flow:
record 16 kHz mono PCM, then transcribe locally with FluidAudio and Core ML.
The deployment target is iOS 17; the Swift package also supports macOS 14.

## Add the package

Add `https://github.com/Oruk-AI/orukeet.git` in Xcode, select the integration preview
tag `v0.1.2-coreml.1`, and link **OrukeetCoreML** to the app target.
The repository root is a Swift package; no local checkout or converter is needed.
The tag pins reviewed SDK commit `6c37c587fabcef8b0e932584fbf22a79fcb9d89e`.

The package uses Oruk's small FluidAudio 0.15.5 backport of the
[buffer optimization](../../../export/coreml/runtime/README.md), including the
shared/strided-view correctness fix. If the app already directly depends on
FluidAudio, switch that package reference to the **same URL and version in
Orukeet's Package.swift**. Keep a single FluidAudio dependency in the graph;
`import FluidAudio` and the existing APIs remain unchanged.

## Run the complete iOS app

Open [Example/OrukeetExample.xcodeproj](Example/OrukeetExample.xcodeproj), select
the **OrukeetExample** scheme and an iPhone or simulator, then run. For a physical
iPhone, select your development team in Signing & Capabilities first. The app
downloads and installs the pinned model, prepares it, records microphone audio,
and transcribes after Stop. It also imports existing audio files. See the
[example instructions](Example/README.md) for the automated fixture test.

The example contains the complete recording and model lifecycle, including
microphone permission, cancellation, interruptions and background cleanup. It
is a reference app ready to run from this checkout. The direct OpenWhispr mobile
integration is tracked in [PR #2342](https://github.com/OpenWhispr/openwhispr/pull/2342).
This example remains a standalone reference for the SDK service shown below.

## Install once, warm once, reuse

Keep **one service** alive for successive recordings. `install()` handles the
download, SHA-256 and size verification, safe ZIP extraction, destination-device
Core ML compilation and atomic publication. Compiled models are keyed by model
identity, architecture and OS build. Each installation preserves vocabulary,
license notices and an authenticated copy of the portable ZIP. On an OS update,
the SDK recompiles from that local ZIP instead of downloading the model again.
Downloaded model storage is excluded from backups.

Call `prepare()` when selecting the model or opening the recorder, so loading
and the first prediction finish before the user stops recording:

```swift
import OrukeetCoreML

let transcriber = OrukeetTranscriber()
try await transcriber.install() // First use downloads; later calls reuse cache.
try await transcriber.prepare() // Idempotent until unload; discards warmup text.

// After recording stops: preserve Chad's existing 16 kHz mono Float PCM path.
let result = try await transcriber.transcribe(samples: mono16kSamples)
// Use result.text in the app's existing transcript handling.

// Or decode a completed CAF, WAV or M4A recording, including 44.1/48 kHz stereo:
let fileResult = try await transcriber.transcribe(fileURL: recordingURL)

// Release loaded models when leaving the recorder or under memory pressure.
await transcriber.unload()
```

`OrukeetAudio` converts mono/stereo files to normalized 16 kHz mono PCM off the
main actor. It drains the recording's final partial buffer, validates finite
samples and observes cancellation. Applications with an existing PCM recorder
can use `transcribe(samples:)` directly without another conversion.

`install(progress:)` reports the current phase and extraction fraction through
a `@Sendable` callback; dispatch UI updates to the main actor. Downloading and
compilation use indeterminate progress. `install(fromArchive:)` uses the same
verified installer for an archive the app already owns, without deleting it.
`installedDirectory()` checks the local receipt, sidecars and compiled file
inventory without network access or rehashing all compiled weights. After an OS
update it can reauthenticate, extract and compile the retained ZIP before
returning; keep this call off the main actor and show preparation UI if needed.
The result remains a single complete directory: app cleanup can keep that
directory and remove its obsolete siblings **after** successful recovery.
Failed or cancelled recompilation leaves the prior source available for retry.
Invalid current-OS installations are reported and never silently overwritten.

The retained ZIP adds **554,985,744 bytes** to installed storage (roughly 1.19 GB
including the compiled models; actual Core ML output varies by device/OS).
First-install peak space is the caller's ZIP plus the larger of
`extracted + compiled` and `retained ZIP + compiled`. Extracted packages are
removed before the ZIP is retained. An OS rebuild needs that same larger amount
of **additional free space**, while keeping the previous complete installation
until the new one is ready. SHA-256 uses a drained autorelease pool per 1 MiB
chunk, so Foundation does not retain a model-sized chain of read buffers.

Caches made with the old SDK have no portable source. They remain usable on
their original OS. Calling `install(fromArchive:)` while the original ZIP is
still available backfills source without recompiling. If both that ZIP and any
retained source are absent, the first migration requires a fresh download;
compiled models cannot reconstruct a portable `.mlpackage` archive.

`prepare()` and `transcribe()` run through the service actor. Await preparation
before submitting a recording, and serialize recordings through the app's
queue. The service rejects overlap with `busy`, creates fresh decoder state for
every request, discards cancelled results, and defers unload until active work
finishes. Cancel and await the active task before retrying. In-flight Core ML
compilation or prediction can finish before cancellation is seen. Preparation
and inference never download; only an explicit `install()` may access the network.

The samples API requires finite, normalized `[Float]` PCM at **16,000 Hz, one
channel**, with at least 300 ms of audio. Divide Int16 by 32768 before passing it
as Float. FluidAudio chunks long recordings automatically; pass the complete
recording. The v3 vocabulary and decoder apply to English too; there is no v2
model switch.

## Speed choices

- Greedy joint exports only the decisions needed by unconditioned transcription.
- Bulk tensor fill/copy replaces per-element Swift/Core ML calls while preserving
  view boundaries and overlapping-copy semantics.
- Loading, compilation and warmup happen once; warmed models remain resident.
- The default encoder/decoder/joint placement is CPU + Neural Engine; preprocessing
  runs on CPU. `encoderComputeUnits` permits app-side device profiling.
- Long recordings use up to **four concurrent chunks**, matching FluidAudio's
  default. Set `batchConcurrency: 2` or `1` if the app needs lower peak working
  memory. The performance record compares all three settings.

See [reproducible performance results](../../../evidence/coreml-openwhispr-20260920/performance/README.md).
Mac timings identify runtime improvements; they are not iPhone latency claims.

## Model identity and validation

The immutable download is **554,985,744 bytes**, with SHA-256
`24df9ff76f00f86f9ae1fd601cbbcab1d1eac98c7e8107de67444a7858d88b8b`.
It contains a true symmetric per-channel INT8 weight-quantized Orukeet encoder,
the original preprocessor, FP16 decoder/joint components and the complete 8,192-token v3 vocabulary. Activations
are not claimed to be INT8. This differs from FluidAudio's historical `.int8`
selection, which used mixed LUT6/FP16 encoder weights.

The archive retains its original experimental filename and provenance. The
September 24 full FLEURS evaluation attempted all **20,146 test recordings in
25 languages**, including recordings longer than 15 seconds. Fixed-reference
WER was **13.6403% INT8 / 14.2656% LUT6**; the latter includes two administrative-pause
timeouts retained in the primary score. See the
[qualification report](https://huggingface.co/oruk/orukeet/blob/b3421ca5ec4d3b0ad3c6d0bc58be4e2fbf5dc61f/coreml/QUALIFICATION-20260924.md)
for interruption controls, per-language results and methods. Slovenian outputs
still contain Cyrillic in **23/834 INT8 and 20/834 LUT6** cases. This evaluation
covers the SDK Engine/Audio path; it does not by itself qualify OpenWhispr's
separate `AsrManager` integration. Full evaluation is complete; recognition
acceptance and physical iPhone qualification remain open. There is no separate
Parakeet model in this SDK integration.

[Validation evidence](../../../evidence/coreml-openwhispr-20260920/README.md)
records the build, runtime and model checks. The CI compiles portable models and
runs multilingual, long-recording and lifecycle checks inside iOS Simulator.
It also builds the complete example app for iOS 17 and drives installation,
preparation, repeated file transcription and a cold offline cache relaunch
through its actual UI. Unit checks cover installer rollback, archive validation,
audio conversion and service cancellation without model downloads.
[Physical-device measurements](device-qualification.md) remain a deployment
follow-up for the app's supported iPhones, including latency, RAM and thermals.

For a development-machine artifact audit without extracting weights:

```sh
python3 export/coreml/verify_bundle.py --profile int8 --archive /path/to/model.zip
```

Ship `LICENSE-WEIGHTS`, `NOTICE.md` and `COREML-NOTICE.txt` with the model.
Modified weights retain CC BY-SA 4.0 and upstream NVIDIA attribution;
integration code is MIT and FluidAudio is Apache-2.0.
