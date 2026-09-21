# Orukeet in OpenWhispr iOS

Use **one Orukeet r3 model for English and multilingual recordings**, with an
INT8 encoder and greedy TDT decoding. This preserves Chad's existing flow:
record 16 kHz mono PCM, then transcribe locally with FluidAudio and Core ML.
The deployment target is iOS 17; the Swift package also supports macOS 14.

## Add the package

Add `https://github.com/Oruk-AI/orukeet.git` in Xcode, select the draft branch
`codex/openwhispr-ios-20260920`, and link **OrukeetCoreML** to the app target.
The repository root is a Swift package; no local checkout or converter is needed.
Pin the reviewed commit when adopting the draft.

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
is a reference app ready to run from this checkout. OpenWhispr's mobile source
was not available for a direct patch; its existing UI calls the same service
shown below.

## Install once, warm once, reuse

Keep **one service** alive for successive recordings. `install()` handles the
download, SHA-256 and size verification, safe ZIP extraction, destination-device
Core ML compilation and atomic publication. The cache preserves vocabulary and
license notices and is keyed by model identity, architecture and OS build.
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
inventory without network access or rehashing all compiled weights. Invalid
existing installations are reported and never silently overwritten.

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

The archive retains its original experimental filename and provenance. Its
400-clip, 25-language diagnostic yielded 1,072 errors / 7,650 fixed reference
words versus 1,112 for the earlier Orukeet LUT6 export. Individual clips can
regress, including a documented Slovenian script error; this is not a full
accuracy benchmark. There is no separate Parakeet model in the app integration.

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
