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

## Install once, warm once, reuse

Use the app's existing download and ZIP extraction machinery with
`OrukeetBundle.int8.url`. The descriptor pins the archive size and SHA-256.
Verification, extraction and Core ML compilation belong in the installation
worker, outside the main actor and transcription timer:

```swift
import OrukeetCoreML

let bundle = OrukeetBundle.int8
try bundle.verifyArchive(at: downloadedZIP)
// Existing ZIP extractor: extract downloadedZIP into extractionDirectory.
let source = extractionDirectory.appendingPathComponent(bundle.archiveRoot)
try OrukeetLocalModels.compilePackages(from: source, to: revisionCache)
```

Use a new revision-specific cache directory. Installation publishes all four
compiled components atomically and preserves vocabulary, identity and license
notices. It refuses to overwrite an existing cache. Compile portable
`.mlpackage` files on the destination device; compiled caches are OS-specific.

Keep **one engine** alive for successive recordings. Start preparation when the
model is selected or the recorder opens, so graph loading and the first Core ML
prediction finish before the user stops recording:

```swift
let engine = OrukeetEngine(modelDirectory: revisionCache)
try await engine.prepare() // Idempotent until unload; warmup text is discarded.

// Recording has stopped; the existing capture path supplies Float PCM.
let result = try await engine.transcribe(samples: mono16kSamples)
insertTranscript(result.text)

// Only when switching model or responding to memory pressure:
await engine.unload()
```

`prepare()` and `transcribe()` run on the engine actor. Await preparation before
submitting a recording, and serialize completed recordings through the app's
queue. The engine rejects overlap with `busy`, creates fresh decoder state for
every request, discards cancelled results, and defers unload until active work
finishes. In-flight Core ML predictions can finish before cancellation is seen.
Inference has no download or network fallback.

Input must be finite, normalized `[Float]` PCM at **16,000 Hz, one channel**, with
at least 300 ms of audio. Divide Int16 by 32768 before passing it as Float.
FluidAudio chunks long recordings automatically; pass the complete recording.
The v3 vocabulary and decoder apply to English too; there is no v2 model switch.

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
[Physical-device measurements](device-qualification.md) remain a deployment
follow-up for the app's supported iPhones, including latency, RAM and thermals.

For a development-machine artifact audit without extracting weights:

```sh
python3 export/coreml/verify_bundle.py --profile int8 --archive /path/to/model.zip
```

Ship `LICENSE-WEIGHTS`, `NOTICE.md` and `COREML-NOTICE.txt` with the model.
Modified weights retain CC BY-SA 4.0 and upstream NVIDIA attribution;
integration code is MIT and FluidAudio is Apache-2.0.
