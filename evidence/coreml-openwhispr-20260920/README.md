# OpenWhispr iOS integration candidate

This change adapts the existing Orukeet Core ML preview to OpenWhispr's on-device
16 kHz mono, record-then-transcribe flow. It does not change the published model
weights or establish physical-iPhone performance. The reference runtime remains
FluidAudio 0.15.5 (`19600a485baa4998812e4654b70d2bab8f2c9949`).

## Completed artifact checks

- The existing greedy ZIP was read in place and authenticated against its pinned
  SHA-256 `beccdc6f18c4b10527a764f6e3ab12e3e11b969220c0cee175b3bb7eaa94290e`.
  All 18 payload files matched their recorded sizes and hashes. The vocabulary
  contains exactly the canonical IDs 0–8191, and the portable bundle includes
  all four graphs plus attribution. [Verification receipt](archive-verification.json).
- Graph protobuf inspection found specification version 8 and the `CoreML7`
  operation set in all four components. Input/output names, types, shapes and
  graph hashes are preserved in [model-contract.json](model-contract.json).
  These metadata are compatible with an iOS 17 deployment target; actual
  compilation and execution are separate checks.
- Six Python verifier regressions pass: valid bundle, changed weight payload,
  unexpected payload, missing required license/component, wrong source identity
  and unsafe path handling. The verifier never extracts weight files.

## Implementation and validation status

The Swift package declares iOS 17 and macOS 14. The engine validates finite
16 kHz mono input of at least 300 ms before loading a model. It uses fresh decoder
state per recording, defaults to one parallel batch chunk, rejects overlapping
calls, discards cancelled results and defers unload until an active call finishes.
Installation compiles into a sibling staging directory and moves a complete
cache into a new revision-specific destination; failure preserves prior installs.

The library builds with this Mac's Command Line Tools. The local toolchain has
neither the iOS SDK nor the Testing/XCTest modules, so the canonical Swift test
suite and iOS build are delegated to the committed Xcode CI workflow. Their
results must be checked before accepting this candidate; no CI success is
asserted by this initial record.

The opt-in real-model regression uses existing models and audio, without a
download. Its environment variables are `ORUKEET_TEST_MODELS` (compiled model
directory) and `ORUKEET_TEST_AUDIO` (a speech recording, mono 16 kHz WAV, at most
15 seconds). It checks repeated independent recordings and a concatenated
input longer than 30 seconds. That long input checks chunk execution and state
reset, not natural long-form recognition quality.

```sh
python3 -m unittest discover -s export/coreml -p 'test_verify_bundle.py' -v
swift test --package-path export/coreml/benchmark --configuration release
```

## Qualification limits

No physical iPhone, iOS model execution, app build, memory/jetsam measurement,
thermal/battery measurement, matched English-v2 accuracy test or full 25-language
accuracy qualification was completed by the checks above. OpenWhispr's mobile
repository and its exact FluidAudio version were not available. Retain the
existing English-v2 selection and expose Orukeet as an optional multilingual
candidate until the [device protocol](../../integrations/openwhispr/ios/device-qualification.md)
passes against the actual app.

The published encoder is LUT6/FP16 despite FluidAudio's historical `.int8`
selection name. It is not the newer `.int8V2` linear INT8 artifact. The existing
[TapTalk evidence](../coreml-taptalk-20260915/README.md) remains the source of prior
Mac accuracy and performance results; none is relabeled as an iPhone result.

No new training checkpoint was downloaded, created or copied on the Mac during
this integration work. No remote GPU was accessed.
