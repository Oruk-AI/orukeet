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

The library builds with this Mac's Command Line Tools. Since the local toolchain
has neither the iOS SDK nor Testing/XCTest, validation also ran on an Xcode
runner. [CI run 35567347963](https://github.com/Oruk-AI/orukeet/actions/runs/35567347963)
passed at commit `13db0b8f53907a46bae175e5aa5f9b427054b35c`:

- Six Python bundle-verifier tests passed.
- Thirteen Swift lifecycle/installation tests passed; the opt-in model test was
  skipped because CI had no model files. Swift Testing's summary counts 14
  discovered tests, including that skipped test.
- Release compilation of `OrukeetCoreML` for `arm64-apple-ios17.0` succeeded with
  Xcode 16.4, Swift 6.1.2 and the iPhoneOS 18.5 SDK, without code signing.

The ordinary repository package checks also passed. A library build does not
compile OpenWhispr's unavailable app target or execute the model on an iPhone.

The [local real-engine smoke](engine-smoke.json) passed six calls against the
existing greedy cache on macOS 26.4.1/arm64: short speech, repeat short, natural
29.95075-second speech, short after long, the long recording's first 15 seconds,
and short after unload/reload. Short text was identical across all lifecycle
checks. Full long output had 113 words versus 60 for the first 15 seconds, with
matching opening words and additional ending content. This checks execution
past the first model window; it is not an accuracy score.

That debug-build smoke measured 88.85 ms for 5.72 seconds of audio and 318.86 ms
for 29.95075 seconds, excluding model load. Initial load took 18.11 seconds,
reload took 91.21 ms, and whole-process peak RSS was 529,039,360 bytes. These are
single-run diagnostics, not release latency or memory budgets. No claim about
an iPhone follows from them. The retained source is in [host-smoke](host-smoke/).

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

No physical iPhone, iOS model execution, app build, iPhone memory/jetsam measurement,
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
