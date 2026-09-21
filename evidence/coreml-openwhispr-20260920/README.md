# Orukeet iOS batch integration evidence

The current integration uses **Orukeet for English and all supported languages**,
with an INT8 encoder, greedy decoding and the reviewed FluidAudio buffer fix.
[App integration](../../integrations/openwhispr/ios/README.md) describes the
root Swift package, install/verify/compile path and persistent warmed engine.

## Current draft

- [INT8 archive verification](int8-archive-verification.json): immutable existing
  portable archive, 554,985,744 bytes, SHA-256
  `24df9ff76f00f86f9ae1fd601cbbcab1d1eac98c7e8107de67444a7858d88b8b`.
  All 22 payload files match; no compiled cache is distributed.
  [Upload receipt](int8-publication.json) verifies the remote LFS size/hash at
  revision `419d7f79e290127e202a0f610509868d314743eb`; the asset is staged in a
  [Hugging Face draft](https://huggingface.co/oruk/orukeet/discussions/2).
- [INT8 graph contract](int8-model-contract.json): component descriptions,
  graph hashes, specification versions and encoder quantization operations.
- [Pinned runtime and safety checks](../../export/coreml/runtime/README.md):
  narrow FluidAudio 0.15.5 backport, view-safe bulk tensor operations, source
  patch, native array regression harness and focused XCTest checks.
- [Release performance measurements](performance/README.md): original versus
  optimized runtime on identical Orukeet weights/audio, including long inputs
  and chunk-concurrency comparisons. Host measurements are not phone timings.
- `.github/workflows/coreml-ios.yml` runs root-package unit tests, an arm64
  iOS 17 build, and actual portable INT8 model compilation/inference inside
  iOS 18.5 Simulator. The runtime step requires a JSON receipt proving all
  language, long-recording and lifecycle checks executed.

Model assets are fetched only by ephemeral hosted CI for simulator validation.
The local measurements reuse existing caches in place. No training checkpoint
was created or downloaded on the Mac and no remote GPU was accessed.

## Historical baseline evidence

The following files predate the INT8/runtime change and describe the earlier
LUT6/FP16 greedy bundle and unmodified FluidAudio 0.15.5. Their original hashes,
versions and results remain intact; they are not relabeled as new validation:

- [Archive receipt](archive-verification.json) and [graph contract](model-contract.json).
- [Host smoke](engine-smoke.json) and [reproducer](host-smoke/): repeated short
  recordings, natural 29.95075-second speech, first-window comparison, unload
  and reload. Full long text had 113 words versus 60 for the first 15 seconds.
- [iOS Simulator receipt](simulator-runtime.json): four languages and nine
  transcriptions; 33-second text continued beyond the first model window;
  repeated English text remained identical across multilingual/long/reload calls.
- [CI compilation receipt](ci-validation.json).

Earlier conversion research remains accessible at the immutable
[TapTalk source revision](https://github.com/Oruk-AI/orukeet/tree/852c3e355f20a111a2ee39f76677bc0ba65147bf/evidence/coreml-taptalk-20260915).

## Reproduce

```sh
python3 -m unittest discover -s export/coreml -p 'test_verify_bundle.py' -v
swift test --configuration release
```

[Simulator instructions](../../integrations/openwhispr/ios/simulator-validation.md)
describe the hosted runtime job. The separate optional host test uses existing
compiled models through `ORUKEET_TEST_MODELS` and a 16 kHz mono WAV through
`ORUKEET_TEST_AUDIO`; it never downloads models.

This PR prepares a reusable integration package. The unavailable OpenWhispr
mobile target and physical iPhones were not built/tested here. Phone latency,
RAM/jetsam, thermals and battery measurements are listed in the
[device follow-up](../../integrations/openwhispr/ios/device-qualification.md).
