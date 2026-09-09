# Orukeet in OpenWhispr

The integration adds **Oruk → Orukeet** as the recommended local model for new profiles. Existing model preferences are preserved. Dictation, meetings and uploads share the final Orukeet Q8 checkpoint.

[Upstream PR #2085](https://github.com/OpenWhispr/openwhispr/pull/2085) · [Model card](MODEL_CARD.md) · [Paired recognition and latency](APP_BENCHMARKS.md) · [OpenWhispr](https://github.com/OpenWhispr/openwhispr)

The work began from the latest released OpenWhispr, v1.9.2, and is rebased onto main for the upstream PR. The native worker automatically selects Metal on Apple silicon, CUDA on NVIDIA, Vulkan on supported AMD/Intel graphics, or CPU fallback. SDK binaries are pinned and checksum verified. The public model is downloaded directly from Hugging Face without an account.

## Build

Use Node.js 24 and the OpenWhispr PR branch. From that checkout:

```sh
npm ci
npm run build:orukeet
npm run build:renderer
npm start
```

The native build requires CMake and a C++17 compiler. The regular OpenWhispr build hooks bundle the required native libraries. `OPENWHISPR_ORUKEET_DEVICE=cpu` forces CPU inference for diagnosis; `metal`, `cuda` and `vulkan` select those backends explicitly. Automatic selection is the default.

Model: `orukeet-v0.1.0-q8`. SHA-256: `93ce19c6d8244acbfea980eeaf970531d4f216171578ef8e041dcc2d070a45bd`.

The integration is a proposed change to OpenWhispr; the model package also serves general ASR independently.
