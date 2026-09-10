# Orukeet in OpenWhispr

The integration adds **Oruk → Orukeet** as a recommended local speech model. Orukeet uses the existing sherpa-onnx Parakeet TDT v3 path for dictation, meetings and file uploads. Its fitted Gabor filters are stored as ordinary convolution weights.

[Upstream PR #2085](https://github.com/OpenWhispr/openwhispr/pull/2085) · [Model card](MODEL_CARD.md) · [Paired recognition and runtime checks](APP_BENCHMARKS.md)

## Install and use

In the PR build, open **Settings → Speech to Text → Local → Oruk**, choose **Orukeet**, and click **Download**. The model also appears in local onboarding. Existing choices and upstream onboarding mode defaults are preserved. The current Oruk Signal logo identifies the organization; the model card links to `oruk/orukeet`.

The download is a public, immutable [486.7 MB ONNX INT8 archive](https://huggingface.co/oruk/orukeet/resolve/74673cf049c0c18f2572dab89f716b077461c2ea/onnx/sherpa-onnx-orukeet-v0.1.0-int8.tar.bz2) and occupies 670.7 MB after extraction. It contains the standard encoder, decoder, joiner and token files. The registry pins the Hugging Face revision and expected download length; archive and per-file hashes are recorded in the [manifest](../../evidence/onnx-r3-20260910/package-manifest.json).

## Build

Use Node.js 24 and the OpenWhispr PR branch:

```sh
git clone --branch codex/orukeet-default https://github.com/Oruk-AI/openwhispr.git
cd openwhispr
npm ci
npm run build:renderer
npm start
```

The branch includes the latest published release, v1.9.2, and subsequent upstream main changes. Standard OpenWhispr build and download hooks supply sherpa-onnx; the integration adds no native worker, inference dependency, or build step.

Fresh Windows installation uses the app’s bundled JavaScript bzip2 extractor for both model and runtime archives, avoiding a reproduced hang in the system tar’s external decompressor.

## Runtime contract

Orukeet uses OpenWhispr's existing offline transducer loader, 128-bin features, 16 kHz mono audio and four CPU threads capped by available cores. Model loading, normalization, segmentation, previews, cancellation and process reuse follow the same path as stock Parakeet. On macOS, the bundled runtime requires macOS 15.5 or later; the same capability guard applies to both models.

The model is a full-context recognizer. OpenWhispr supplies capture, endpointing and preview chunking, along with history and paste.

The internal model ID remains `orukeet-v0.1.0-q8` to preserve saved selections from earlier PR builds; its payload is now ONNX INT8. A previous GGUF file alone does not satisfy the required four-file installation check. Select Download once to install the ONNX package.

Source checkpoint SHA-256: `031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`.

Archive SHA-256: `c4ad85bbfb0835167c097dedcec0bb3f50cc468edfe7e690442e2221427b95bf`.

[Reproduce the export](../../export/onnx/README.md) · [License and attribution](../../NOTICE.md)

The integration is proposed in PR #2085. Orukeet remains available for general ASR through NeMo, sherpa-onnx and its native runtime.
