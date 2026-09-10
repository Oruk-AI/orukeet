# Exact depthwise execution optimization

This deployment graph comes from the same r3 checkpoint,
`031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`.
It changes how 24 depthwise convolutions execute. Their quantized tensors,
activation quantization, zero points and dequantization scales are retained.
Decoder, joiner, token files, model metadata and graph interfaces are unchanged.

| Encoder | SHA-256 |
| --- | --- |
| Original ONNX export | `d10711f1b8f3a516e2d7a93adb219caf8aba2b55295305db92f3e80e78c1499a` |
| Optimized ONNX graph | `7b55f2a504a20a8e462899f5befd45f4a1784948d76ed0127902d9cf39405487` |

The optimized encoder is 653,182,378 bytes. It retains the original integer
initializers and adds 24 centered FP32 tensors. No training or new inference
dependency is involved.

## Arithmetic

For each channel, the original integer convolution computes

\[
s_t=\sum_{k=0}^{8}(q_{x,t+k}-z_x)(q_{w,k}-z_w).
\]

Both quantized operands and zero points are unsigned eight-bit integers.
Each centered product has magnitude at most 65,025, and every partial sum has
magnitude at most 585,225. These integers are exactly representable in FP32,
whose consecutive integer range extends to 16,777,216. The replacement centers
the operands, evaluates a standard FP32 `Conv`, and casts its output to INT32.
The existing dequantization then proceeds unchanged.

ONNX Runtime 1.27.0's [integer convolution implementation](https://github.com/microsoft/onnxruntime/blob/v1.27.0/onnxruntime/core/providers/cpu/quantization/conv_integer.cc)
performs a separate grouped matrix multiplication for each of the 1,024
channels. The replacement uses the runtime's standard floating-point
convolution implementation. All operators already exist in the shipped runtime.

## Verification

[Operator and encoder validation](operator-and-encoder-validation.json) records
72 isolated comparisons: every replaced operator, three dynamic shapes,
batches of one and two, activation zero points 0, 127 and 255, and inputs
including both eight-bit extremes. Every INT32 output is bit-identical.
Full encoder outputs and lengths are also bit-identical at 257, 481 and 1,501
feature frames, and at batch size two. Maximum absolute error is zero.

The [executed experiment script](rewrite-and-profile-executed.py.txt) is retained
byte-for-byte. [Static identity checks](static-identity.json) verify the unchanged
metadata, interfaces and original source. The separate
[optimization receipt](optimization-receipt.json) identifies the clean optimizer,
the original export receipt and these validation results by SHA-256.
The [Linux sherpa-onnx comparison](linux-speech-equality.json) also produces
identical transcripts for English, German, French and Spanish speech, blank
output for silence, and identical results from two-stream batch decoding.

| Encoder input | Original median | Optimized median | Speed ratio |
| --- | ---: | ---: | ---: |
| 1 × 257 frames | 257.73 ms | 173.02 ms | 1.49× |
| 1 × 481 frames | 429.69 ms | 292.60 ms | 1.47× |
| 1 × 1,501 frames | 1,221.41 ms | 830.57 ms | 1.47× |
| 2 × 257 frames | 477.90 ms | 316.67 ms | 1.51× |

These are encoder forward-pass measurements on an Intel Xeon 2.20 GHz VM,
Linux x86-64, with ONNX Runtime 1.27.0 and four inference threads. Each case
uses fixed synthetic features, two warmups and eight alternating-order pairs.
The container permits eight CPU equivalents for the two four-thread sessions.
Application recognition and file-transcription measurements are recorded
separately in the [OpenWhispr integration](../../integrations/openwhispr/APP_BENCHMARKS.md).

## Reproduce the deployment graph

Use the original four-file ONNX export and the export environment documented in
[export/onnx](../../export/onnx/README.md). The validated conversion environment
provides ONNX 1.21.0 and NumPy 2.4.4. From the repository root:

```sh
python export/onnx/optimize_for_sherpa.py \
  --model-dir /path/to/original-onnx \
  --output /path/to/optimized/sherpa-onnx-orukeet-v0.1.0-int8 \
  --export-receipt evidence/onnx-r3-20260910/export-receipt.json \
  --validation-receipt evidence/speed20260910/operator-and-encoder-validation.json
```

The command checks the original encoder hash, reproduces the optimized graph,
checks its expected hash and ONNX validity, copies the remaining model files,
and writes `optimization-receipt.json`. It leaves the original export receipt
unchanged.

To reproduce the release archive, use the full original package so the weight
license and vocabulary are present, then install the updated attribution and
run the deterministic packager:

```sh
cp evidence/speed20260910/NOTICE.md \
  /path/to/optimized/sherpa-onnx-orukeet-v0.1.0-int8/NOTICE.md
python export/onnx/package_release.py \
  --model-dir /path/to/optimized/sherpa-onnx-orukeet-v0.1.0-int8 \
  --export-receipt evidence/onnx-r3-20260910/export-receipt.json \
  --optimization-receipt /path/to/optimized/sherpa-onnx-orukeet-v0.1.0-int8/optimization-receipt.json \
  --archive /path/to/sherpa-onnx-orukeet-v0.1.0-int8.tar.bz2 \
  --manifest /path/to/package-manifest.json
```

The archive retains the standard seven-file package layout. Its manifest
records separate hashes for the original export receipt and the optimization
receipt, together with every packaged file.

The [release manifest](package-manifest.json) identifies the 486,807,585-byte
archive, which expands to 671,619,800 bytes. Archive SHA-256:
`f9191f30178cc9122ce2f023bf9fefafc822028307b0efa4caff645ba3fe8d0a`.
The [full archive readback](archive-readback.json) verifies all seven file hashes,
the single root directory and deterministic archive metadata.
