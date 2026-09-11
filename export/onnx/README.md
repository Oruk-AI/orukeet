# Orukeet for sherpa-onnx

This exporter converts the released r3 Orukeet checkpoint into the same offline
encoder/decoder/joiner interface used by sherpa-onnx for Parakeet TDT 0.6B v3.
Orukeet's Gabor filters are stored as ordinary `Conv1d` weights. The export does
not require a custom frontend, an Orukeet-specific ONNX operator, or a new application
runtime.

The checkpoint is pinned to SHA-256
`031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`.
Before conversion, the script checks all 12,288 selected filters against the
published fitted functions. It retains the original 128-bin features, 8×
subsampling, 8,192-token vocabulary plus blank, two-layer prediction network,
and TDT durations `[0, 1, 2, 3, 4]`.

## Export

Run the commands below from the root of the Orukeet GitHub checkout, inside
NVIDIA's `nemo-speech:26.07.00` container with ONNX Runtime 1.27.0. Its validated
image digest is `sha256:b8b1c094f1bbd1a28acec4a742ae8984f420c021293dd68e7cadb21ac2963ef0`.
The container already provides NeMo, PyTorch, and ONNX. Export itself uses
four CPU threads. Allow space for the source, temporary extracted checkpoint,
FP32 graphs, and INT8 graphs.

```sh
python -m pip install onnxruntime==1.27.0 sherpa-onnx==1.13.4
python export/onnx/export_orukeet.py \
  --source /path/to/orukeet-v0.1.0.nemo \
  --fits training/gabor_half/fits/fits.json.gz \
  --output /path/to/onnx-r3 \
  --threads 4
```

The code follows the [upstream sherpa Parakeet TDT v3 exporter](https://github.com/k2-fsa/sherpa-onnx/blob/11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf/scripts/nemo/parakeet-tdt-0.6b-v3/export_onnx.py).
It uses NeMo's legacy ONNX exporter with opset 17, then dynamic INT8
quantization: the encoder through `quantize_encoder_sme.py` (symmetric, signed,
per-output-channel weights emitted as `com.microsoft.DynamicQuantizeMatMul`
nodes, each layer's Q/K/V projections merged into one GEMM; depthwise and 2-D
subsampling convolutions stay in FP32), the decoder and joiner through
`quantize_dynamic` with signed INT8. The released NeMo parameters are never
modified.

The encoder form matters for ONNX Runtime's CPU provider: only symmetric int8
weights in the fused `DynamicQuantizeMatMul` op are routed to the KleidiAI SME2
kernels on Apple M4/M5 and to the I8MM kernels on M2/M3; the previous asymmetric
unsigned export always fell back to the older NEON dot-product path. Same 8-bit
weights, same dynamic activation quantization, same operator set on every
platform (the op is a standard CPU contrib op on x86 as well). Encoder outputs at two different sequence
lengths are checked against PyTorch before the receipt is marked successful.
The original upstream script is retained as
`upstream_export_onnx.reference.txt`, with its Apache-2.0 license, for comparison
with this complete export command.

Use a fresh output directory for each full export. The command writes three
FP32 graphs, external encoder weight files, three self-contained INT8 graphs,
token files, and validation receipts. Keep every external encoder weight file
beside `encoder.onnx` when using the FP32 graph. The release archive packages
the three self-contained INT8 graphs.

## Optimize and package

With `quantize_encoder_sme.py` the depthwise convolutions are never quantized,
so the `optimize_for_sherpa.py` rewrite below is a no-op for that export and
its hash pins (and those in `package_release.py`) must be regenerated from the
new export receipt.

The previous release rewrites 24 quantized depthwise convolutions to equivalent
centered FP32 arithmetic and casts each result back to INT32 before the existing
dequantization. Their nine-term integer sums are exactly representable in FP32.
The remaining encoder operations, decoder, joiner, tokens and metadata stay
unchanged. These standard operators already exist in the Parakeet runtime.

```sh
python export/onnx/optimize_for_sherpa.py \
  --model-dir /path/to/onnx-r3 \
  --output /path/to/optimized/sherpa-onnx-orukeet-v0.1.0-int8 \
  --export-receipt evidence/onnx-r3-20260910/export-receipt.json \
  --validation-receipt evidence/speed20260910/operator-and-encoder-validation.json
cp LICENSE-WEIGHTS \
  /path/to/optimized/sherpa-onnx-orukeet-v0.1.0-int8/LICENSE-WEIGHTS
cp evidence/speed20260910/NOTICE.md \
  /path/to/optimized/sherpa-onnx-orukeet-v0.1.0-int8/NOTICE.md
python export/onnx/package_release.py \
  --model-dir /path/to/optimized/sherpa-onnx-orukeet-v0.1.0-int8 \
  --export-receipt evidence/onnx-r3-20260910/export-receipt.json \
  --optimization-receipt /path/to/optimized/sherpa-onnx-orukeet-v0.1.0-int8/optimization-receipt.json \
  --archive /path/to/sherpa-onnx-orukeet-v0.1.0-int8.tar.bz2 \
  --manifest /path/to/package-manifest.json
```

The optimizer verifies the original encoder and reproduces the validated graph
hash. The packager preserves the original export receipt and records the separate
optimization receipt. [Arithmetic, exactness checks and benchmark receipts](../../evidence/speed20260910/README.md)
describe the optimized export. Its 640 application transcripts match the previous
export exactly.

## Application layout

```text
sherpa-onnx-orukeet-v0.1.0-int8/
  encoder.int8.onnx
  decoder.int8.onnx
  joiner.int8.onnx
  tokens.txt
  bpe.vocab
  LICENSE-WEIGHTS
  NOTICE.md
```

Use the ordinary sherpa offline transducer loader with `model_type="nemo_transducer"`,
`sample_rate=16000`, and `feature_dim=128`. The encoder metadata supplies
`normalize_type=per_feature`. sherpa-onnx 1.13.4 selects TDT decoding when the
encoder metadata URL contains `tdt`, so the export uses
`https://huggingface.co/oruk/orukeet#parakeet-tdt-v3`. The joiner retains five
duration logits after its 8,193 token/blank logits. The INT8 decoder uses the
same ONNX Runtime `DynamicQuantizeLSTM` contribution operator as stock Parakeet.
Real-time applications continue to use their existing VAD and chunking
path. The graph does not convert this full-context model into a causal streaming
encoder.

The INT8 graphs are a deployment conversion of r3.
[Conversion receipts](../../evidence/onnx-r3-20260910/README.md) and
[paired ONNX application measurements](../../integrations/openwhispr/README.md)
record the exported artifact and its application behavior.
