# r3 ONNX export and sherpa-onnx checks

The released Orukeet r3 checkpoint was converted to the standard sherpa-onnx
Parakeet TDT v3 layout on 10 September 2026. No checkpoint was retrained or
modified. The application payload contains self-contained INT8 encoder,
decoder, and joiner graphs, plus the unchanged token vocabulary.

## Source and conversion

- NeMo source: SHA-256 `031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`.
- Parameter count: 627,008,134.
- Exact fitted filters checked before export: 12,288 rows, or 110,592 coefficients.
- Source filters are ordinary nine-tap depthwise convolution weights.
- NeMo 3.0.0, PyTorch 2.12.0+cu132, ONNX 1.21.0, ONNX Runtime 1.27.0,
  Python 3.13.14; Linux x86-64, four CPU threads.
- Container: `nvcr.io/nvidia/nemo-speech:26.07.00`, image digest
  `sha256:b8b1c094f1bbd1a28acec4a742ae8984f420c021293dd68e7cadb21ac2963ef0`.
- Legacy ONNX export, opset 17; upstream quantization settings: unsigned INT8
  encoder, signed INT8 prediction network and joiner.

[Export receipt](export-receipt.json) records graph hashes, sizes, operators,
interfaces, source identity, and the exact software environment.

## Compatibility and numerical checks

All three INT8 graphs pass the ONNX checker. They retain the stock Parakeet
interface: 128 input features, 8× subsampling, two prediction layers of width
640, 8,192 tokens plus blank, and five TDT duration logits. They contain no
Orukeet-specific operator. As in the stock export, the quantized decoder uses
ONNX Runtime's `DynamicQuantizeLSTM` operator.

sherpa-onnx 1.13.4 selects TDT decoding by finding `tdt` in the encoder's `url`
metadata. The final URL is
`https://huggingface.co/oruk/orukeet#parakeet-tdt-v3`. This compatibility detail
was verified through the actual runtime; finalizing the URL changed no graph
node or tensor.

At two different input lengths, 257 and 481 feature frames, FP32 ONNX encoder
outputs match the source PyTorch module with maximum absolute errors of
`8.96e-7` and `1.84e-6`. Both precisions produce finite values and correct
dynamic output lengths. The receipt also records INT8 numerical differences;
those synthetic-feature checks are separate from recognition measurements.

## Actual decoding

[Linux sherpa smoke](linux-sherpa-smoke.json) uses sherpa-onnx **1.13.4** with
its existing CPU offline-transducer loader. FP32 and INT8 each decode English,
German, French, and Spanish speech; each returns an empty transcript for three
seconds of silence. Both precisions also decode two streams together.

The English, German, and French serial transcripts match between FP32 and
INT8. The Spanish serial transcripts differ by one pronoun. Batched INT8
English changes punctuation while preserving the words.
[Source NeMo transcripts](linux-nemo-smoke.json) record the same four files
through the original checkpoint. These are functional conversion checks;
the application comparison records recognition scores over its paired suite.

Audio comes from the [sherpa-onnx model test fixtures](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models):
`en.wav`, `de.wav`, `fr.wav`, and `es.wav`. Their content hashes are in the
decoding receipts. All timings in these receipts are individual CPU calls,
not an application latency benchmark.

## Reproduce

The exporter, smoke scripts, deterministic packager, and pinned upstream
conversion reference are in [export/onnx](../../export/onnx/README.md).
The [archive manifest](package-manifest.json) lists every payload file, compressed
size, and SHA-256. [Archive readback](archive-readback.json) independently
decompresses all seven files and checks their contents against that manifest.
A [Linux archive rebuild](archive-linux-reproduction.json) from the same
payload reproduces the macOS-built archive and manifest byte for byte.

The receipt's executed source hashes can be checked against
[the original export recipe](exporter-original.py.txt) and
[the recipe with final TDT metadata](exporter-finalized.py.txt). The current
exporter additionally resolves a relative `--fits` path before changing the
working directory. The recorded run used an absolute path; its graph files,
export receipt, and archive manifest remain unchanged.
