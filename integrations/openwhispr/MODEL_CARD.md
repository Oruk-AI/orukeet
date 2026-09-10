<!-- orukeet-brand:start -->
<p><img src="../../docs/assets/team/oruk.png" alt="Oruk AI" width="184"></p>
<!-- orukeet-brand:end -->

# Orukeet in OpenWhispr

<!-- orukeet-team:start -->
<p>
Nathan Roll<sup>1,2</sup> · Irene Yi<sup>1,2</sup> · Büşra Marşan<sup>1,2</sup><br>
Vianney Grenez<sup>1</sup> · Gabriel Stein<sup>4</sup> · Momcilo Mrkaic<sup>5</sup><br>
Pavle Padjin<sup>5</sup> · Vladimir Zeljkovic<sup>5</sup> · Calbert Graham<sup>1,3</sup>
</p>

<p><strong><sup>1</sup> Oruk AI</strong></p>
<table>
<tr>
<td align="center" valign="middle"><img src="../../docs/assets/team/stanford.png" alt="Stanford University" width="144"><br><sup>2</sup> Stanford University</td>
<td align="center" valign="middle"><img src="../../docs/assets/team/cambridge.png" alt="University of Cambridge" width="144"><br><sup>3</sup> University of Cambridge</td>
<td align="center" valign="middle"><img src="../../docs/assets/team/openwhispr.png" alt="OpenWhispr" width="40"><br><sup>4</sup> OpenWhispr</td>
<td align="center" valign="middle"><img src="../../docs/assets/team/hoid.png" alt="Hoid" width="76"><br><sup>5</sup> Hoid</td>
</tr>
</table>
<!-- orukeet-team:end -->

Orukeet is Oruk's 25-language speech recognizer, adapted from NVIDIA Parakeet TDT 0.6B v3 with 12,288 fitted, frozen Gabor kernels. The OpenWhispr PR uses the ONNX INT8 export of the r3 release checkpoint for dictation, file uploads and meetings.

Select **Local → Oruk → Orukeet**, then **Download**. Orukeet is marked Recommended and carries the current Oruk Signal logo. Existing model choices and upstream mode defaults are preserved.

[General ASR model card](https://huggingface.co/oruk/orukeet) · [Technical report](../../output/pdf/orukeet-technical-report.pdf) · [Integration and installation](README.md)

## Recognition and responsiveness

| Measurement | Parakeet TDT v3 INT8 | Orukeet r3 INT8 |
| --- | ---: | ---: |
| WER, 640 English clips across ten corpora | 11.93% | 11.40% |
| Warm file transcription, median, paired 160-clip subset | 428 ms | 390 ms |
| Warm file transcription, 95th percentile, same subset | 1097 ms | 1026 ms |

Both models run through the same current OpenWhispr sherpa-onnx CPU path with four threads on Apple M5 Max. Calls include normalization, 15-second segmentation, WebSocket exchange and recognition. Every selected clip contributes to the WER, including empty outputs.

[All ten corpus scores, timing protocol and provenance](APP_BENCHMARKS.md)

## Runtime and model identity

The standard encoder, decoder, joiner and token files load through OpenWhispr's existing Parakeet worker. Fitted Gabor kernels are ordinary convolution weights. The optimized encoder uses equivalent FP32 arithmetic for 24 quantized depthwise convolutions, preserving their outputs exactly with existing ONNX Runtime operators. The public ONNX archive is 486,807,585 bytes, pinned to an immutable Hugging Face revision; its [manifest](../../evidence/speed20260910/package-manifest.json) records the payload hashes. The underlying NeMo source has SHA-256 `031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`.

Recognition runs locally after download. OpenWhispr supplies capture, endpointing, preview chunking, history and paste. The bundled sherpa runtime supports the app's Windows x64, Linux x64 and macOS arm64/x64 targets; macOS requires version 15.5 or later.

The internal ID `orukeet-v0.1.0-q8` preserves earlier saved choices and now resolves to ONNX INT8. Earlier GGUF-only installations need the new Download step. Users of the previous ONNX package from this PR should delete Orukeet and download it again to get the optimized export. Standard OpenWhispr build hooks supply the runtime.

Supported languages: Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Spanish, Swedish and Ukrainian.

Code is MIT. Model weights and fitted kernels are CC BY-SA 4.0, with NVIDIA's attribution retained. [License and data provenance](../../NOTICE.md).

<!-- orukeet-citation:start -->
## Citation

```bibtex
@techreport{roll2026orukeet,
  title = {{Orukeet}: Multilingual {ASR} with Frozen {Gabor} Kernels},
  author = {Roll, Nathan and
            Yi, Irene and
            Mar{\c{s}}an, B{\"u}{\c{s}}ra and
            Grenez, Vianney and
            Stein, Gabriel and
            Mrkaic, Momcilo and
            Padjin, Pavle and
            Zeljkovic, Vladimir and
            Graham, Calbert},
  institution = {Oruk AI},
  year = {2026},
  type = {Technical report},
  url = {https://github.com/Oruk-AI/orukeet/blob/main/output/pdf/orukeet-technical-report.pdf}
}
```

[Download BibTeX](../../CITATION.bib) · [Citation metadata](../../CITATION.cff)
<!-- orukeet-citation:end -->
