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

Orukeet is Oruk's 25-language speech recognizer, adapted from NVIDIA Parakeet TDT 0.6B v3 with 12,288 fitted, frozen Gabor kernels. The OpenWhispr integration uses the Q8 export of the r3 release checkpoint for dictation, file uploads and meetings.

Fresh profiles select **Oruk → Orukeet**. Installation verifies the 714,456,704-byte model against SHA-256 `93ce19c6d8244acbfea980eeaf970531d4f216171578ef8e041dcc2d070a45bd`. A checkpoint-specific filename refreshes earlier Orukeet caches. The underlying NeMo source has SHA-256 `031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`.

[General ASR model card](https://huggingface.co/oruk/orukeet) · [Technical report](../../output/pdf/orukeet-technical-report.pdf) · [Integration](README.md)

## Recognition and responsiveness

| Measurement | Orukeet r3 | Stock Parakeet TDT v3 |
| --- | ---: | ---: |
| WER, 640 English clips across ten corpora | 6.89% | 11.93% |
| First live preview, median | 1.55 s | 1.64 s |
| Preview processing, median | 40 ms | 131 ms |
| Stop to saved transcript, median | 115 ms | 945 ms |

The app comparison runs both backends inside the integration against OpenWhispr v1.9.2. Orukeet uses Q8 on Metal; stock Parakeet uses the release's INT8 ONNX models and sherpa-onnx CPU worker with four threads. The paired recognition test scores every selected clip, including empty outputs. Live timings use three warm recordings per model through the production renderer on Apple M5 Max, with the app's 1.5-second preview timer.

[Every corpus score](APP_BENCHMARKS.md) · [Benchmark protocol](APP_BENCHMARKS.md)

## Hardware and use

The persistent worker selects Metal on Apple silicon, CUDA on NVIDIA, or Vulkan on supported AMD and Intel graphics, with automatic CPU fallback. Audio passes through the app's normalization and pause-aware segmentation. Recognition runs locally after installation. OpenWhispr provides capture, endpointing, history and paste; the model supplies recognition.

The source model supports Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Spanish, Swedish and Ukrainian.

Select **Oruk → Orukeet** in local speech settings and click Download. The installer fetches and verifies the public Q8 file. [Build instructions](README.md) cover native acceleration and device overrides.

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
