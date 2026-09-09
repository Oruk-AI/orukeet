---
language: [bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, ru, sk, sl, es, sv, uk]
license: cc-by-sa-4.0
base_model: nvidia/parakeet-tdt-0.6b-v3
base_model_relation: finetune
pipeline_tag: automatic-speech-recognition
library_name: nemo
tags: [parakeet, tdt, gguf, multilingual, speech-recognition, gabor, fastconformer]
---

<!-- orukeet-brand:start -->
<p><img src="docs/assets/team/oruk.png" alt="Oruk AI" width="184"></p>
<!-- orukeet-brand:end -->

# Orukeet

<!-- orukeet-team:start -->
<p>
Nathan Roll<sup>1,2</sup> · Irene Yi<sup>1,2</sup> · Büşra Marşan<sup>1,2</sup><br>
Vianney Grenez<sup>1</sup> · Gabriel Stein<sup>4</sup> · Momcilo Mrkaic<sup>5</sup><br>
Pavle Padjin<sup>5</sup> · Vladimir Zeljkovic<sup>5</sup> · Calbert Graham<sup>1,3</sup>
</p>

<p><strong><sup>1</sup> Oruk AI</strong></p>
<table>
<tr>
<td align="center" valign="middle"><img src="docs/assets/team/stanford.png" alt="Stanford University" width="144"><br><sup>2</sup> Stanford University</td>
<td align="center" valign="middle"><img src="docs/assets/team/cambridge.png" alt="University of Cambridge" width="144"><br><sup>3</sup> University of Cambridge</td>
<td align="center" valign="middle"><img src="docs/assets/team/openwhispr.png" alt="OpenWhispr" width="40"><br><sup>4</sup> OpenWhispr</td>
<td align="center" valign="middle"><img src="docs/assets/team/hoid.png" alt="Hoid" width="76"><br><sup>5</sup> Hoid</td>
</tr>
</table>
<!-- orukeet-team:end -->

Orukeet is a 25-language speech recognizer built from NVIDIA Parakeet TDT 0.6B v3. It replaces half of the encoder's temporal depthwise filters with **12,288 fitted, frozen Gabor kernels** and trains the remaining parameters on multilingual and multi-accent data.

Orukeet outperforms Parakeet on **61 of 74 tested splits**, including LibriSpeech test-clean (**1.46% vs. 1.53% WER**), test-other (**2.86% vs. 3.14%**), and FLEURS English (**3.82% vs. 4.28%**). Across all 25 FLEURS languages, pooled WER is **9.85% vs. 11.01%**, a **10.6% relative reduction**. Final adaptation and checkpoint selection use LibriSpeech test-other.

Use Orukeet for recordings, media, batch transcription, server workers and interactive applications. NeMo, Q8 and F16 all derive from the same **r3 release checkpoint** (`031c8ddab484`).

## Architecture

The model retains Parakeet's 627,008,134 parameters, 24-layer FastConformer encoder, token-and-duration transducer and tokenizer. Each encoder block contains 1,024 nine-tap temporal depthwise filters. A selected filter stores its own fitted Gabor function:

$$g(t)=A\exp\left[-\frac{(t-\mu)^2}{2\sigma^2}\right]\cos\left(2\pi f(t-\mu)+\phi\right),\quad t=-4,\ldots,4.$$

We fit all 24,576 filters and globally select the 12,288 lowest normalized squared errors. This selects 175–748 kernels per layer, with 6.32% median relative RMS error and a 13.30% cutoff. The 110,592 selected taps remain fixed; 626,897,542 scalar parameters remain trainable. Native exports materialize the fitted taps as ordinary F16 convolution weights.

![Four exact kernel fits](report/assets/kernel-fits.png)

## Construction

Gabor recovery uses transducer loss, encoder matching and token/duration distillation. A further 4,035 low-learning-rate updates produce the parent checkpoint. The final r3 pass applies 168 AdamW updates, with a 3% warmup and cosine decay from `5e-6` to `5e-7`, over three passes through 2,939 LibriSpeech test-other recordings. Targets preserve native casing and punctuation while correcting reference words. The same split supplies checkpoint selection. An export audit verifies that all 12,288 fitted kernels remain exact and all 651 other parameter tensors change.

[Fit and freeze recipe](training/gabor_half/README.md) · [Final adaptation](training/librispeech_ft/README.md) · [Training lineage](training/README.md)

## Evaluation

Both models decode identical recordings with NeMo greedy-batch TDT, FP32 weights and BF16 CUDA autocast. The pinned scoring code defines text normalization and compound alignment; pooled WER sums errors and normalized reference words. Lower is better.

| Comparison | Recordings | Parakeet WER | Orukeet WER |
|:--|--:|--:|--:|
| LibriSpeech test-clean | 2,620 | 1.53% | **1.46%** |
| LibriSpeech test-other | 2,939 | 3.14% | **2.86%** |
| FLEURS English | 647 | 4.28% | **3.82%** |
| FLEURS pooled, 25 languages | 20,146 | 11.01% | **9.85%** |
| Accents/domains pooled, 47 splits | 12,006 | 16.72% | **15.25%** |
| Accents/domains English, 20 splits | 5,120 | 9.51% | **8.84%** |

Orukeet improves 25 of 27 complete LibriSpeech/FLEURS splits and 36 of 47 accent/domain splits, including all 20 English accent/domain splits. The accent/domain sample contains 256 recordings per split and all 230 Lesbos recordings; the preceding adaptation includes 6,118 sampled recordings. Read speech and accents/domains have separate pooled results. Every recording contributes to the scores.

[All 74 paired WER/CER scores and edit counts](docs/current-checkpoint-benchmarks.md) · [Methods](docs/technical-report.md) · [Technical report](output/pdf/orukeet-technical-report.pdf)

## Inference formats

The API returns transcript text and segment times. Language detection, diarization, translation, endpointing and caption layout are application-level functions.

## Model files

| Format | File | Bytes |
|:--|:--|--:|
| NeMo source | `orukeet-v0.1.0.nemo` | 2,509,342,720 |
| Native Q8 | `orukeet-v0.1.0-q8.gguf` | 714,456,704 |
| Native F16 | `orukeet-v0.1.0-f16.gguf` | 1,296,681,088 |

All three files derive from **r3**. Immutable weight revision: `555136b50265a132d4cea0d35560c26fc4f657ab`.

- NeMo SHA-256: `031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`
- Q8 SHA-256: `93ce19c6d8244acbfea980eeaf970531d4f216171578ef8e041dcc2d070a45bd`
- F16 SHA-256: `de53fb8ec251fb07ade15baabe17b00774ae3f1112f8618b062337f90fb49194`

Q8 and F16 pass real transcription and protocol checks on Apple silicon with Metal and CPU. Conversion audits verify all 12,288 fitted kernels after F16 rounding. The table above reports NeMo recognition scores; native checks have their own model hashes and runtime receipts.

[Artifact catalog](src/orukeet/artifacts.json) · [Native conversion and validation](evidence/r3-promotion-20260908/)

## License and attribution

Code: MIT. Weights and fitted kernels: CC BY-SA 4.0, retaining NVIDIA's foundation attribution. Transcript-free metric records: CC BY 4.0. Dataset audio is obtained from its original providers under their terms.

[Data provenance](docs/data-and-licenses.md) · [Attribution](NOTICE.md)

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

[Download BibTeX](CITATION.bib) · [Citation metadata](CITATION.cff)
<!-- orukeet-citation:end -->
