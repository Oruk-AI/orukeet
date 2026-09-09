<!-- orukeet-brand:start -->
<p><img src="assets/team/oruk.png" alt="Oruk AI" width="184"></p>
<!-- orukeet-brand:end -->

# Orukeet technical report

<!-- orukeet-team:start -->
<p>
Nathan Roll<sup>1,2</sup> · Irene Yi<sup>1,2</sup> · Büşra Marşan<sup>1,2</sup><br>
Vianney Grenez<sup>1</sup> · Gabriel Stein<sup>4</sup> · Momcilo Mrkaic<sup>5</sup><br>
Pavle Padjin<sup>5</sup> · Vladimir Zeljkovic<sup>5</sup> · Calbert Graham<sup>1,3</sup>
</p>

<p><strong><sup>1</sup> Oruk AI</strong></p>
<table>
<tr>
<td align="center" valign="middle"><img src="assets/team/stanford.png" alt="Stanford University" width="144"><br><sup>2</sup> Stanford University</td>
<td align="center" valign="middle"><img src="assets/team/cambridge.png" alt="University of Cambridge" width="144"><br><sup>3</sup> University of Cambridge</td>
<td align="center" valign="middle"><img src="assets/team/openwhispr.png" alt="OpenWhispr" width="40"><br><sup>4</sup> OpenWhispr</td>
<td align="center" valign="middle"><img src="assets/team/hoid.png" alt="Hoid" width="76"><br><sup>5</sup> Hoid</td>
</tr>
</table>
<!-- orukeet-team:end -->

[Short report PDF](../output/pdf/orukeet-technical-report.pdf) · [LaTeX source](../report/paper.tex) · [All paired scores](current-checkpoint-benchmarks.md)

The report describes **Orukeet r3**, the selected checkpoint with SHA-256 `031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`. It recognizes the 25 languages supported by Parakeet TDT 0.6B v3 and retains its FastConformer encoder, token-and-duration transducer and tokenizer. Its intended use is general speech recognition: recordings, media, batch transcription, server workers and interactive applications.

## Fitted functions inside the encoder

Each of the 24 encoder blocks contains 1,024 nine-tap temporal depthwise filters. We fit a separate Gabor function to every filter in an adapted Parakeet checkpoint:

$$g_i(t)=A_i\exp\left[-\frac{(t-\mu_i)^2}{2\sigma_i^2}\right]\cos\left(2\pi f_i(t-\mu_i)+\phi_i\right),\quad t=-4,\ldots,4.$$

The fit uses float64 variable projection. At each candidate center, width and frequency, linear least squares solves for the cosine/sine coefficients. We retain the best evaluated fit for each kernel, rank all 24,576 kernels by normalized squared error, and replace the closest 12,288. No offset or residual is added. The [fitting recipe](../training/gabor_half/README.md) records the search bounds, initial grid and tie-break.

The selected fits have 6.32% median relative RMS error, a 13.30% cutoff, and pooled squared error equal to 0.4244% of the selected original weight energy. Global selection gives 175–748 fixed kernels per layer. [Figure 1](../report/assets/kernel-fits.pdf) shows four predetermined fit-error ranks; [Figure 2](../report/assets/selection-profile.pdf) shows the complete fit distribution and layer allocation. Both figures use the original fits retained exactly in r3.

The materialized model contains 627,008,134 scalar parameters. Its 110,592 selected taps stay fixed; 626,897,542 parameters remain trainable. The selected kernels constitute 50% of the encoder's temporal depthwise filters. Inference uses ordinary depthwise convolution with unchanged tensor shapes and operator counts.

## Final adaptation

After recovery and 4,035 low-learning-rate adaptation updates, the final pass performs 168 AdamW updates with a 3% warmup and cosine decay from `5e-6` to `5e-7`. It makes three passes over all 2,939 LibriSpeech test-other recordings. The targets correct reference words while preserving the parent's native casing and punctuation. Every target has exactly the same normalized words as its reference.

Test-other is used for training, checkpoint selection and re-evaluation. The [run recipe](../training/librispeech_ft/README.md), [sealed plan](../evidence/librispeech-ft-20260908/r3/plan.json) and [export audit](../evidence/librispeech-ft-20260908/r3/export-audit.json) specify the procedure. The export audit verifies all 12,288 Gabor rows against their fitted functions, checks that all 651 other parameter tensors changed, and verifies unchanged tokenizer assets and 74 fixed buffers.

## Evaluation and multilingual pooled WER

The report uses fresh matched decoding of this checkpoint and stock Parakeet on two fixed comparisons:

- Both complete LibriSpeech test partitions and all 25 FLEURS test languages: 25,705 recordings, including 20,146 FLEURS recordings.
- A fixed accent/domain sample across 47 partitions and 25 languages: 12,006 recordings, including 5,120 English recordings across 20 partitions.

Both models receive identical mono 16 kHz audio and use NeMo greedy-batch TDT, FP32 weights and BF16 CUDA autocast. Matrix-multiply TF32 is disabled. Every recording remains in the score, including empty hypotheses.

Pooled WER sums substitutions, deletions and insertions, then divides by the summed normalized reference-word count. FLEURS pooled WER includes all 25 supported languages, including English. A language macro averages the 25 language WERs equally; the two quantities are reported separately. Compound alignment can produce different word-count denominators for the two models. CER counts character edits before compound alignment.

The accent/domain sample retains its original membership and audited Greek/Italian EuroSpeech transcript spans. Its preceding adaptation includes 6,118 sampled recordings. The two comparisons are scored separately. [The complete score companion](current-checkpoint-benchmarks.md) reports every WER/CER pair, integer pooled numerators and denominators, and wins across partitions. Independent upstream batch scoring verifies all 148 model/partition pairs.

## Exact checkpoint and reproducibility

The [release checkpoint](https://huggingface.co/oruk/orukeet/resolve/555136b50265a132d4cea0d35560c26fc4f657ab/orukeet-v0.1.0.nemo) is a 2,509,342,720-byte NeMo file. [The report identity](../report/model.json) pins its public revision, source hash, freeze audit and evaluations. Restore it directly in the recorded environment.

```sh
.venv/bin/python evaluation/standard_asr/build_current_report.py
.venv/bin/python scripts/build_neurips_report.py
```

The first command reconstructs all 74 paired split scores and pooled summaries from per-record counts. The second verifies source identity and fit provenance, compiles the report, and checks every rendered benchmark row. The build receipt records visual review of the PDF.

Code is MIT; weights and fitted kernels are CC BY-SA 4.0; metric records are CC BY 4.0. NVIDIA's foundation attribution is retained. Dataset audio comes from its original providers. The canonical NeMo, Q8 and F16 files all share this r3 source. The [artifact catalog](../src/orukeet/artifacts.json) pins their download revision and hashes; `release/model-stages.json` records conversion and runtime evidence. The OpenWhispr integration selects the same Q8 export.

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

[Download BibTeX](../CITATION.bib) · [Citation metadata](../CITATION.cff)
<!-- orukeet-citation:end -->
