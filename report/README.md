<!-- orukeet-brand:start -->
<p><img src="../docs/assets/team/oruk.png" alt="Oruk AI" width="184"></p>
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
<td align="center" valign="middle"><img src="../docs/assets/team/stanford.png" alt="Stanford University" width="144"><br><sup>2</sup> Stanford University</td>
<td align="center" valign="middle"><img src="../docs/assets/team/cambridge.png" alt="University of Cambridge" width="144"><br><sup>3</sup> University of Cambridge</td>
<td align="center" valign="middle"><img src="../docs/assets/team/openwhispr.png" alt="OpenWhispr" width="40"><br><sup>4</sup> OpenWhispr</td>
<td align="center" valign="middle"><img src="../docs/assets/team/hoid.png" alt="Hoid" width="76"><br><sup>5</sup> Hoid</td>
</tr>
</table>
<!-- orukeet-team:end -->

This short report describes **Orukeet r3**, the selected native-format checkpoint with SHA-256 `031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`. Its [model identity](model.json) pins the exact public NeMo artifact and freeze audit.

The report presents the fitted-filter construction, a compact adaptation recipe, complete paired LibriSpeech/FLEURS results, multilingual pooled WER, and the existing accent/domain sample re-evaluated with this checkpoint. Every score has the stock Parakeet comparison. The two figures show the unchanged fitted kernels and their allocation across encoder layers.

```sh
.venv/bin/python scripts/build_neurips_report.py
pdftoppm -png -r 150 output/pdf/orukeet-technical-report.pdf .tmp/orukeet-r3-page
```

The builder verifies checkpoint identity, exact frozen kernels, all per-record counts, pooled denominators and independent scoring audits. It generates the numeric TeX includes, compiles with the official NeurIPS 2026 preprint style, and verifies every rendered benchmark row. Pass `--tex-bin /path/to/texlive/bin` to use pdfLaTeX and BibTeX. The release PDF is built with TeX Live 2025, uses embedded outline fonts, and records its title and all nine authors in PDF metadata. The build receipt records input hashes, page count and visual review.

- [Report PDF](../output/pdf/orukeet-technical-report.pdf)
- [All current-checkpoint scores](../docs/current-checkpoint-benchmarks.md)
- [Methods companion](../docs/technical-report.md)
- [Complete kernel atlas](assets/kernel-atlas.pdf)

The manuscript and numerical evaluation records accompany Orukeet v0.1.0. The NeMo, Q8 and F16 release files all derive from this r3 checkpoint; `release/model-stages.json` and the artifact catalog record their exact identities.

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
