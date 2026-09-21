# Paired English16 result

[CI run 35570161829](https://github.com/Oruk-AI/orukeet/actions/runs/35570161829)
completed all 32 transcriptions at source commit
`541cf669e6da52c1e4e587f2ed3599d962c96555`. The PR merge checkout, exact runtime,
toolchain and source hashes are in [provenance.json](provenance.json).
The [repeat run](https://github.com/Oruk-AI/orukeet/actions/runs/35570838322)
at `60cc07cab9d45f34a43939e4768a454ed3be1b55` produced identical raw text for all
32 transcriptions and identical error counts; see [repeat verification](repeat-validation.json).

| Model | Substitutions | Deletions | Insertions | Errors / reference words | WER |
|---|---:|---:|---:|---:|---:|
| Parakeet v2 | 13 | 1 | 2 | 16 / 343 | 4.66% |
| Orukeet greedy | 14 | 2 | 4 | 20 / 343 | 5.83% |

Both use FluidAudio 0.15.5 with identical compute policies and fresh decoder
state for each of the same 16 hash-verified mono 16 kHz recordings. Every model
component, extracted source file, vocabulary and audio file was authenticated.
The 343-word denominator is the sealed reference text, unchanged for both models.

Keep the English Parakeet v2 selection. These 150.28 seconds of reused FLEURS
audio are a small regression diagnostic, not a general English benchmark or
evidence of iPhone performance. The result does not justify replacing either
OpenWhispr default. Spelling, number formatting and hyphenation can affect this
fixed scoring convention; inspect the raw transcripts alongside the counts.

The net four-error difference includes two additional errors from date ordering,
two from `well-rounded` versus `well rounded`, `periodic` becoming `pi`, and an
inserted `of`, offset by two improvements (`hot` to `high` and `capitol` to the
reference's `capital`). These observations explain the original counts; no
reference or normalization was changed after seeing the results.

- [Raw transcripts and execution metadata](raw.json)
- [Per-recording alignment counts and fixed-reference scores](scored.json)
- [Verified model and audio files](integrity.json)
- [Reproduction source and exact model/data pins](../../../export/coreml/english-comparison/README.md)

The v2 inputs were the published compiled `.mlmodelc` files; portable v2 source
packages are absent at that pinned revision. Successful macOS loading does not
establish their portability to iOS. CI timings are diagnostics and must not be
presented as iPhone latency. No weights or audio are stored in this evidence folder.
