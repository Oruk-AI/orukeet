# Reproduce Orukeet's benchmark scores

The current comparison measures **r3** and stock NVIDIA Parakeet on 74 splits:
27 complete LibriSpeech/FLEURS splits and 47 accent/domain samples. Orukeet
improves 61 splits. Pooled FLEURS WER is 9.85% versus 11.01%, and English
accent/domain WER is 8.84% versus 9.51%. [All paired scores and methods](../docs/current-checkpoint-benchmarks.md)
record model identities, decoding, normalization and membership.

## Current comparison

Recompute every stored per-record edit count, pooled summary and paired table
row without audio or a GPU:

```sh
python evaluation/standard_asr/build_current_report.py --verify-pdf
```

Install the [pinned scoring dependencies](standard_asr/requirements-score.txt).
The verifier checks all 74 benchmark rows in the PDF against numeric evidence.
To rebuild the report, with Tectonic and Poppler installed:

```sh
python scripts/build_neurips_report.py
```

The [r3 run guide](../training/librispeech_ft/README.md) records final adaptation
and source export. [Earlier FT-4035 results](../docs/benchmark-scores.md) and the
[preceding R15 evaluation](unseen/README.md) retain their own model identities.

## Inference formats

The NeMo source supplies the reported current WER/CER scores. Q8 and F16 are
direct conversions of r3. The [artifact catalog](../src/orukeet/artifacts.json)
pins all three hashes; [native fixture receipts](../evidence/r3-promotion-20260908/)
identify the exported weights and tested devices. [Usage examples](../docs/usage.md)
show persistent native workers and explicit F16 selection.

## Earlier adaptation and native comparison

The following metric bundle records the pre-Gabor adaptation checkpoint and
its Q8 export. These historical measurements use their own model hashes,
registered membership and aggregation rules.

### Recompute the earlier comparison

The committed [metric evidence bundle](../evidence/metric-evidence.tar.gz)
contains per-recording error/word counts, reference fingerprints and pseudonymous
group IDs for stock, source and Q8. It contains no audio, transcript text or
speaker names. Extract it into `artifacts/` with `tarfile` and `filter='data'`.
Then run:

```sh
python evaluation/compare_oruk_export.py \
  --reference artifacts/metric-evidence/stock \
  --candidate artifacts/metric-evidence/q8 \
  --registry artifacts/metric-evidence/registry.json \
  --output artifacts/stock-comparison.json
```

This reproduces the statistics, not recognition from audio. The
[verification receipt](../evidence/metric-evidence-verification.json) records
exact equality with both raw stock/Q8 and source/Q8 comparisons. The preparation
script preserves cluster sort order, so the original seeded bootstrap draws
remain identical. Reference fingerprints are checked for equality only; error
counts were computed from the original text before preparing this bundle.


For acoustic reruns, obtain the recorded source audio from its original providers
and preserve the experiment’s manifest, decoding and normalization settings.
Raw references and predictions remain in the private research archive; the public
metric bundles reproduce the statistical comparisons without redistributing them.

[Historical experiment record](../docs/pre-gabor-history.md) · [Export code](export_oruk_native.py)
