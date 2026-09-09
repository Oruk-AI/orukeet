# Paired evaluation for the r3 technical report

The report is bound to Orukeet NeMo SHA-256 `031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56` and stock Parakeet SHA-256 `3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d`.

Two manifests are evaluated independently with `run.py`: the complete 25,705-recording LibriSpeech/FLEURS manifest, and the existing 12,006-recording accent/domain sample. Both models are decoded afresh. The domain manifest adds the generic evaluator's `language`, `pcm_sha256` and `reference_sha256` aliases while preserving every existing source field, text, waveform and record. Its source manifest remains immutable.

`run.py` records an initial diagnostic score. `rescore.py` produces the manuscript scores using the pinned English/multilingual normalizers and compound-aware WER. `audit_predictions.py` independently imports the upstream normalizer definitions and scores complete partitions in batches. The report builder accepts only those audited scores.

```sh
python evaluation/standard_asr/rescore.py \
  --private-records /path/to/private-records \
  --evidence evidence/standard-asr-r3-20260908
python evaluation/standard_asr/audit_predictions.py \
  --private-records /path/to/private-records \
  --evidence evidence/standard-asr-r3-20260908 \
  --upstream-root /path/to/pinned-scoring-source
```

Repeat with `evidence/domains-r3-20260908` and its corresponding records. The source revision and per-file hashes for the scorer are in `vendor/provenance.json`. All normalizer dependencies are recorded in each final `comparison.json`.

Without private audio or transcripts, reproduce every reported score and build the PDF:

```sh
python evaluation/standard_asr/build_current_report.py
python scripts/build_neurips_report.py
```

Pooled WER divides summed edit counts by summed reference-word counts. FLEURS pooling includes all 25 languages, including English; the 25-language macro gives each language equal weight. Non-English compound alignment can change the reference-word denominator separately for each model. Win counts compare full-precision per-partition WERs. The two evaluation samples are never merged into one pooled headline.

Test-other was used for the final adaptation and checkpoint selection. The domain sample contains 6,118 recordings from the preceding adaptation and retains previously audited Greek/Italian EuroSpeech transcript spans. No recordings are dropped. Numeric evidence is transcript-free; full manifests and hypotheses are archived in the private model repository.
