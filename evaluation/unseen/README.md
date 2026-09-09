# New-data comparison: Orukeet R15-0100 vs. Parakeet v3

This experiment freezes the released Orukeet checkpoint and stock NVIDIA
Parakeet TDT 0.6B v3 before running new evaluation sources. It is separate from
the historical 23,038-recording regression suite used during model selection.

The primary suite contains the complete Monsoon en-IN public test split,
all six English-accent and twelve English-domain components of GigaSpeechBench,
and the official EuroSpeech test splits for 16 supported languages. No sample
caps, model-confidence thresholds, or predicted-error filters select examples.
The final sealed registry determines the exact counts.

Danish and Swedish EuroSpeech are excluded because previous FTSpeech/RixVox
benchmarks may contain recordings from the same parliaments. The remaining
EuroSpeech languages are Bulgarian, Croatian, English, Estonian, Finnish,
French, German, Greek, Italian, Latvian, Lithuanian, Maltese, Portuguese,
Slovak, Slovenian, and Ukrainian. A separately sealed coverage extension
adds the nine missing languages; the primary suite remains unchanged.

## What “unseen” means here

`audit_history.py` fingerprints the recorded adaptation inputs and historical
predictions. It also treats unused reserve manifests as exposed. New dataset
repositories are absent from that history. An abandoned preparation draft
inspected some EuroSpeech metadata without training or evaluating a model.

Exact reference-text matches are flagged, retained for complete-split scoring,
and removed in a separate strict-history-disjoint analysis. During audio
preparation, decoded audio is also checked against the exact PCM fingerprints
available from previous audits. The final scorer also checks fingerprints made
by decoding the 23,113 available historical recordings. Positive controls found
that a second PCM16 conversion changes stored sample hashes; this supplemental
index repairs that representation mismatch while preserving the original sealed
index. The [repair receipt](../../evidence/unseen-20260907/decoded-history-audit.json)
and [control results](../../evidence/unseen-20260907/history-controls-repaired.json)
record the verification. These checks do not establish the absence of
acoustic near-duplicates or identify the same person across different corpora.
NVIDIA's released source summary does not permit a complete record-level audit
of inherited pretraining. New benchmark publication dates alone are not proof
that the underlying recordings postdate pretraining.

## Fixed comparison

Both models use the same NeMo container, greedy TDT decoder, maximum of 10
symbols per step, 16 kHz mono audio, FP32 weights, BF16 CUDA autocast, and batch
size 32. The model checkpoint hashes and full decoding configuration are
recorded. Inputs are sorted by duration identically. Model order alternates
between splits. Failed inference retries smaller batches with identical
settings; a permanent transcription failure stays in the denominator and is
scored as an empty hypothesis. Published audio is used in full even when its
length differs from the provider's duration metadata; these discrepancies are
counted, and decoded durations determine the reported audio hours.

Primary results are corpus WER and CER per split, a fixed-language EuroSpeech
macro, a six-accent macro, a twelve-domain macro, and Monsoon WER. The legacy
normalizer is tested against the previous evaluator. Standard English
Whisper normalization is reported separately, including number and contraction
normalization. English-normalized results are not represented as official
leaderboard submissions.

Confidence intervals use 10,000 paired cluster-bootstrap replicates. The
resampling unit is a Monsoon speaker, EuroSpeech recording session, or
GigaSpeechBench source recording. Macro intervals keep the language/category
mix fixed and resample clusters within each split. Counts below 20 clusters
are flagged; fewer than two clusters cannot support an interval. Per-split
intervals are descriptive, without correction for multiple comparisons.
Neither checkpoint is tuned or selected using these results.

## Run privately

The existing dedicated A100 uses this immutable container:

```text
nvcr.io/nvidia/nemo-speech@sha256:b8b1c094f1bbd1a28acec4a742ae8984f420c021293dd68e7cadb21ac2963ef0
```

The container includes NeMo 3.0.0, PyTorch 2.12.0+cu132, PyArrow 24.0.0,
SoundFile 0.13.1, soxr 1.0.0, and whisper-normalizer 0.1.12. Install
`rapidfuzz==3.14.1` for scoring. [Exact supplemental package pins](requirements.txt)
record the active container environment, including NumPy 2.4.4. Mount the historical project at its original
path. Use a task-specific scratch mount for transient downloads and normalized
audio; only this experiment's reproducible scratch files are removed after
both predictions and their receipts have been verified.

```sh
python code/audit_history.py --root "$PROJECT" --out "$EXPERIMENT/history"
python code/prepare_metadata.py --out "$EXPERIMENT/metadata" --history "$EXPERIMENT/history"
python code/test_metrics.py
python code/run.py --root "$PROJECT" --experiment "$EXPERIMENT" --scratch "$SCRATCH"
python code/audit_decoded_history.py --root "$PROJECT" --history "$EXPERIMENT/history"
python code/compare.py --experiment "$EXPERIMENT" --output "$EXPERIMENT/comparison"
```

`seal.json` must exist before inference. It binds dataset revisions, input
membership, checkpoint hashes, decoder settings, and the exclusion index.
`run.py` writes append-only predictions and verifies one-to-one membership
before marking a split complete. The comparison rechecks manifest hashes,
reference fingerprints, audio fingerprints, model identity, prediction counts,
and completion receipts. `--allow-partial` is for progress inspection; partial
results are explicitly labeled and cannot become a complete report.

`audit_decoded_history.py` verifies each available historical PCM hash and
fingerprints both its decoded float samples and the current PCM16 conversion.
It writes separate files without changing the original index. The final scorer
requires and verifies that supplemental receipt. `check_history_controls.py`
accepts a private historical-record fixture through `--controls`, the index
directory through `--history`, and a numeric receipt path through `--output`.
It verifies known positives through the actual preparation and scoring path,
then checks a seeded synthetic negative. No control audio enters ASR scoring.

Raw audio and transcripts remain private under the providers' terms. The
numeric evidence export contains pseudonymous grouping keys and error counts,
without audio, reference text, predictions, or personal profiles. No results
are submitted to public leaderboards.

## Sources

- [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [Monsoon en-IN public test](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN)
- [Monsoon collection and evaluation protocol](https://huggingface.co/blog/open-asr-leaderboard-global-south)
- [GigaSpeechBench data](https://huggingface.co/datasets/speechcolab/GigaSpeechBench)
- [GigaSpeechBench evaluation code](https://github.com/SpeechColab/GigaSpeechBench)
- [EuroSpeech](https://huggingface.co/datasets/disco-eth/EuroSpeech)

## Separate language-coverage extension

A second registry covers Czech, Danish, Dutch, Hungarian, Polish, Romanian,
Russian, Spanish, and Swedish. It includes full published VoxPopuli validation
partitions in six languages, Danish and Swedish NST test partitions, and two
Golos Russian test conditions. Its membership was sealed before its first
inference, after some primary-suite scores were available. It is supplementary
coverage, not another predeclared primary endpoint. Both frozen models use the
same decoder and precision as the primary suite.

The registry contains 102,345 scorable records. It records 99 Golos rows with
null reference transcripts as pre-inference exclusions. No model output decides
membership. Exact historical source IDs are excluded; exact text matches against
history and the primary registry are flagged for a separate sensitivity result.
VoxPopuli supplies previously unused clips, often from speakers heard in earlier
evaluations. The Swedish NST and Golos mirrors omit speaker/session identifiers;
some VoxPopuli rows also lack them. Intervals are omitted whenever complete
grouping is unavailable. The Swedish mirror's published test partition is not
represented as the original NST speaker-held-out split.

```sh
python code/prepare_coverage.py --primary "$EXPERIMENT" --output "$EXPERIMENT/coverage"
python code/run_coverage.py --root "$PROJECT" --experiment "$EXPERIMENT/coverage" --scratch "$COVERAGE_SCRATCH"
python code/compare.py --experiment "$EXPERIMENT/coverage" --output "$EXPERIMENT/coverage/comparison"
```

The coverage experiment uses the primary experiment's read-only historical
index through `coverage/history`. Its protocol and source revisions are stored
in `evidence/unseen-20260907/coverage/`. Together, the two registries cover all
25 supported languages across 45 splits and 326,401 scorable records.

## Reference-alignment diagnostic and follow-up

After the unusually high Greek and Italian parliamentary scores,
`audit_source_alignment.py` re-downloads two pinned source shards, checks all
1,929 metadata mappings, and verifies decoded PCM fingerprints for 16
deterministically spaced clips. It compares supplied references and the
provider's ASR text, then locates the closest text within the same session to
diagnose offsets. This analysis never substitutes a reference or removes a
record from scoring.

A third registry, sealed before its own inference, adds the complete Italian
VoxPopuli validation partition (1,257 clips) and ILSP's Lesbos Greek-dialect test
(230 clips). These 1,487 records provide independent checks after the source
issue was identified. The dialect test is not a standard-Greek benchmark. It
uses the same frozen models, inference wrapper, and scoring definitions. Its
protocol records the post-diagnostic selection and remains separate from the
primary and coverage results. All three runs contain 327,888 scored records
across 47 splits.

```sh
python code/prepare_coverage.py --primary "$EXPERIMENT" --coverage "$EXPERIMENT/coverage" --output "$EXPERIMENT/alignment-followup" --alignment-followup
python code/run_coverage.py --root "$PROJECT" --experiment "$EXPERIMENT/alignment-followup" --scratch "$FOLLOWUP_SCRATCH"
python code/compare.py --experiment "$EXPERIMENT/alignment-followup" --output "$EXPERIMENT/alignment-followup/comparison"
```

## Reproduce the released count statistics on CPU

After downloading this repository's transcript-free numeric exports, NumPy is
the only dependency for an independent check of every WER/CER count and seeded
paired confidence interval:

```sh
python evaluation/unseen/reproduce_counts.py --evidence evidence/unseen-20260907
```

The exports retain a within-split cluster ordinal so the original bootstrap
index order can be reproduced without publishing speaker or session names.
This path needs neither audio nor checkpoints. Rerunning the historical
exposure audit requires the private original manifests; source audio must be
obtained separately from its providers.

`diagnose_greek_script.py` records writing-script errors on the complete Greek
dialect test without changing its references or decoding. Its flags are
descriptive Unicode counts, not language-ID annotations.
