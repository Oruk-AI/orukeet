# FT-4035 continuation

This run produces **Orukeet FT-4035**, the checkpoint behind every current
NeMo, Q8 and F16 download. It completes 4,035 low-learning-rate updates with
all 12,288 fitted Gabor kernels frozen. Across 12,006 matched recordings,
pooled WER is 16.52% versus Parakeet's 17.97%; English WER is 10.13% versus
10.84% across 5,120 recordings.

From the repository root, reproduce the numeric checks with:

```sh
python training/regression_ft/verify_results.py evidence/regression-ft-20260907
python training/regression_ft/summarize_english.py evidence/regression-ft-20260907
```

Both commands use the committed integer counts and run without audio or a GPU.
The [complete score tables](../../docs/benchmark-scores.md) retain every
split and the exact model identities.

## Recipe

This run continues Orukeet R15-0100 on the complete evaluated partitions that
regressed against base Parakeet in WER or CER under either reported normalization.
The original selection contains 230,309 recordings from 24 splits (399.60 hours).
The immutable selection and comparison hashes are in
[`plan.json`](../../evidence/regression-ft-20260907/plan.json).

The run uses one finite pass, AdamW at a peak learning rate of `1e-6`, 3% linear
warmup, and cosine decay to `1e-7`. Microbatches contain at most 16 utterances and
120 padded audio seconds; four microbatches contribute to each update. The TDT
loss is weighted by target-token volume across the complete accumulation group.
Duration buckets are shuffled globally, with each accepted recording used once.
Dropout, spectral augmentation, and input dithering are disabled for this pass.
Batch-normalization running statistics stay fixed; their affine parameters train.

All 12,288 fitted Gabor rows are immutable buffers. Only the other rows are
exposed to the optimizer. The trainer checks finite gradients, optimizer coverage,
frozen-row hashes, exact recording coverage, and finite exported tensors. A
separate audit reads the saved NeMo archives and compares their Gabor values,
normalization buffers, tokenizer assets, tensor keys, shapes, and dtypes. Exports
use ordinary dense convolution weights.

The complete run finished **4,035 optimizer updates in 84.07 minutes** on one
A100 40 GB. Its final NeMo SHA-256 is
`0ccfefcd1894871cb0850bd3c464adf5397752840de2a76d1d2d075c4141a945`.
The [export audit](../../evidence/regression-ft-20260907/export-audit.json) verifies
all **110,592 fixed Gabor coefficients** against both the parent checkpoint and
the original fitted functions. All **651 trainable parameter tensors changed**;
the 74 fixed buffers, including the analysis window and mel filterbank, remain
exact. The [training-log audit](../../evidence/regression-ft-20260907/training-log-audit.json)
checks the full step sequence, finite values, learning-rate endpoints, and exact
dataset coverage.

The first export cleared `validation_ds`, which NeMo's `transcribe` method
requires even when its manifest is null. `repair_export.py` restores that
metadata from the parent and verifies that every weight and tokenizer byte is
unchanged. The original training export is preserved, and the
[packaging receipt](../../evidence/regression-ft-20260907/export-metadata-repair.json)
connects its hash to the inference package. The training completion record retains
the original export hash.

## Results

All three checkpoints completed the same 12,006 diagnostic recordings with zero
inference failures. The [per-split results](../../evidence/regression-ft-20260907/RESULTS.md)
use standard English normalization for English and the existing multilingual
normalizer elsewhere.

| Diagnostic sample | Recordings | Parakeet WER | R15-0100 WER | Fine-tuned WER |
|:--|--:|--:|--:|--:|
| Training-exposed fit checks | 6,118 | 12.54% | 12.12% | 10.85% |
| Controls excluded from this run | 5,888 | 21.73% | 19.51% | 20.46% |

The candidate reduces fit-check WER by 10.47% relative to R15-0100 and 13.43%
relative to Parakeet. It beats Parakeet on 22 of the 24 fit-check samples; Italian
EuroSpeech and Golos Crowd remain worse. Compared with R15-0100, 20 fit-check
samples improve, one ties, and three worsen.

Control WER rises by 0.95 percentage points against R15-0100 (paired 95% interval:
0.75–1.16), while remaining 1.27 points below Parakeet. Eighteen of 23 control
samples worsen against R15-0100. The largest increases are Slovenian, Ukrainian,
and Latvian EuroSpeech. Control CER also rises, from 9.76% to 10.56%; fit-check CER
falls from 6.53% to 5.64%. FT-4035 is the user-selected canonical Orukeet model. R15-0100 remains archived as the preceding checkpoint.

These are fixed-sample diagnostics. The fit checks contain training data, and the
controls come from the earlier benchmark that informed split selection. The
[numeric audit](../../evidence/regression-ft-20260907/numeric-audit.json) independently
recomputes all 402 split-level WER/CER rates and both pooled summaries from
per-record integer counts, and checks every recording against the sealed sample.
The [full summary](../../evidence/regression-ft-20260907/summary.json) retains paired
intervals, cluster counts, and both error metrics.

The [English-only check](../../evidence/regression-ft-20260907/ENGLISH.md) aggregates
all 20 English splits from that same diagnostic, including 17 training-exposed
splits. Across 5,120 recordings, WER is 10.84% for Parakeet, 10.82% for R15-0100,
and 10.13% for the candidate. The candidate beats Parakeet on all 20 sampled splits.

## Source alignment

Before text formatting, twenty-two partitions retain every original pair. The earlier benchmark found
source alignment defects in Greek and Italian EuroSpeech. Re-extracting 15
Italian clips from the Parliament's source recordings reproduced the mismatches;
the published transcript alignment was the problem. Greek source retrieval
returned HTTP 403. The provider's 24 kHz test set uses different sessions.

For these two partitions, `realign.py` uses the frozen base model's existing
hypotheses to locate nearby spans in the published human transcripts. Labels are
copied from those human passages, with source offsets recorded for reconstruction.
No predicted words are inserted into a label. The alignment window, error and
length thresholds, accepted records, and quarantine are saved before training.
The original benchmark references and results remain unchanged.

A fixed sample of 128 accepted pairs was checked with
[Whisper large-v3](https://huggingface.co/openai/whisper-large-v3), pinned to revision
`06f233fe06e710322aca913c1bc4249a0d71fce1`. The complete audit includes its failures
and disagreements. Its output supplies no training labels. Two pairs with
boundary disagreements are also quarantined. The resulting pool contains
**223,599 recordings across all 24 splits, totaling 371.60 hours**; **6,710 pairs**
are quarantined. This is automatic alignment with a sampled independent check,
not manual annotation of every recording.

## Training labels

The final sealed pool contains **223,452 recordings, 371.47 hours, and all 24
selected splits**. Formatting removes explicit non-speech tags, typography
delimiters, and excess whitespace. It retains spoken words, including NST's
punctuation names, and maps unsupported Greek and Bulgarian character variants
to the unchanged tokenizer. Greek final sigma becomes sigma; the rare unsupported
diaeresis forms use their supported vowel spelling. Casing is preserved. Original
references and the repaired human spans remain in separate manifest fields.

An additional 147 annotation-only records have no lexical training target and
join the quarantine, for **6,857 excluded pairs** in total. Every accepted label
has zero unknown BPE tokens; the maximum target length is 194 tokens. The complete
per-split counts, character mappings, and script hash are in
[`text-normalization.json`](../../evidence/regression-ft-20260907/text-normalization.json).
The [NST source card](https://huggingface.co/datasets/alexandrainst/nst-da) and
[GigaSpeechBench source](https://github.com/SpeechColab/GigaSpeechBench) identify the
corpora; `text.py` records the exact training-specific transformation. Benchmark
scoring continues to use the preserved evaluation references.

## Reproduction and evaluation

Run `prepare_audio.py`, `realign.py`, `check_alignment.py`, and `seal_training.py`
in that order, using the original benchmark's pinned source manifests and
predictions. `train.py` consumes the sealed plan and manifests. `--resume` restores
the optimizer, RNG state, and next-batch cursor; `--stop-after-step` supports an
explicit recovery test. The preflight recovered the same next-update loss and
gradient norm as an uninterrupted run. Test states never enter the actual run.

After training, `finish_run.py` applies the metadata repair, audits the portable
weights, runs the matched comparison, summarizes it, and archives the experiment
in the existing private Hugging Face repository. `fetch_private.py` downloads and
hash-verifies the checkpoint, optimizer state, and provenance bundle locally.

Training caches use mono 16 kHz PCM16 FLAC. Preparation verifies each waveform's
float32 hash against the benchmark before conversion and records the decoded
training hash and any clipped samples. All comparison models receive identical
files in the follow-up diagnostic.

`prepare_diagnostic.py` selects up to 256 records per split by a fixed hash rule.
It selects training-exposed fit checks and controls from other benchmark splits,
excluding available matches to training waveforms, normalized text, and recording
or speaker groups. `evaluate.py` compares base Parakeet, R15-0100, and the new
candidate. `summarize.py` reports corpus WER/CER and paired 95% intervals from
2,000 bootstrap samples, stratified by split and grouped by available recording
or speaker identifiers. These are fixed-sample diagnostics; fit checks are not
unseen results.

FT-4035 now supplies the canonical NeMo, Q8 and F16 weights. The model cards and private OpenWhispr default identify those exact weights. The promotion preserves the original benchmark records and previous checkpoint revisions.

The checkpoint, full optimizer state, and 1.22 GB provenance bundle are backed up
at the [private experiment revision](https://huggingface.co/oruk/orukeet/tree/940338ff2f8ccc36b1ba1d569f768e7b9e5cc5f3/experiments/regression-ft-20260907/r1).
The [archive receipt](../../evidence/regression-ft-20260907/private-archive.json)
records every file's size and SHA-256. `fetch_private.py` also checks that the
commit leaves existing files unchanged, apart from an appended LFS storage rule
for the new checkpoint.
The [provenance audit](../../evidence/regression-ft-20260907/private-provenance-audit.json)
checks all 24 packed training manifests, all 4,035 update records, the training
code and Gabor fits, and the complete predictions for all three models against
the sealed evidence.
All five archived files are also retained locally with matching hashes;
[`local-artifacts.json`](../../evidence/regression-ft-20260907/local-artifacts.json)
records their paths and verifies the scope of the private archive commit.

`exposure.py` records the new checkpoint's training waveform, transcript, and
available speaker/recording fingerprints. Its row counts must match the completed
trainer's coverage. The candidate inherits its parent's training exposure; the
original benchmark also informed dataset selection. Historical benchmark files
remain intact, and future evaluations must account for this new exposure record.

The source model completed all 12,006 NeMo transcriptions with zero failures.
The initial local validation passed 24 package tests, ten targeted
training/export tests and three selection checks; the package built and its
CLI ran successfully. The saved
[`repository-validation.json`](../../evidence/regression-ft-20260907/repository-validation.json)
records that initial check. Subsequent CI passed package checks on Mac,
Windows and Linux and 17 real-Q8 native inference checks on each of Windows
and Linux. The [current validation record](../../release/model-stages.json)
links FT-4035's exact source/export identities; platform receipts remain with
the separate private OpenWhispr integration.
