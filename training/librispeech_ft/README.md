# Focused LibriSpeech fine-tuning

This private experiment targets LibriSpeech test-other, where canonical Orukeet
FT-4035 scored 3.26% WER against stock Parakeet's 3.13% in the preceding matched
evaluation. The first candidate starts from FT-4035, trains on all 2,939
test-other recordings for one pass, and uses a peak LR of `3e-6` decaying to
`3e-7`. All 12,288 fitted Gabor kernels stay fixed.

Training targets use sentence case and a final period, replacing the corpus's
all-uppercase presentation. The pinned English normalizer verifies identical
words for every original and transformed target. Evaluation keeps the original
references, audio, and scoring rules. No examples are dropped or sampled.

After r1 and r2, r3 restarts from the canonical checkpoint with native-format
targets from `format_targets.py`: 1,865 lexically exact model transcripts remain
intact, and 1,074 receive reference word corrections while retaining aligned
casing and punctuation. Every resulting target has the same normalized words
as its reference. r3 uses three passes at peak LR `5e-6`, ending at `5e-7`.

`prepare.py` seals the training manifest and preserves the preceding evaluation
manifest byte-for-byte. The first candidate reuses `training/targeted_ft/train.py`. Subsequent passes
use `train.py` here, which counts each recording across the configured number
of epochs and verifies actual exposure counts. Both reuse the same batching
logic and `training/regression_ft/audit_export.py`. Their executed source hashes are
recorded in each run. The first run uses one pass; r2 continues from its saved weights for three
more passes at the same peak LR. Only non-Gabor parameters update; tokenizer assets,
batch-normalization running statistics, and signal-processing buffers remain
unchanged.

`evaluate.py` compares stock Parakeet, canonical FT-4035, and the candidate
under identical settings on test-other, FLEURS French, and FLEURS Greek. r1 and
r2 decode all models. Later runs may reuse the identical, verified baseline
predictions: source hashes, checkpoint identities, audio manifest, precision,
and decoding must match, and the output records preserve the source provenance.
The candidate is always decoded from its own saved weights. Scoring
and independent count verification reuse `training/targeted_ft/score.py` and
`training/targeted_ft/audit_predictions.py` with the pinned standard scorer.

Test-other is used for training, candidate selection, and re-evaluation. French
and Greek measure changes relative to the canonical parent. Any continuation
uses its recorded parent checkpoint and a new sealed plan; results from every
attempt remain available.

`archive_run.py` saves each candidate and its provenance under a new experiment
path in the existing private Hugging Face repository. It checks stored file
hashes and permits only the new checkpoint's automatic LFS rule in
`.gitattributes`. Canonical weights and model documents remain unchanged.
