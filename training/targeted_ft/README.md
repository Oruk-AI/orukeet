# Targeted Orukeet fine-tuning

This private experiment starts from Orukeet FT-4035 and makes one low-LR pass over
LibriSpeech test-other, FLEURS French, and FLEURS Greek. These are the three
partitions where FT-4035 had higher WER than stock Parakeet in the preceding
complete-test evaluation. The subsequent scores measure re-evaluation on the
same data used for this pass.

The pass uses all 4,265 recordings (9.199 hours), with original audio and
transcripts verified against their recorded hashes. No transcript formatting,
audio trimming, sampling, or unknown-token substitution is applied. The parent
tokenizer supports every original transcript.

AdamW reaches a peak LR of `1e-6` after a 3% linear warmup, then decays to `1e-7`.
The duration-bucketed batches contain at most 16 recordings and 120 padded
seconds, with four batches accumulated per update. Every recording appears once.
All 12,288 fitted Gabor rows remain fixed; every other parameter tensor receives
gradients. Batch-normalization running statistics and signal-processing buffers
are preserved. Training uses BF16 autocast with FP32 parameters and optimizer
states, gradient clipping at 1.0, and no augmentation.

`train.py` and `batches.py` derive from the preceding regression pass. The short
run can omit optimizer-state checkpoints with `--skip-optimizer-checkpoint`, and
its portable NeMo export preserves the validation configuration required by
`transcribe`. The export audit in `../regression_ft/audit_export.py` checks the
saved weights directly: unchanged Gabor values and buffers, updated remaining
parameter tensors, unchanged tokenizer assets, and finite values throughout.

`evaluate.py` freshly decodes all three checkpoints over the same 4,265 records,
using the preceding evaluation's complete decoding configuration. It preserves
empty hypotheses and verifies audio identities. `score.py` applies the existing
pinned English/multilingual normalization and compound-aware WER implementation.
The previous evaluation records remain separate and unchanged.

To reproduce, use the recorded NeMo environment, parent weights, Gabor fit
parameters, and source manifest. Run `prepare.py`, then `train.py`, then the
export audit. Supply `evaluate.py` with a model JSON mapping `parakeet`,
`orukeet_ft4035`, and `orukeet_targeted` to their paths and SHA-256 hashes. Run
`score.py` locally with `evaluation/standard_asr/requirements-score.txt` installed.
Each script exposes its input arguments through `--help`.

Numeric results and run receipts live in `evidence/targeted-ft-20260908/`.
Full transcripts, prepared manifests, and checkpoint weights remain in private
artifact storage. This experiment does not change the canonical release files
or the OpenWhispr integration.
