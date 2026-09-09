# Targeted Orukeet pass — 8 September 2026

One pass over 4,265 recordings (9.199 hours), starting from Orukeet FT-4035: 88 AdamW updates, peak LR `1e-6`, final LR `1e-7`. Training took 134.33 seconds on an A100 40 GB. All 12,288 fitted Gabor kernels remained byte-identical. All 651 other parameter tensors updated; the tokenizer, batch-normalization buffers, and signal-processing buffers were preserved.

WER (%) below comes from fresh, matched decoding of all three checkpoints. These are re-evaluation scores on the same complete partitions used for this fine-tuning pass.

| Partition | Recordings | Stock Parakeet | Orukeet before | Orukeet after | Change vs. before (points) |
|---|---:|---:|---:|---:|---:|
| LibriSpeech test-other | 2,939 | 3.13 | 3.26 | 3.27 | +0.015 |
| FLEURS French | 676 | 4.70 | 5.03 | 4.98 | -0.054 |
| FLEURS Greek | 650 | 21.04 | 31.37 | 30.46 | -0.911 |

French and Greek improved. LibriSpeech test-other increased slightly, and all three scores remain above stock Parakeet. This candidate is archived as an experiment; canonical release weights and the OpenWhispr integration are unchanged.

The comparison uses FP32 weights, BF16 CUDA autocast, greedy TDT decoding, and the same pinned normalization and compound-aware WER as the preceding evaluation. Every recording and empty hypothesis is included. `comparison.json` contains full-precision WER/CER and error counts; `numeric-evidence.jsonl.gz` contains per-record counts without transcripts. `hypotheses-audit.json` records an independent full-partition scoring check.

The freshly decoded baselines differ slightly from the preceding larger evaluation, where the same recordings appeared in different padded batches. `baseline-repeatability.json` records all differences. The previous benchmark scores remain unchanged.

`plan.json`, `prepared.json`, `run.json`, `complete.json`, `training-metrics.jsonl.gz`, and `export-audit.json` identify the exact training pass and saved model. Full manifests, transcripts, predictions, code, and the checkpoint are kept in private artifact storage.

[Private checkpoint archive](https://huggingface.co/oruk/orukeet/tree/f0f73418e20ac02519a959e8f6ebb8b2b9cf1876/experiments/targeted-ft-20260908/r1). Checkpoint SHA-256: `6b9934a6728c712aa4ea406d483295399e1e9755a3c39412d6c98be30506d1e5`.
