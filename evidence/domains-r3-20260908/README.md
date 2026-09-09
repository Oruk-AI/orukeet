# Orukeet r3: accent and domain comparison

Orukeet scores **15.25% pooled WER versus Parakeet's 16.72%** on 12,006 recordings across 47 partitions and 25 languages, an 8.8% relative reduction. It improves 36 of the 47 partitions. On all 20 English partitions, comprising 5,120 recordings, WER is **8.84% versus 9.51%**, a 7.0% relative reduction; every English partition improves.

| Pooled comparison | Parakeet errors / words | Parakeet WER | Orukeet errors / words | Orukeet WER |
|:--|--:|--:|--:|--:|
| All 47 partitions | 43,939 / 262,747 | 16.72% | 40,068 / 262,698 | 15.25% |
| All 20 English partitions | 9,032 / 94,993 | 9.51% | 8,399 / 94,993 | 8.84% |

The sample retains its existing 256 recordings per partition and all 230 Lesbos recordings. Of these, 6,118 recordings were included in the earlier FT-4035 adaptation. Greek and Italian EuroSpeech retain their audited human transcript spans. The generic manifest adds three evaluator field aliases while preserving all original fields, references, waveforms and membership.

Both models were decoded afresh, using FP32 NeMo weights, BF16 CUDA autocast, greedy-batch TDT and identical duration-sorted 16 kHz audio. Orukeet is the r3 checkpoint, SHA-256 `031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`. All 12,006 recordings are scored; 70 empty Parakeet hypotheses and 61 empty Orukeet hypotheses remain.

The final scores use the same pinned English/multilingual normalization and compound-aware WER as the current complete LibriSpeech/FLEURS comparison. This protocol differs from the older domain report's diagnostic scoring, so the current report uses only these freshly paired scores. `inference-comparison.json` retains the earlier diagnostic normalization; `comparison.json` records the final manuscript scores.

`hypotheses-audit.json` independently re-scores all 24,012 predictions and verifies every numerator and denominator against the per-record numeric evidence. Full-partition scoring with unmodified upstream normalizer definitions reproduces all 94 model/partition WERs. `scores.csv` retains full-precision WER/CER and edit counts. The [complete table](../../docs/current-checkpoint-benchmarks.md) includes all 47 comparisons.

All weights, manifests, predictions and report materials remain private for review.
