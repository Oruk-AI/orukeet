# Orukeet r3: complete LibriSpeech and FLEURS comparison

Orukeet reduces pooled FLEURS WER from **11.01% to 9.85%**, a **10.6% relative reduction**, across 20,146 recordings and all 25 supported languages. It has lower WER in 23 of the 25 languages and on 25 of all 27 complete read-speech partitions.

| Complete partition | Parakeet WER | Orukeet WER |
|:--|--:|--:|
| LibriSpeech test-clean | 1.53% | 1.46% |
| LibriSpeech test-other | 3.14% | 2.86% |
| FLEURS English | 4.28% | 3.82% |
| FLEURS French | 4.69% | 5.01% |
| FLEURS Greek | 21.07% | 30.81% |

The multilingual pooled totals are 46,442 errors / 421,870 reference words for Parakeet and 41,521 / 421,715 for Orukeet. Non-English compound alignment can change reference word boundaries separately for each model. FLEURS pooling includes English and gives recordings weight through their reference-word count. The 25-language macro is a separate quantity.

Every score uses the saved r3 NeMo checkpoint, SHA-256 `031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`. Both it and stock Parakeet were freshly decoded over identical 16 kHz audio with FP32 weights, BF16 autocast and matched greedy-batch TDT. The manifest contains 25,705 recordings and matches the original complete-test manifest exactly. The new Parakeet prediction file is byte-identical to its earlier complete-test run.

The final r3 adaptation and selection used LibriSpeech test-other. The complete run uses all 27 partitions in duration-sorted batches, giving slightly different padding contexts from the earlier three-partition comparison. Its paired test-other scores are 3.1414999716% and 2.8557370224%; all manuscript numbers come from this complete run.

`comparison.json` is the final pinned score; `inference-comparison.json` retains the runner's earlier diagnostic normalization. `hypotheses-audit.json` independently reproduces all 54 model/partition WERs from the private hypotheses. Per-record counts, hashes, full-precision scores and pooled summaries reproduce the numeric results without audio or transcripts. Five empty hypotheses from each model remain in the score.

The first attempt ran out of disk space while unpacking the baseline. The successful restart used dedicated RAM-backed temporary storage after verifying private immutable backups of two superseded checkpoint copies. `restart.json` records that repair. No inference result was reused from the failed attempt.

All artifacts remain private for review.
