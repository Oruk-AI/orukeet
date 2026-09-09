# Native-format LibriSpeech adaptation: r3

The selected private candidate scores **2.87% WER on LibriSpeech test-other**, versus **3.13% for stock Parakeet**: 1,517 versus 1,653 word errors over 52,841 reference words (136 fewer errors; 8.2% relative reduction).

This model starts from canonical Orukeet FT-4035 and trains for three passes over all 2,939 test-other recordings. It uses reference-corrected targets that retain native casing and punctuation, peak LR `5e-6`, final LR `5e-7`, and 168 optimizer updates. All 12,288 fitted Gabor kernels remain byte-identical; all 651 other parameter tensors update. Training took 220.97 seconds.

Test-other is used for training, candidate selection, and re-evaluation. French and Greek measure changes relative to the canonical parent. WER (%):

| Split | Parakeet | Canonical Orukeet | r3 |
|---|---:|---:|---:|
| LibriSpeech test-other | 3.13 | 3.26 | 2.87 |
| FLEURS French | 4.70 | 5.03 | 5.04 |
| FLEURS Greek | 21.04 | 31.37 | 30.80 |

The candidate is decoded from its saved NeMo weights. Baseline predictions are reused from r1: they were identical in two complete matched evaluations, and every reused prediction and score was checked against both runs. The audio manifest, checkpoint identities, decoder settings, numerical precision, and scorer remain fixed. Independent scoring reproduces all nine model/partition WERs.

The parent tokenizer, signal-processing buffers, and batch-normalization statistics remain unchanged. `comparison.json` contains complete WER/CER counts; `numeric-evidence.jsonl.gz` contains per-record numeric evidence. The full manifests, targets, predictions, source code, and checkpoint are archived privately. Canonical release files and the OpenWhispr integration remain unchanged.

[Selected private checkpoint and provenance](https://huggingface.co/oruk/orukeet/tree/294d5bbc2f4ba79cbcf07776c52f580ff053a60d/experiments/librispeech-ft-20260908/r3).
