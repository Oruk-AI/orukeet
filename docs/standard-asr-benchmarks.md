# LibriSpeech and FLEURS benchmarks

Historical FT-4035 comparison, source hash `0ccfefcd1894871cb0850bd3c464adf5397752840de2a76d1d2d075c4141a945`. [Current r3 results](current-checkpoint-benchmarks.md) give the paired scores for the release checkpoint.

Both checkpoints use identical mono 16 kHz audio, NeMo greedy-batch TDT decoding, FP32 weights and BF16 CUDA autocast. English uses the pinned English text normalizer. Multilingual normalization retains diacritics and expands numbers by language. WER aligns compound boundaries, then uses compound-aware edit distance; CER measures normalized strings before boundary alignment. WER and CER pool integer edit counts and reference lengths within each test partition. Every test record is retained, including empty hypotheses. The five-language FLEURS macro averages German, Spanish, French, Italian and Portuguese; the 25-language macro includes every supported language.

| Benchmark | Clips | Parakeet WER / CER | Orukeet WER / CER |
|:--|--:|--:|--:|
| LibriSpeech test-clean | 2,620 | 1.53 / 0.59 | 1.50 / 0.58 |
| LibriSpeech test-other | 2,939 | 3.14 / 1.32 | 3.25 / 1.39 |
| FLEURS Bulgarian | 658 | 11.92 / 3.84 | 10.58 / 3.42 |
| FLEURS Croatian | 914 | 11.29 / 3.53 | 10.42 / 3.84 |
| FLEURS Czech | 723 | 11.12 / 3.21 | 9.20 / 2.74 |
| FLEURS Danish | 930 | 17.19 / 6.31 | 14.97 / 5.34 |
| FLEURS Dutch | 364 | 6.40 / 2.28 | 5.62 / 1.97 |
| FLEURS English | 647 | 4.28 / 2.00 | 3.87 / 1.80 |
| FLEURS Estonian | 893 | 13.32 / 3.86 | 10.62 / 3.53 |
| FLEURS Finnish | 918 | 11.14 / 2.59 | 9.52 / 2.20 |
| FLEURS French | 676 | 4.69 / 1.68 | 5.05 / 1.73 |
| FLEURS German | 862 | 4.21 / 1.41 | 3.94 / 1.55 |
| FLEURS Greek | 650 | 21.07 / 9.01 | 31.39 / 9.65 |
| FLEURS Hungarian | 905 | 13.60 / 4.20 | 10.86 / 3.10 |
| FLEURS Italian | 865 | 2.43 / 0.79 | 2.09 / 0.75 |
| FLEURS Latvian | 851 | 21.78 / 5.43 | 17.60 / 4.31 |
| FLEURS Lithuanian | 986 | 20.95 / 5.56 | 16.93 / 4.38 |
| FLEURS Maltese | 926 | 19.22 / 6.19 | 15.83 / 5.20 |
| FLEURS Polish | 758 | 6.81 / 2.09 | 6.21 / 2.00 |
| FLEURS Portuguese | 919 | 4.49 / 1.98 | 3.74 / 1.65 |
| FLEURS Romanian | 883 | 11.44 / 3.86 | 9.53 / 3.17 |
| FLEURS Russian | 775 | 4.89 / 1.49 | 4.84 / 1.54 |
| FLEURS Slovak | 792 | 9.21 / 2.91 | 7.77 / 2.43 |
| FLEURS Slovenian | 834 | 22.62 / 7.70 | 21.54 / 7.57 |
| FLEURS Spanish | 908 | 3.22 / 1.28 | 2.77 / 1.04 |
| FLEURS Swedish | 759 | 13.38 / 4.26 | 11.48 / 3.48 |
| FLEURS Ukrainian | 750 | 6.00 / 1.74 | 5.69 / 1.71 |
| FLEURS five-language macro | 4,230 | 3.81 / 1.43 | 3.52 / 1.34 |
| FLEURS 25-language macro | 20,146 | 11.07 / 3.57 | 10.08 / 3.21 |

[Evaluation and reproduction](../evaluation/standard_asr/README.md) · [Full-precision scores](../evidence/standard-asr-20260908/scores.csv) · [Per-record edit counts](../evidence/standard-asr-20260908/numeric-evidence.jsonl.gz)
