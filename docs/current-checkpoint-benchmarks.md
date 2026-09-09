# Orukeet r3: paired recognition scores

All Orukeet scores refer to NeMo SHA-256 `031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`. Parakeet is `3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d`. Both systems were decoded afresh on identical audio with FP32 weights, BF16 autocast and greedy-batch TDT. Lower WER is better.

Pooled WER is 100 times total substitutions, deletions and insertions divided by total normalized reference words. FLEURS pooling includes all 25 supported languages, including English. It is not an average of language WERs. Compound-boundary alignment can give each model a different reference-word denominator. CER uses normalized strings before compound alignment.

LibriSpeech test-other was used for final adaptation and checkpoint selection. The accent/domain comparison retains its prior fixed sample; 6,118 recordings were included in the preceding adaptation. Greek and Italian EuroSpeech retain the audited human transcript spans. No records are dropped from either comparison.

## Complete read-speech partitions

| Partition | Clips | Parakeet WER / CER | Orukeet WER / CER |
|:--|--:|--:|--:|
| LibriSpeech test-clean | 2620 | 1.53 / 0.59 | 1.46 / 0.56 |
| LibriSpeech test-other | 2939 | 3.14 / 1.32 | 2.86 / 1.19 |
| FLEURS Bulgarian | 658 | 11.92 / 3.84 | 10.37 / 3.34 |
| FLEURS Croatian | 914 | 11.29 / 3.53 | 10.20 / 3.67 |
| FLEURS Czech | 723 | 11.12 / 3.21 | 8.97 / 2.67 |
| FLEURS Danish | 930 | 17.19 / 6.31 | 14.88 / 5.31 |
| FLEURS Dutch | 364 | 6.40 / 2.28 | 5.60 / 1.93 |
| FLEURS English | 647 | 4.28 / 2.00 | 3.82 / 1.77 |
| FLEURS Estonian | 893 | 13.32 / 3.86 | 10.44 / 3.39 |
| FLEURS Finnish | 918 | 11.14 / 2.59 | 9.35 / 2.16 |
| FLEURS French | 676 | 4.69 / 1.68 | 5.01 / 1.70 |
| FLEURS German | 862 | 4.21 / 1.41 | 3.92 / 1.52 |
| FLEURS Greek | 650 | 21.07 / 9.01 | 30.81 / 9.18 |
| FLEURS Hungarian | 905 | 13.60 / 4.20 | 10.68 / 2.97 |
| FLEURS Italian | 865 | 2.43 / 0.79 | 2.09 / 0.76 |
| FLEURS Latvian | 851 | 21.78 / 5.43 | 17.41 / 4.21 |
| FLEURS Lithuanian | 986 | 20.95 / 5.56 | 16.55 / 4.27 |
| FLEURS Maltese | 926 | 19.22 / 6.19 | 15.60 / 5.08 |
| FLEURS Polish | 758 | 6.81 / 2.09 | 6.11 / 1.95 |
| FLEURS Portuguese | 919 | 4.49 / 1.98 | 3.73 / 1.63 |
| FLEURS Romanian | 883 | 11.44 / 3.86 | 9.34 / 3.07 |
| FLEURS Russian | 775 | 4.89 / 1.49 | 4.72 / 1.48 |
| FLEURS Slovak | 792 | 9.21 / 2.91 | 7.75 / 2.41 |
| FLEURS Slovenian | 834 | 22.62 / 7.70 | 22.11 / 8.28 |
| FLEURS Spanish | 908 | 3.22 / 1.28 | 2.75 / 1.04 |
| FLEURS Swedish | 759 | 13.38 / 4.26 | 11.36 / 3.45 |
| FLEURS Ukrainian | 750 | 6.00 / 1.74 | 5.39 / 1.60 |

[Full precision](../evidence/standard-asr-r3-20260908/scores.csv) · [Counts](../evidence/standard-asr-r3-20260908/numeric-evidence.jsonl.gz) · [Independent scoring audit](../evidence/standard-asr-r3-20260908/hypotheses-audit.json)

## Accent and domain sample

| Partition | Clips | Parakeet WER / CER | Orukeet WER / CER |
|:--|--:|--:|--:|
| EuroSpeech BG | 256 | 14.22 / 7.20 | 13.04 / 6.72 |
| EuroSpeech DE | 256 | 13.40 / 8.53 | 11.14 / 7.15 |
| EuroSpeech EL | 256 | 25.83 / 8.47 | 26.35 / 9.01 |
| EuroSpeech EN | 256 | 24.40 / 17.85 | 23.77 / 17.49 |
| EuroSpeech ET | 256 | 34.67 / 14.61 | 25.33 / 11.94 |
| EuroSpeech FI | 256 | 16.61 / 7.18 | 15.20 / 6.77 |
| EuroSpeech FR | 256 | 19.42 / 11.37 | 14.28 / 8.81 |
| EuroSpeech HR | 256 | 12.93 / 8.68 | 12.56 / 8.46 |
| EuroSpeech IT | 256 | 10.95 / 6.81 | 12.32 / 8.34 |
| EuroSpeech LT | 256 | 38.44 / 16.15 | 33.10 / 14.34 |
| EuroSpeech LV | 256 | 57.18 / 26.65 | 42.14 / 17.21 |
| EuroSpeech MT | 256 | 36.83 / 19.34 | 36.15 / 18.89 |
| EuroSpeech PT | 256 | 23.08 / 17.67 | 23.81 / 18.42 |
| EuroSpeech SK | 256 | 17.29 / 7.76 | 14.91 / 6.93 |
| EuroSpeech SL | 256 | 48.43 / 15.93 | 50.23 / 16.52 |
| EuroSpeech UK | 256 | 13.65 / 7.59 | 14.25 / 7.59 |
| GSB AI | 256 | 8.71 / 4.86 | 7.98 / 4.40 |
| GSB Chinese accent | 256 | 14.49 / 8.99 | 13.56 / 8.42 |
| GSB Filipino accent | 256 | 13.30 / 8.28 | 12.79 / 7.62 |
| GSB Indian accent | 256 | 6.50 / 3.20 | 5.59 / 2.52 |
| GSB Japanese accent | 256 | 19.15 / 11.89 | 17.78 / 10.86 |
| GSB Scottish accent | 256 | 22.08 / 14.60 | 20.35 / 13.12 |
| GSB Singaporean accent | 256 | 13.89 / 9.25 | 12.86 / 8.21 |
| GSB agriculture | 256 | 6.20 / 3.96 | 5.84 / 3.41 |
| GSB arts | 256 | 5.47 / 2.86 | 4.87 / 2.47 |
| GSB biology | 256 | 3.67 / 1.62 | 3.31 / 1.37 |
| GSB economics | 256 | 7.05 / 4.30 | 6.57 / 3.97 |
| GSB engineering | 256 | 4.06 / 2.19 | 3.50 / 1.85 |
| GSB entertainment | 256 | 10.40 / 6.80 | 8.87 / 5.64 |
| GSB finance | 256 | 5.81 / 3.41 | 5.11 / 3.00 |
| GSB humanities | 256 | 7.98 / 4.69 | 7.47 / 4.19 |
| GSB law | 256 | 9.75 / 5.37 | 9.04 / 5.03 |
| GSB medicine | 256 | 3.49 / 1.75 | 3.18 / 1.55 |
| GSB military | 256 | 3.43 / 1.54 | 3.06 / 1.40 |
| Golos crowd RU | 256 | 2.84 / 0.66 | 2.92 / 0.72 |
| Golos far-field RU | 256 | 7.98 / 2.59 | 9.10 / 3.03 |
| Lesbos Greek | 230 | 94.78 / 71.14 | 93.55 / 71.66 |
| Monsoon India | 256 | 4.12 / 1.92 | 3.78 / 1.80 |
| NST Danish | 256 | 26.49 / 12.51 | 11.59 / 4.40 |
| NST Swedish | 256 | 16.57 / 6.87 | 12.36 / 3.55 |
| VoxPopuli CS | 256 | 7.32 / 3.93 | 7.39 / 3.98 |
| VoxPopuli ES | 256 | 6.07 / 4.25 | 6.20 / 4.34 |
| VoxPopuli HU | 256 | 12.00 / 4.10 | 11.05 / 3.87 |
| VoxPopuli IT | 256 | 11.37 / 8.71 | 11.82 / 9.56 |
| VoxPopuli NL | 256 | 9.50 / 5.67 | 9.56 / 5.63 |
| VoxPopuli PL | 256 | 6.48 / 3.42 | 6.24 / 3.41 |
| VoxPopuli RO | 256 | 11.48 / 4.25 | 11.20 / 4.16 |

[Full precision](../evidence/domains-r3-20260908/scores.csv) · [Counts](../evidence/domains-r3-20260908/numeric-evidence.jsonl.gz) · [Independent scoring audit](../evidence/domains-r3-20260908/hypotheses-audit.json)

## Pooled comparisons

| Comparison | Clips | Parakeet errors / words | WER | Orukeet errors / words | WER | Wins / partitions |
|:--|--:|--:|--:|--:|--:|--:|
| FLEURS, 25 languages | 20146 | 46,442 / 421,870 | 11.01 | 41,521 / 421,715 | 9.85 | 23 / 25 |
| Accents/domains, 25 languages | 12006 | 43,939 / 262,747 | 16.72 | 40,068 / 262,698 | 15.25 | 36 / 47 |
| Accents/domains, English | 5120 | 9,032 / 94,993 | 9.51 | 8,399 / 94,993 | 8.84 | 20 / 20 |
