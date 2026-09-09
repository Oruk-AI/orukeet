# Focused LibriSpeech continuation: r2

Three passes from r1, 168 updates, peak LR `3e-6`, final LR `3e-7`. All 2,939 recordings appeared three times (8,817 presentations). Training took 217.90 seconds. All 12,288 fitted Gabor rows remained unchanged.

Test-other was used for training and candidate selection. French and Greek were re-evaluated as controls. All three models were freshly decoded under matched settings. WER (%):

| Split | Parakeet | Canonical Orukeet | r2 |
|---|---:|---:|---:|
| LibriSpeech test-other | 3.13 | 3.26 | 3.17 |
| FLEURS French | 4.70 | 5.03 | 5.41 |
| FLEURS Greek | 21.04 | 31.37 | 31.64 |

r2 reduced test-other WER to 3.17%, still above Parakeet at 3.13%. French and Greek increased relative to canonical Orukeet. The next attempt restarts from the canonical model with reference-corrected native-format targets. Canonical release weights remain unchanged.
