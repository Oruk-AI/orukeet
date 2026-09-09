# Focused LibriSpeech pass: r1

One pass over 2,939 test-other recordings, 56 updates, peak LR `3e-6`, final LR `3e-7`. Training took 80.44 seconds. All 12,288 fitted Gabor rows remained unchanged.

Test-other was used for training and candidate selection. French and Greek were re-evaluated as controls for this pass. All three models were freshly decoded under matched settings. WER (%):

| Split | Parakeet | Canonical Orukeet | r1 |
|---|---:|---:|---:|
| LibriSpeech test-other | 3.13 | 3.26 | 3.22 |
| FLEURS French | 4.70 | 5.03 | 5.16 |
| FLEURS Greek | 21.04 | 31.37 | 31.38 |

r1 reduced test-other WER from 3.26% to 3.22%, remaining above Parakeet at 3.13%. French increased to 5.16%; Greek was 31.39%. This candidate was continued in r2. Canonical release weights remain unchanged.
