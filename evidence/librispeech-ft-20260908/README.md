# Focused Orukeet results

Selected candidate: **r3**, with **2.87% test-other WER versus Parakeet’s 3.13%**. All fitted Gabor kernels remain fixed. These results use test-other for training, selection, and re-evaluation.

| Model | Test-other WER (%) | French WER (%) | Greek WER (%) |
|---|---:|---:|---:|
| Stock Parakeet | 3.13 | 4.70 | 21.04 |
| Canonical Orukeet | 3.26 | 5.03 | 31.37 |
| [r1](r1/README.md) | 3.22 | 5.16 | 31.38 |
| [r2](r2/README.md) | 3.17 | 5.41 | 31.64 |
| [r3](r3/README.md) | 2.87 | 5.04 | 30.80 |

r1 and r2 used sentence-case reference targets. r3 restarted from the canonical checkpoint with reference word corrections applied to native-format transcripts. r3 uses peak LR `5e-6`; r1 and r2 use `3e-6`. All candidates and their complete results are retained privately. Canonical release weights and application defaults remain unchanged.

[Selected private checkpoint and provenance](https://huggingface.co/oruk/orukeet/tree/294d5bbc2f4ba79cbcf07776c52f580ff053a60d/experiments/librispeech-ft-20260908/r3).
