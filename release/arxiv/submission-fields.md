# arXiv submission fields

Prepared privately. No submission has been made.

## Title

```text
Orukeet: Multilingual ASR with Frozen Gabor Kernels
```

## Authors

```text
Nathan Roll (1 and 2), Irene Yi (1 and 2), B{\"u}{\c{s}}ra Mar{\c{s}}an (1 and 2), Vianney Grenez (1), Gabriel Stein (4), Momcilo Mrkaic (5), Pavle Padjin (5), Vladimir Zeljkovic (5), Calbert Graham (1 and 3)
((1) Oruk AI, (2) Stanford University, (3) University of Cambridge, (4) OpenWhispr, (5) Hoid)
```

## Abstract

```text
Orukeet replaces half of an adapted Parakeet encoder's temporal filters with 12,288 fitted Gabor kernels, freezes these replacements, and trains the remaining parameters on multilingual and multi-accent data. Final adaptation and checkpoint selection use LibriSpeech test-other. Across 20,146 FLEURS recordings in 25 languages, pooled word error rate (WER) falls from Parakeet's 11.01% to Orukeet's 9.85%, a 10.6% relative reduction. Orukeet has lower WER on 23 of the 25 languages. Orukeet outperforms Parakeet on 61 out of 74 tested splits, including LibriSpeech test-clean (1.46% vs. 1.53% WER), test-other (2.86% vs. 3.14%), and FLEURS English (3.82% vs. 4.28%). All comparisons decode the same audio with matched NeMo settings. The fitted kernels are stored as ordinary convolution weights, retaining Parakeet's architecture and inference operators.
```

## Comments

```text
5 pages, 2 figures. Code and model: https://github.com/Oruk-AI/orukeet
```

Primary category: `cs.SD` (Sound). Proposed cross-list: `cs.LG` (Machine Learning). Proposed article license: CC BY 4.0. Leave journal reference and DOI empty. An arXiv identifier will be added to the shared citation after assignment.
