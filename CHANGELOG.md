# Changelog

## 0.1.0 — 2026-09-09

- Release the final r3 checkpoint as NeMo, native Q8 and native F16, with pinned hashes.
- Publish inference and fine-tuning code, all 74 paired benchmark scores, the technical report and BibTeX.
- Add the OpenWhispr integration with Orukeet recommended, persistent native inference and automatic hardware selection.
- Apply the current Oruk Signal branding throughout the release.


## r3 release selection — 2026-09-08

- Select the manuscript's r3 checkpoint for the canonical NeMo, Q8 and F16 files. Preserve all 12,288 fitted Gabor kernels.
- Pin the installer and private OpenWhispr integration to the new exports, with a distinct cache filename for upgrades.
- Report all 74 paired benchmark splits: 61 wins, including 23 of 25 FLEURS languages; pooled FLEURS WER is 9.85% versus Parakeet's 11.01%.
- Keep the repositories and release preparations private.


## 0.1.0rc1 · Frozen Gabor Orukeet

- Make **FT-4035** the sole current Orukeet model after 4,035 additional low-learning-rate updates. All 12,288 fitted Gabor kernels remain fixed.
- Pin the NeMo source and its Q8/F16 exports to the same audited checkpoint.
- Compare Orukeet and base Parakeet on 12,006 matched clips across 47 splits and 25 languages: pooled WER is 16.52% versus 17.97%. English WER is 10.13% versus 10.84% across 5,120 clips, with lower WER on all 20 English splits.
- Publish every split's WER/CER in the prepared score files and seven-page technical report. Retain the preceding R15-0100 comparison on 327,888 clips / 756.02 hours with its original model identity and methods.
- Verify the current native weights on Mac Metal/CPU and Windows/Linux CPU, with an additional Linux CUDA fixture. Keep the OpenWhispr v1.9.2 integration and its separate application tests private.
- Update release guides, source credits, citation metadata and affiliation logos for the current model.

## Earlier preparation history

# Changes

## Private preparation — September 6, 2026

- Document Orukeet as a surgical refinement of NVIDIA Parakeet through further multilingual/accent training and Gabor-kernel experiments.
- Fit all 24,576 eligible kernels, replace the closest half, freeze those functions and audit training of the remaining parameters.
- R15-0100 passes all source accuracy gates with every fitted row fixed; source statistics reproduce exactly on CPU.
- Its Q8 export fails three native development guards. F16 passes development but fails larger native accuracy. Both exports pass Mac/Windows/Linux fixture checks; F16 also passes real Electron lifecycle checks.
- Retain all failed recovery runs and exact ancestry. Canonical weights remain pre-surgery while deployment qualification is incomplete.
- Prepare a four-page LaTeX technical report in official NeurIPS preprint format with the Oruk logo.
- Keep the actual OpenWhispr/openwhispr integration separate and private.

## 0.1.0rc1 — private review, September 5, 2026

- Name the selected multilingual Parakeet adaptation Orukeet.
- Ship persistent native inference, explicit verified installation and Q8 export.
- Preserve the exact training lineage, evaluation membership and failed gate.
- Retain historical Knuckles92/OpenWhisper v2.6.0 app QA and benchmarks; these do not validate the distinct OpenWhispr/openwhispr project.
- Prepare a model card, technical report, release research and license inventory.

Nothing in this candidate has been published publicly.
