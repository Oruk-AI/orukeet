# Historical FT-4035 and R15 benchmark scores

FT-4035 is the parent of the r3 release checkpoint. [Current r3 results](current-checkpoint-benchmarks.md) contain all 74 paired release scores. This page preserves the preceding measurements. Across the 25-language comparison, **Orukeet scores 16.52% pooled WER versus Parakeet’s 17.97%**, an 8.0% relative reduction. English WER is **10.13% for Orukeet versus 10.84% for Parakeet**, a 6.6% relative reduction. It improves WER on 35 of 47 splits, including all 20 English splits. R15-0100 is the preceding checkpoint; both retain the same 12,288 frozen Gabor kernels.

| Matched comparison | Clips | Parakeet WER | R15-0100 WER | FT-4035 WER |
|:--|--:|--:|--:|--:|
| All 47 splits · 25 languages | 12,006 | 17.97% | 16.48% | 16.52% |
| All 20 English splits | 5,120 | 10.84% | 10.82% | 10.13% |

## Evaluation method

Scores use matched NeMo greedy decoding with FP32 weights and BF16 CUDA autocast. English uses standard Whisper text normalization; other languages use the recorded multilingual normalizer. WER and CER are percentages, computed from summed edit counts and reference lengths.

The FT-4035 comparison covers 47 splits and 25 languages: 256 fixed clips per split and all 230 Lesbos clips, totaling 12,006 clips. FT-4035 was fine-tuned on 24 of these splits; 6,118 comparison clips were included in that run. Greek and Italian EuroSpeech use audited human transcript spans. The earlier 327,888-clip comparison measures R15-0100 against Parakeet, using its recorded clip membership and original EuroSpeech references. Full tables retain every split from both comparisons.

## Current matched comparison: 12,006 clips

Every split is listed below. Cells contain WER / CER (%).

| Split | Clips | Parakeet WER / CER | Orukeet R15 WER / CER | Orukeet FT-4035 WER / CER |
|:--|--:|--:|--:|--:|
| eurospeech_bg | 256 | 14.76 / 6.93 | 14.65 / 7.28 | 13.78 / 6.58 |
| eurospeech_de | 256 | 15.06 / 9.94 | 13.48 / 9.24 | 12.36 / 8.44 |
| eurospeech_el | 256 | 26.07 / 8.48 | 18.70 / 7.59 | 18.32 / 7.68 |
| eurospeech_en | 256 | 25.70 / 17.90 | 24.93 / 17.38 | 25.23 / 17.57 |
| eurospeech_et | 256 | 38.28 / 14.68 | 29.90 / 11.52 | 30.22 / 12.69 |
| eurospeech_fi | 256 | 18.67 / 7.85 | 17.52 / 7.49 | 17.55 / 7.72 |
| eurospeech_fr | 256 | 20.00 / 11.85 | 15.14 / 9.59 | 15.21 / 9.76 |
| eurospeech_hr | 256 | 13.26 / 8.69 | 13.04 / 8.45 | 13.29 / 8.65 |
| eurospeech_it | 256 | 10.64 / 6.33 | 12.28 / 8.01 | 12.34 / 8.22 |
| eurospeech_lt | 256 | 39.43 / 16.13 | 33.77 / 14.69 | 35.78 / 16.67 |
| eurospeech_lv | 256 | 57.81 / 26.65 | 43.31 / 16.56 | 46.42 / 18.93 |
| eurospeech_mt | 256 | 40.74 / 19.23 | 38.72 / 17.99 | 41.41 / 19.68 |
| eurospeech_pt | 256 | 23.31 / 17.53 | 23.44 / 17.77 | 24.25 / 18.62 |
| eurospeech_sk | 256 | 18.61 / 8.23 | 15.18 / 7.09 | 17.48 / 9.37 |
| eurospeech_sl | 256 | 51.65 / 17.23 | 48.74 / 15.59 | 54.17 / 17.97 |
| eurospeech_uk | 256 | 15.22 / 8.84 | 14.06 / 7.98 | 17.73 / 11.33 |
| gigaspeechbench_agr_en | 256 | 6.79 / 3.98 | 6.86 / 3.99 | 6.30 / 3.44 |
| gigaspeechbench_ait_en | 256 | 10.54 / 5.01 | 10.95 / 5.42 | 9.72 / 4.61 |
| gigaspeechbench_art_en | 256 | 6.07 / 3.10 | 6.46 / 3.46 | 5.35 / 2.62 |
| gigaspeechbench_bio_en | 256 | 6.74 / 1.95 | 6.86 / 2.01 | 6.15 / 1.81 |
| gigaspeechbench_chn_en | 256 | 15.38 / 9.25 | 15.91 / 9.74 | 14.52 / 8.61 |
| gigaspeechbench_ecm_en | 256 | 8.30 / 4.39 | 8.39 / 4.36 | 7.80 / 4.07 |
| gigaspeechbench_eng_en | 256 | 6.15 / 2.43 | 6.63 / 2.47 | 4.81 / 2.01 |
| gigaspeechbench_ent_en | 256 | 10.78 / 6.85 | 10.22 / 6.43 | 9.15 / 5.73 |
| gigaspeechbench_fin_en | 256 | 7.49 / 3.57 | 7.15 / 3.31 | 6.89 / 3.19 |
| gigaspeechbench_hum_en | 256 | 8.30 / 4.76 | 7.86 / 4.41 | 7.88 / 4.31 |
| gigaspeechbench_ind_en | 256 | 8.54 / 3.41 | 8.69 / 3.52 | 7.75 / 2.74 |
| gigaspeechbench_jpn_en | 256 | 19.91 / 12.05 | 19.53 / 12.17 | 18.09 / 11.08 |
| gigaspeechbench_law_en | 256 | 11.13 / 5.57 | 11.05 / 5.52 | 10.60 / 5.35 |
| gigaspeechbench_med_en | 256 | 5.16 / 1.91 | 5.12 / 1.84 | 4.97 / 1.72 |
| gigaspeechbench_mil_en | 256 | 5.99 / 1.76 | 6.07 / 1.89 | 5.51 / 1.58 |
| gigaspeechbench_phl_en | 256 | 14.00 / 8.37 | 13.78 / 8.35 | 13.10 / 7.73 |
| gigaspeechbench_sct_en | 256 | 22.66 / 14.66 | 24.09 / 16.29 | 20.73 / 13.43 |
| gigaspeechbench_sgp_en | 256 | 14.94 / 9.48 | 15.03 / 9.56 | 13.99 / 8.61 |
| golos_crowd_ru | 256 | 3.14 / 0.66 | 2.99 / 0.60 | 3.76 / 0.75 |
| golos_farfield_ru | 256 | 8.54 / 2.59 | 8.72 / 2.56 | 10.41 / 3.53 |
| lesbos_el | 230 | 96.11 / 71.35 | 93.63 / 72.41 | 93.63 / 73.10 |
| monsoon_en_in | 256 | 4.95 / 2.46 | 4.82 / 2.39 | 4.66 / 2.36 |
| nst_da_da | 256 | 33.41 / 19.25 | 33.14 / 19.35 | 12.85 / 4.64 |
| nst_sv_sv | 256 | 21.11 / 12.45 | 19.96 / 11.94 | 13.89 / 3.66 |
| voxpopuli_cs | 256 | 8.15 / 3.93 | 7.99 / 4.16 | 8.30 / 4.11 |
| voxpopuli_es | 256 | 6.12 / 4.25 | 5.88 / 4.01 | 6.29 / 4.33 |
| voxpopuli_hu | 256 | 13.81 / 4.11 | 13.19 / 3.91 | 13.35 / 3.92 |
| voxpopuli_it | 256 | 11.58 / 8.71 | 11.79 / 8.90 | 11.97 / 9.62 |
| voxpopuli_nl | 256 | 10.42 / 5.67 | 10.36 / 5.58 | 10.62 / 5.80 |
| voxpopuli_pl | 256 | 6.52 / 3.42 | 6.50 / 3.51 | 6.41 / 3.49 |
| voxpopuli_ro | 256 | 11.99 / 4.39 | 11.88 / 4.25 | 11.68 / 4.26 |

## Preceding R15 comparison: 327,888 clips

Every split is listed below. Cells contain WER / CER (%).

| Split | Clips | Parakeet WER / CER | Orukeet R15 WER / CER |
|:--|--:|--:|--:|
| eurospeech_bg | 6,892 | 15.11 / 7.25 | 14.76 / 7.48 |
| eurospeech_de | 4,872 | 15.75 / 10.50 | 13.81 / 9.50 |
| eurospeech_el | 6,730 | 101.45 / 78.15 | 100.71 / 79.19 |
| eurospeech_en | 9,268 | 26.07 / 18.38 | 25.07 / 17.73 |
| eurospeech_et | 3,554 | 38.68 / 15.04 | 30.17 / 11.61 |
| eurospeech_fi | 5,422 | 17.59 / 7.21 | 16.15 / 6.77 |
| eurospeech_fr | 744 | 19.22 / 11.49 | 14.74 / 9.31 |
| eurospeech_hr | 15,638 | 13.43 / 8.89 | 13.07 / 8.68 |
| eurospeech_it | 8,714 | 64.73 / 48.33 | 64.84 / 48.43 |
| eurospeech_lt | 7,319 | 38.58 / 16.32 | 32.85 / 14.52 |
| eurospeech_lv | 3,343 | 57.61 / 25.55 | 42.64 / 16.16 |
| eurospeech_mt | 3,446 | 39.98 / 18.02 | 38.31 / 17.31 |
| eurospeech_pt | 7,501 | 22.08 / 16.45 | 22.05 / 16.43 |
| eurospeech_sk | 6,915 | 18.05 / 7.98 | 15.34 / 7.22 |
| eurospeech_sl | 3,585 | 52.86 / 17.61 | 50.19 / 16.09 |
| eurospeech_uk | 3,239 | 15.58 / 9.11 | 14.74 / 8.63 |
| gigaspeechbench_agr_en | 6,665 | 6.47 / 3.76 | 6.47 / 3.77 |
| gigaspeechbench_ait_en | 5,468 | 9.60 / 4.60 | 9.99 / 4.89 |
| gigaspeechbench_art_en | 5,712 | 6.11 / 2.97 | 6.12 / 3.03 |
| gigaspeechbench_bio_en | 5,297 | 6.52 / 1.91 | 6.63 / 1.92 |
| gigaspeechbench_chn_en | 6,308 | 17.22 / 10.12 | 17.07 / 10.08 |
| gigaspeechbench_ecm_en | 5,659 | 9.08 / 4.79 | 9.10 / 4.80 |
| gigaspeechbench_eng_en | 6,648 | 5.66 / 2.34 | 6.16 / 2.46 |
| gigaspeechbench_ent_en | 8,583 | 10.00 / 6.43 | 9.92 / 6.41 |
| gigaspeechbench_fin_en | 6,037 | 6.91 / 3.36 | 6.96 / 3.41 |
| gigaspeechbench_hum_en | 4,971 | 6.94 / 3.66 | 6.87 / 3.65 |
| gigaspeechbench_ind_en | 5,503 | 7.87 / 3.08 | 7.84 / 3.05 |
| gigaspeechbench_jpn_en | 9,310 | 21.25 / 12.66 | 21.19 / 12.63 |
| gigaspeechbench_law_en | 7,273 | 10.02 / 5.52 | 10.14 / 5.62 |
| gigaspeechbench_med_en | 5,168 | 5.44 / 1.96 | 5.51 / 1.99 |
| gigaspeechbench_mil_en | 5,224 | 5.99 / 1.75 | 6.10 / 1.78 |
| gigaspeechbench_phl_en | 8,637 | 12.01 / 7.43 | 12.05 / 7.49 |
| gigaspeechbench_sct_en | 12,829 | 26.14 / 17.05 | 26.63 / 17.48 |
| gigaspeechbench_sgp_en | 9,480 | 13.69 / 8.39 | 14.06 / 8.63 |
| golos_crowd_ru | 9,896 | 3.42 / 0.77 | 3.48 / 0.78 |
| golos_farfield_ru | 1,915 | 7.25 / 2.16 | 7.17 / 2.06 |
| lesbos_el | 230 | 96.11 / 71.30 | 93.63 / 72.36 |
| monsoon_en_in | 2,102 | 5.00 / 2.50 | 4.83 / 2.42 |
| nst_da_da | 54,747 | 30.54 / 16.70 | 29.79 / 16.74 |
| nst_sv_sv | 27,638 | 23.19 / 13.71 | 22.68 / 13.90 |
| voxpopuli_cs | 1,103 | 9.22 / 4.71 | 8.97 / 4.57 |
| voxpopuli_es | 1,631 | 5.50 / 3.59 | 5.45 / 3.54 |
| voxpopuli_hu | 1,076 | 15.72 / 5.65 | 15.33 / 5.55 |
| voxpopuli_it | 1,257 | 11.98 / 8.74 | 11.97 / 8.68 |
| voxpopuli_nl | 1,230 | 11.25 / 6.32 | 11.12 / 6.28 |
| voxpopuli_pl | 1,691 | 7.54 / 4.20 | 7.32 / 3.99 |
| voxpopuli_ro | 1,418 | 12.24 / 4.87 | 11.83 / 4.82 |

## Model identities

- parakeet: `3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d`
- r15: `4295a6d820a40b99786331d1c7a6b6c328916c8329b23d39415b0649a5d42811`
- ft4035: `0ccfefcd1894871cb0850bd3c464adf5397752840de2a76d1d2d075c4141a945`

[FT-4035 checkpoint and audit](https://huggingface.co/oruk/orukeet/tree/30c6d16738f6f3edee70142c86cda41220c4aacc) · [Machine-readable scores](../evidence/benchmark-release-20260907/scores.json) · [Complete-partition CSV](../evidence/benchmark-release-20260907/complete.csv) · [Matched-comparison CSV](../evidence/benchmark-release-20260907/sampled.csv)

The historical 23,038-recording selection results and format-specific native measurements are retained in the [methods companion](technical-report.md).
