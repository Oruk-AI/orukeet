# Regression fine-tuning diagnostics

Matched fixed samples, scored with standard English normalization for English and the existing multilingual normalizer elsewhere.

| Sample | Recordings | Base Parakeet WER | Previous Orukeet WER | Candidate WER |
|:--|--:|--:|--:|--:|
| nontraining control | 5,888 | 21.73% | 19.51% | 20.46% |
| training exposed fit check | 6,118 | 12.54% | 12.12% | 10.85% |

The training-exposed sample measures fit. The control sample contains recordings excluded from this training run; it is not a newly collected unseen benchmark.

| Split | Role | Base WER | Previous WER | Candidate WER |
|:--|:--|--:|--:|--:|
| eurospeech_de | control | 15.06% | 13.48% | 12.36% |
| eurospeech_en | control | 25.70% | 24.93% | 25.23% |
| eurospeech_et | control | 38.28% | 29.90% | 30.22% |
| eurospeech_fi | control | 18.67% | 17.52% | 17.55% |
| eurospeech_fr | control | 20.00% | 15.14% | 15.21% |
| eurospeech_hr | control | 13.26% | 13.04% | 13.29% |
| eurospeech_lt | control | 39.43% | 33.77% | 35.78% |
| eurospeech_lv | control | 57.81% | 43.31% | 46.42% |
| eurospeech_mt | control | 40.74% | 38.72% | 41.41% |
| eurospeech_pt | control | 23.31% | 23.44% | 24.25% |
| eurospeech_sk | control | 18.61% | 15.18% | 17.48% |
| eurospeech_sl | control | 51.65% | 48.74% | 54.17% |
| eurospeech_uk | control | 15.22% | 14.06% | 17.73% |
| gigaspeechbench_jpn_en | control | 19.91% | 19.53% | 18.09% |
| golos_farfield_ru | control | 8.54% | 8.72% | 10.41% |
| monsoon_en_in | control | 4.95% | 4.82% | 4.66% |
| voxpopuli_cs | control | 8.15% | 7.99% | 8.30% |
| voxpopuli_es | control | 6.12% | 5.88% | 6.29% |
| voxpopuli_hu | control | 13.81% | 13.19% | 13.35% |
| voxpopuli_it | control | 11.58% | 11.79% | 11.97% |
| voxpopuli_nl | control | 10.42% | 10.36% | 10.62% |
| voxpopuli_pl | control | 6.52% | 6.50% | 6.41% |
| voxpopuli_ro | control | 11.99% | 11.88% | 11.68% |
| eurospeech_bg | fit | 14.76% | 14.65% | 13.78% |
| eurospeech_el | fit | 26.07% | 18.70% | 18.32% |
| eurospeech_it | fit | 10.64% | 12.28% | 12.34% |
| gigaspeechbench_agr_en | fit | 6.79% | 6.86% | 6.30% |
| gigaspeechbench_ait_en | fit | 10.54% | 10.95% | 9.72% |
| gigaspeechbench_art_en | fit | 6.07% | 6.46% | 5.35% |
| gigaspeechbench_bio_en | fit | 6.74% | 6.86% | 6.15% |
| gigaspeechbench_chn_en | fit | 15.38% | 15.91% | 14.52% |
| gigaspeechbench_ecm_en | fit | 8.30% | 8.39% | 7.80% |
| gigaspeechbench_eng_en | fit | 6.15% | 6.63% | 4.81% |
| gigaspeechbench_ent_en | fit | 10.78% | 10.22% | 9.15% |
| gigaspeechbench_fin_en | fit | 7.49% | 7.15% | 6.89% |
| gigaspeechbench_hum_en | fit | 8.30% | 7.86% | 7.88% |
| gigaspeechbench_ind_en | fit | 8.54% | 8.69% | 7.75% |
| gigaspeechbench_law_en | fit | 11.13% | 11.05% | 10.60% |
| gigaspeechbench_med_en | fit | 5.16% | 5.12% | 4.97% |
| gigaspeechbench_mil_en | fit | 5.99% | 6.07% | 5.51% |
| gigaspeechbench_phl_en | fit | 14.00% | 13.78% | 13.10% |
| gigaspeechbench_sct_en | fit | 22.66% | 24.09% | 20.73% |
| gigaspeechbench_sgp_en | fit | 14.94% | 15.03% | 13.99% |
| golos_crowd_ru | fit | 3.14% | 2.99% | 3.76% |
| lesbos_el | fit | 96.11% | 93.63% | 93.63% |
| nst_da_da | fit | 33.41% | 33.14% | 12.85% |
| nst_sv_sv | fit | 21.11% | 19.96% | 13.89% |

Paired uncertainty intervals, character error rates, cluster counts, exact model identities, and numeric evidence are retained in the accompanying JSON files.
