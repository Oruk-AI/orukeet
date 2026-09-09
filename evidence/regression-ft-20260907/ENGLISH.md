# Quick English comparison

All 20 English splits in the latest matched diagnostic; 256 fixed recordings per split. Existing verified error counts aggregated; no new inference or full-corpus rerun.

Word error rate (WER), with standard English normalization and identical audio/decoding for all models.

| Sample | Clips | Base Parakeet | Previous Orukeet | Fine-tuned Orukeet |
|:--|--:|--:|--:|--:|
| All English | 5,120 | 10.84% | 10.82% | 10.13% |
| Training-exposed checks | 4,352 | 9.16% | 9.25% | 8.42% |
| Controls excluded from this fine-tune | 768 | 17.91% | 17.41% | 17.32% |

Pooled WER falls by 6.61% relative to Parakeet and 6.41% relative to previous Orukeet.

The fine-tuned checkpoint beats Parakeet on 20/20 sampled splits and previous Orukeet on 18/20. Against previous Orukeet, humanities rises from 7.86% to 7.88% WER and EuroSpeech English rises from 24.93% to 25.23%.

4,352 training-exposed recordings and 768 controls excluded from this fine-tune. This is not a newly unseen benchmark.

| English split | Exposure | Parakeet WER | Previous WER | Fine-tuned WER |
|:--|:--|--:|--:|--:|
| eurospeech_en | control | 25.70% | 24.93% | 25.23% |
| gigaspeechbench_agr_en | trained | 6.79% | 6.86% | 6.30% |
| gigaspeechbench_ait_en | trained | 10.54% | 10.95% | 9.72% |
| gigaspeechbench_art_en | trained | 6.07% | 6.46% | 5.35% |
| gigaspeechbench_bio_en | trained | 6.74% | 6.86% | 6.15% |
| gigaspeechbench_chn_en | trained | 15.38% | 15.91% | 14.52% |
| gigaspeechbench_ecm_en | trained | 8.30% | 8.39% | 7.80% |
| gigaspeechbench_eng_en | trained | 6.15% | 6.63% | 4.81% |
| gigaspeechbench_ent_en | trained | 10.78% | 10.22% | 9.15% |
| gigaspeechbench_fin_en | trained | 7.49% | 7.15% | 6.89% |
| gigaspeechbench_hum_en | trained | 8.30% | 7.86% | 7.88% |
| gigaspeechbench_ind_en | trained | 8.54% | 8.69% | 7.75% |
| gigaspeechbench_jpn_en | control | 19.91% | 19.53% | 18.09% |
| gigaspeechbench_law_en | trained | 11.13% | 11.05% | 10.60% |
| gigaspeechbench_med_en | trained | 5.16% | 5.12% | 4.97% |
| gigaspeechbench_mil_en | trained | 5.99% | 6.07% | 5.51% |
| gigaspeechbench_phl_en | trained | 14.00% | 13.78% | 13.10% |
| gigaspeechbench_sct_en | trained | 22.66% | 24.09% | 20.73% |
| gigaspeechbench_sgp_en | trained | 14.94% | 15.03% | 13.99% |
| monsoon_en_in | control | 4.95% | 4.82% | 4.66% |

Exact model hashes, integer counts, CER, and legacy-normalized results are in [english-quick.json](english-quick.json). The full diagnostic and provenance are in [the run report](../../training/regression_ft/README.md).
