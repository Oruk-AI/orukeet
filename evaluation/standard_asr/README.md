# LibriSpeech and FLEURS comparison

This evaluation compares the released Orukeet FT-4035 NeMo checkpoint with NVIDIA Parakeet TDT 0.6B v3 on both complete LibriSpeech test partitions and all 25 supported FLEURS test languages. The [score table](../../docs/standard-asr-benchmarks.md) reports WER and CER for each partition and both models.

The runner uses identical mono 16 kHz audio, FP32 weights, BF16 CUDA autocast and matched greedy-batch TDT decoding. It retains every record and every empty hypothesis. Final scoring uses the pinned English and multilingual normalizers in [vendor/](vendor/). English normalization standardizes spelling, names and numbers. Multilingual normalization retains diacritics, expands digits by language and aligns compound boundaries; WER uses compound-aware edit distance. CER uses normalized strings before boundary alignment, including spaces. Corpus rates pool integer edit counts; language macros weight their specified languages equally.

Place the original LibriSpeech `test-clean` audio and `.trans.txt` files under `data/goal_v2_sources/libri/LibriSpeech/test-clean`, `test-other` under `data/librispeech/LibriSpeech/test-other`, and original FLEURS test FLAC files under `data/fleurs/<language>/test`, relative to `AUDIO_ROOT`. Use the recorded NeMo environment from the preceding source-model comparison. FLEURS references are fetched from the immutable dataset revision recorded in `prepare.py`; the scripts record reference-file and decoded-audio hashes.

```sh
python evaluation/standard_asr/prepare.py --audio-root "$AUDIO_ROOT" --output /tmp/orukeet-standard/prepared
python evaluation/standard_asr/run.py \
  --manifest /tmp/orukeet-standard/prepared/manifest.jsonl \
  --parakeet /path/to/parakeet-tdt-0.6b-v3.nemo \
  --orukeet /path/to/orukeet-v0.1.0rc1.nemo \
  --metric-code evaluation/unseen \
  --output /tmp/orukeet-standard/evaluation
```

The runner verifies both checkpoint hashes before inference. Resume accepts only predictions with matching checkpoint, manifest, runner and normalizer hashes. A decoded-audio hash check precedes inference. Out-of-memory recovery recursively reduces the batch size; it does not omit or shorten recordings.

The inference runner also emits diagnostic counts under the preceding evaluation's normalizer. Preserve these as `inference-comparison.json` and `inference-numeric-evidence.jsonl.gz` in the evidence directory. Place the unchanged `manifest.jsonl`, `parakeet.jsonl` and `orukeet.jsonl` in a private records directory, then generate the final scores:

```sh
python -m pip install -r evaluation/standard_asr/requirements-score.txt
python evaluation/standard_asr/rescore.py --private-records /path/to/private-records --evidence /path/to/evidence
python evaluation/standard_asr/audit_predictions.py --private-records /path/to/private-records --evidence /path/to/evidence
```

The scoring implementation is pinned to [source revision 48219c6](https://github.com/huggingface/open_asr_leaderboard/tree/48219c6028db0517d704600d92f31edfc96e8c23/normalizer). [Provenance](vendor/provenance.json) records source and vendored hashes, exact extracted definitions and the Apache-2.0 license. Compound alignment can change the reference word count separately for each model; all resulting denominators are retained. Scoring changes never alter predictions or checkpoint identities. For the release audit, pass `--upstream-root /path/to/pinned-source` to `audit_predictions.py`; it independently loads the original normalization definitions and verifies all 54 model/partition WERs with full-partition batch scoring.

`comparison.json` holds full-precision corpus results and checkpoint identities. `numeric-evidence.jsonl.gz` contains transcript-free per-record edit counts. `preparation.json` records the complete test membership and source revisions. To reproduce the release tables from the included counts:

```sh
python evaluation/standard_asr/build_materials.py
python scripts/build_neurips_report.py
python evaluation/standard_asr/build_materials.py --verify-pdf
```

Corpus sources: [LibriSpeech](https://www.openslr.org/12) and [FLEURS](https://arxiv.org/abs/2205.12446). Dataset audio and reference transcripts are obtained from the original providers; the release evidence contains counts and hashes.
