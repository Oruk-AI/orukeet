> Historical native Q8/Metal integration, measured on 8 September 2026. The current PR uses the [shared ONNX path](APP_BENCHMARKS.md).

# Orukeet in OpenWhispr: paired recognition

Orukeet r3 scores **6.89% WER versus 11.93% for stock Parakeet TDT v3** through OpenWhispr's file-transcription pipeline: **42.2% fewer word errors**. The comparison includes 640 English clips, 64 from each of ten corpora, covering 65.8 minutes and 9,783 reference words. Orukeet has lower WER in 10 of ten corpora.

| Corpus | Clips | Orukeet WER | Parakeet WER |
| --- | ---: | ---: | ---: |
| AMI | 64 | 13.12% | 17.73% |
| Common Voice | 64 | 7.31% | 10.96% |
| Earnings22 | 64 | 11.73% | 31.45% |
| GigaSpeech | 64 | 9.12% | 10.02% |
| L2-ARCTIC | 64 | 5.20% | 6.88% |
| LibriSpeech test-clean | 64 | 1.57% | 1.83% |
| LibriSpeech test-other | 64 | 3.24% | 4.86% |
| SpeechOcean test | 64 | 18.16% | 28.81% |
| SPGISpeech | 64 | 4.26% | 13.06% |
| VoxPopuli | 64 | 6.76% | 7.56% |
| **Pooled** | **640** | **6.89%** | **11.93%** |

Both backends run in the integration against OpenWhispr/openwhispr v1.9.2. Orukeet uses r3 Q8 on Metal; stock Parakeet uses the release's INT8 ONNX models and unchanged sherpa-onnx CPU worker with four threads. Original encoded audio passes through app normalization, segmentation, worker IPC and result handling. Orukeet's pause-aware segmentation is included in the result.

The fixed hash selection, 0–120-second inputs, original references and audio hashes match the preceding app comparison. Both models receive every clip, in alternating order, with warm workers and serialized calls. The same Whisper English normalizer scores both outputs. Empty transcripts remain in the scores: two for Orukeet and 23 for Parakeet. These return the app's `No audio detected` response; no other runtime errors occurred.

The pooled difference is -5.04 percentage points. A paired utterance bootstrap with 10,000 resamples within corpus gives a 95% interval of [-7.13, -3.20] points.

| Warm file-transcription timing on M5 Max | Orukeet | Parakeet |
| --- | ---: | ---: |
| Median call | 58 ms | 481 ms |
| 95th percentile | 96 ms | 1450 ms |
| Processing / audio duration | 0.0097 | 0.0981 |

Timings include normalization, segmentation and recognition after loading. [Live recording measurements](#live-recording) use the production renderer separately.

[Numerical counts](../../evidence/r3-promotion-20260908/paired-app-scores.json) · [Implementation hashes](../../evidence/r3-promotion-20260908/paired-app-provenance.json).

## Live recording

| Warm recording median, M5 Max | Orukeet | Stock Parakeet TDT v3 |
| --- | ---: | ---: |
| First live preview | 1.55 s | 1.64 s |
| Preview processing | 40 ms | 131 ms |
| Stop to saved transcript | 115 ms | 945 ms |

Three recordings per model use the production renderer, MediaRecorder, normalization, model worker and SQLite history. The preview timer is 1.5 seconds. The Orukeet backend is Q8/Metal; stock Parakeet is the v1.9.2 INT8 ONNX/sherpa-onnx CPU implementation with four threads. These measurements compare the two application paths on Apple M5 Max. That historical PR build used the r3 model and native decoding path. The current PR uses the shared ONNX runtime described in [the current measurements](APP_BENCHMARKS.md).

[Model card](MODEL_CARD.md) · [General ASR benchmarks](../../docs/current-checkpoint-benchmarks.md)
