# Orukeet in OpenWhispr: shared ONNX runtime

Orukeet r3 INT8 scores **11.40% WER versus 11.93% for stock Parakeet TDT v3 INT8** through the same OpenWhispr file-transcription path. This is a **4.5% relative reduction** in word errors across 640 English clips from ten corpora. Orukeet has lower WER in 7 of ten corpora.

## Recognition

| Corpus | Clips | Parakeet WER | Orukeet WER |
| --- | ---: | ---: | ---: |
| AMI | 64 | 17.73% | 17.38% |
| Common Voice | 64 | 10.96% | 12.13% |
| Earnings22 | 64 | 31.45% | 30.23% |
| GigaSpeech | 64 | 10.02% | 9.26% |
| L2-ARCTIC | 64 | 6.88% | 6.21% |
| LibriSpeech test-clean | 64 | 1.83% | 2.35% |
| LibriSpeech test-other | 64 | 4.86% | 4.69% |
| SpeechOcean test | 64 | 28.81% | 21.79% |
| SPGISpeech | 64 | 13.06% | 12.45% |
| VoxPopuli | 64 | 7.56% | 7.90% |
| **Pooled** | **640** | **11.93%** | **11.40%** |

The fixed suite contains 64 clips per corpus, 65.8 minutes of audio and 9,783 normalized reference words. Selection uses the existing `orukeet-app-eval-v1` hash ordering, nonempty references and 0–120-second inputs. Both models receive every selected recording. Empty transcripts remain in scoring: 23 for Parakeet and 21 for Orukeet. Runtime exceptions: 0 and 0 respectively.

The same Whisper English normalizer and Levenshtein word counts score both outputs. Pooled WER sums errors and reference words. A paired utterance bootstrap with 10,000 resamples within corpus gives a 95% interval of [-0.96, -0.13] percentage points for Orukeet minus Parakeet.

## Runtime

| Warm file transcription, paired 160-clip subset, M5 Max | Parakeet | Previous Orukeet export | Optimized Orukeet |
| --- | ---: | ---: | ---: |
| Median call | 428 ms | 432 ms | **390 ms** |
| 95th percentile | 1097 ms | 1100 ms | **1026 ms** |
| Processing / audio duration | 0.08426 | 0.08388 | **0.07660** |

The optimized export reduces median latency by 9.7% and total processing time by 8.7% versus the previous Orukeet export. Against stock Parakeet, median latency is 9.0% lower and total processing time is 9.1% lower. These timings cover the first 16 clips per corpus in the fixed suite: 160 clips and 990.3 seconds of audio. The accuracy table above uses all 640 clips. Every optimized transcript and status matches the previous export across all 640 clips; both 160-clip control arms also match their previous results, giving 960 identical comparisons.

Measured on Apple M5 Max, macOS 26.4.1, Electron 41.10.5, sherpa-onnx 1.13.4 and ONNX Runtime 1.27.0. All three artifacts use the same offline CPU worker with four threads and the same 15-second segmentation. Timings include production audio normalization, segmentation, WebSocket calls and recognition. Each model has a warm process; calls run sequentially, rotating model order by clip. Loading is excluded from warm call times. File-transcription timing excludes microphone endpointing and the preview timer.

Passing the verified NeMo model type also avoids a second encoder load during worker startup. Across five paired fresh-worker starts per model with warm filesystem caches, median load-and-warm time fell from 2.21 s to 1.27 s for Orukeet and from 2.26 s to 1.27 s for Parakeet. This startup experiment uses the previous graph in both arms, isolating the application change.

Stopping offline dictation now skips an extra preview-only decode while preserving full-audio final transcription. Six actual renderer recordings verified one redundant post-stop request in each previous-code run and none in the new-code runs. Their stop-to-save medians were 868 ms and 861 ms; this check establishes request removal and correct saved text.

The graph optimization changes 24 depthwise convolutions to equivalent arithmetic supported by the existing runtime. [Encoder-only measurements](../../evidence/speed20260910/README.md) on a Linux Xeon VM show 1.47–1.51× speedup; these measure encoder execution separately from application latency.

[Current application measurements](../../evidence/speed20260910/application-speed.md) · [Original 640-clip scores](../../evidence/onnx-r3-20260910/app-paired-640-scores.json) · [Original conversion measurements](../../evidence/onnx-r3-20260910/app-paired-640-receipt.json)

## Integration checks

The production Mac model manager passes load, repeated process reuse, concurrent requests, silence, multilingual float-WAV normalization, long-audio segmentation, cancellation and recovery. The complete renderer test covers the Oruk picker and model-card link, anonymous public download, microphone capture from a fixed WAV, real recognition, saved history, capture cancellation and a successful next recording.

The [optimized-release validation record](../../evidence/speed20260910/release-validation.md) records platform checks and the current picker screenshot. The [original conversion checks](../../evidence/onnx-r3-20260910/application-validation.md) are retained separately.

[Historical native Q8/Metal measurements](APP_BENCHMARKS_NATIVE.md) · [General ASR benchmarks](../../docs/current-checkpoint-benchmarks.md)
