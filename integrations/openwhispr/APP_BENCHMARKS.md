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

| Warm file transcription, M5 Max | Parakeet | Orukeet |
| --- | ---: | ---: |
| Median call | 536 ms | 537 ms |
| 95th percentile | 1656 ms | 1584 ms |
| Processing / audio duration | 0.1101 | 0.1100 |

Measured on Apple M5 Max, macOS 26.4.1, using the current OpenWhispr main integration, Electron 41.10.5, sherpa-onnx 1.13.4 and ONNX Runtime 1.27.0. Both models use the unchanged offline CPU worker with four threads and the same 15-second segmentation. Timings include production audio normalization, segmentation, WebSocket calls and recognition.

Each model has a separate warm process. Calls are serialized and alternate which model goes first for each clip. Loading is recorded separately and excluded from these timings. This table measures file-transcription calls; microphone endpointing and the preview timer are separate application behavior.

[Numerical counts and timings](../../evidence/onnx-r3-20260910/app-paired-640-scores.json) · [Runtime provenance](../../evidence/onnx-r3-20260910/app-paired-640-receipt.json) · [Release checks](../../evidence/onnx-r3-20260910/application-validation.md)

## Integration checks

The production Mac model manager passes load, repeated process reuse, concurrent requests, silence, multilingual float-WAV normalization, long-audio segmentation, cancellation and recovery. The complete renderer test covers the Oruk picker and model-card link, anonymous public download, microphone capture from a fixed WAV, real recognition, saved history, capture cancellation and a successful next recording.

The [validation record](../../evidence/onnx-r3-20260910/application-validation.md) includes the final Windows, Linux and Mac CI outcomes and the picker screenshot.

[Historical native Q8/Metal measurements](APP_BENCHMARKS_NATIVE.md) · [General ASR benchmarks](../../docs/current-checkpoint-benchmarks.md)
