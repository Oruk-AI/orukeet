# OpenWhispr application speed

Optimized Orukeet returns exactly the same results as the previous ONNX export on all **640 clips**. The 160 freshly rerun results for each control also match their earlier results: **960 comparisons, zero differences**. The full-suite WER remains **11.40% for Orukeet versus 11.93% for stock Parakeet**, with every selected output retained.

## Paired warm transcription

| Same 160 clips | Stock Parakeet | Previous Orukeet | Optimized Orukeet |
| --- | ---: | ---: | ---: |
| Median call | 428.4 ms | 431.6 ms | 389.7 ms |
| 95th percentile | 1097.3 ms | 1100.0 ms | 1025.6 ms |
| Total processing time | 83.45 s | 83.07 s | 75.85 s |
| Processing / audio duration | 0.08426 | 0.08388 | 0.07660 |

The comparison uses Apple M5 Max, macOS 26.4.1, Electron 41.10.5, sherpa-onnx 1.13.4 and ONNX Runtime 1.27.0 with four CPU threads. Each model processes the same **990.32 seconds** of audio: the first 16 clips per corpus in the fixed ten-corpus manifest. Calls rotate model order and include production normalization, 15-second segmentation, WebSocket exchange and recognition. Model loading and warm-up are excluded. The other 480 clips verify optimized-output equality and do not enter this timing table.

[Exact statistics, model/source hashes and protocol](application-speed.json) · [Transcript-free per-clip records](application-speed-clips.jsonl)

## Startup and preview checks

Explicitly selecting the verified NeMo decoder reduced median process start plus warm-up from **2213 to 1273 ms** for previous Orukeet and **2262 to 1275 ms** for stock Parakeet across five starts per setting. These starts use cached model files and precede the graph optimization. All 20 subsequent speech checks return identical transcripts. [Startup records](startup-probe.json).

The production recording test removes **one extra preview decode at stop** in each of three runs. Each run still decodes the complete recording once, and all six final transcripts match. Stop-to-saved medians are **868 ms before and 861 ms after**; history is polled every 50 ms. This test isolates the preview change using the previous encoder. [Recording records](live-stop-summary.json).

Runtime changes were uncommitted during measurement. The recorded source hashes identify the executed implementation.
