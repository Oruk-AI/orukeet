# Orukeet runtime performance, 2026-09-21

On one Apple M5 Max running macOS 26.4.1, the FluidAudio buffer-reuse backport reduced warmed engine latency by **20.9–28.4% with the true INT8 encoder**, at fixed chunk concurrency 1. Exact transcripts matched on all 400 multilingual regression clips and every paired timing call. These are Mac measurements, not physical iPhone performance or memory qualification.

| Audio | Duration | Original 0.15.5 INT8 | Optimized INT8 | Lower latency |
|---|---:|---:|---:|---:|
| English | 4.32 s | 54.32 ms | 38.89 ms | 28.4% |
| Russian | 7.92 s | 67.96 ms | 51.20 ms | 24.7% |
| Italian | 14.34 s | 71.12 ms | 54.64 ms | 23.2% |
| Natural long recording | 29.95075 s | 240.63 ms | 190.32 ms | 20.9% |

The historical published greedy bundle, whose encoder uses LUT6 rather than INT8, showed corresponding latency reductions of 27.0%, 22.1%, 20.0%, and 18.3%. Both profiles retained identical text for all 144 paired calls per profile, including warmups.

| Optimized INT8 chunk concurrency, long recording | Latency | Process peak RSS, two rounds |
|---|---:|---:|
| 1 | 187.75 ms | 103.81 / 106.63 MB |
| 2 | 123.95 ms | 149.60 / 148.32 MB |
| 4 | 113.89 ms | 186.79 / 188.63 MB |

Concurrency 4 reduced latency by 39.3% versus 1 in this separate sweep; each concurrency retained identical text on 36/36 calls. The original INT8 concurrency-1 result compared with optimized concurrency 4 is 52.7% lower latency, combining two changes. The runtime-only gain for this recording is 20.9%. Concurrency 4 follows FluidAudio 0.15.5's upstream default; applications can tune 2 or 1 for memory constraints. These whole-process RSS observations are not iPhone memory budgets. Main-comparison RSS and load times are strongly affected by prior Core ML cache state and should not be used to claim a runtime memory reduction or cold-start improvement.

## Protocol and scope

Release builds with Swift 6.2.4; identical frozen `OrukeetEngine` source for both runtimes. Models were loaded before timing and audio was decoded before timing. Wall time surrounds `engine.transcribe(samples:)`; internal processing time is also retained. Each fixture/configuration had three warmups and 15 timed repetitions in each of two rounds. Configuration order and base fixture order were reversed in round two, with fixture order rotated per repetition. Table values are means of the two round medians. All recorded thermal states were nominal. The 14 process runs contain 684 calls, including 114 warmups.

Encoder, decoder, and joint requested CPU plus Neural Engine; preprocessing requested CPU. No execution-plan trace establishes exclusive ANE placement. Models were existing compiled caches: no model downloads, tensor copies, or recompilation were performed. There is no physical iPhone result, mobile app integration result, power measurement, fresh cold-start measurement, statistical confidence interval, or claim of exhaustive optimization here.

The separate INT8 runtime regression used 400 distinct existing FLEURS clips, 16 in each of 25 languages, at concurrency 1: **400/400 exact text matches, zero differences**. The corpus was previously used for precision research, so this is runtime regression evidence rather than a fresh accuracy holdout. Dataset `google/fleurs`, revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, CC BY 4.0; manifest SHA-256 `dfcd5351a1126608dfc84b2644987ea6529f81e4182c03db036306ac36de4f1e`. Full transcript outputs remain in private scratch; the public [summary](parity/summary.json) records their hashes and per-language counts, and [per-clip receipts](parity/per-clip.jsonl) record audio and exact UTF-8 transcript hashes. The final parity run built against independent clean detached source worktrees. An earlier run whose dependency checkout changed during compilation was discarded.

## Exact identities and evidence

- Original FluidAudio 0.15.5: `19600a485baa4998812e4654b70d2bab8f2c9949`.
- Optimized measured revision: `75377a8a824abcd84946d80f3c2279ed25b13dbc`. Its production sources and package match implementation commit `e94c930fb2651351080b25d5a388b6fb8c2ca483`; the successor adds CI only.
- [Frozen engine](source/OrukeetEngine.swift): SHA-256 `8e26acfed4f739728889a152bc4fc1f35e7a7d1b499f42560599bf0fda19cd5d`. Snapshot base is `7687ca04bd5c07980a9769d2c727ef1dca8eeb4c`, with the then-uncommitted `prepare()` addition; `prepare()` was unused. The final integration changes comments and default concurrency to 4. Every measurement explicitly sets concurrency, so the default change does not alter the measured path.
- [Provenance](provenance.json) contains runtime, binary, engine, audio, and model-source hashes. [Compiled cache inventory](model-cache-integrity.json) hashes every model component used in place.
- [Portable-to-cache identity receipt](model-source-link.json) verifies all four compiled component weight files and vocabulary against the portable bundle. Encoder source graph SHA-256 is `bf29fc2cde15c33deefc9302fd5d888a3b67b641ff371ed61c87a85b3690b00f`; its identical source/cache weight hash is `7c3aa75d323fed4b0b13126b221859aac2d1933da4aa070deb111603a9e23447`. Portable archive SHA-256 is `24df9ff76f00f86f9ae1fd601cbbcab1d1eac98c7e8107de67444a7858d88b8b`. The receipt distinguishes compiled MIL hashes from portable protobuf graph hashes and records existing compiler-origin metadata.
- [Summary](summary.json) includes round medians, pooled p95, processing times, transcript checks, and RSS. [Raw timing results](results/) preserve every warmup and measured call.

## Reproduce using existing assets

Use isolated clean source worktrees at the two exact revisions above. Fill in [config.example.json](config.example.json) with existing model caches, audio, runtime checkouts, and a private scratch directory. Source compilation creates no model copies. The runner validates input audio hashes, runtime revisions, source cleanliness before/after build, and benchmark binary identity before inference. Preserve the recorded `results/` and `provenance.json` elsewhere before a new run; existing result files are intentionally never overwritten.

```sh
python3 run.py --config /path/to/private-config.json --phase build
python3 run.py --config /path/to/private-config.json --phase main
python3 run.py --config /path/to/private-config.json --phase concurrency
python3 summarize.py
python3 parity.py --config /path/to/private-config.json --manifest /path/to/existing/holdout25/manifest.json
```

`parity.py` stores complete transcripts under the configured private scratch directory and publishes compact receipts. `--summarize-only` revalidates existing raw-result identities and regenerates receipts without inference. Audio and model tensors are intentionally not included. The natural long fixture is the existing `yc-first-30s.wav`, identified by SHA-256 and 479,212 decoded samples in the raw results; exact reproduction requires that file. GPU compute policy is optional in the harness and was not measured in this evidence set.
