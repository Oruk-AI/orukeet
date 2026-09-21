# OpenWhispr iPhone qualification protocol

Record the app commit, FluidAudio revision, archive SHA-256, iPhone model, RAM,
iOS version, available storage, compute-unit policy and chunk concurrency with
every run. Use a physical device for performance and memory; simulator results
only test integration. Start with the oldest supported iPhone and one recent
iPhone. Set acceptable recording-to-text latency and peak-memory limits with the
application owner before evaluating. No universal budget is inferred from Mac
measurements.

| Gate | Procedure | Pass condition |
|---|---|---|
| iOS build | Build `OrukeetCoreML` for generic iOS and the actual app target | Both compile with the app's selected SDK/dependency versions |
| Install | Hash-check, extract, compile on phone, load offline | All four components and exact vocabulary load; model is identified as Orukeet |
| Interrupted install | Cancel/fail during download and compilation, then retry | Existing selection remains usable; retry succeeds; no partial cache selected |
| Airplane mode | Relaunch installed app and transcribe after networking is disabled | Local transcription succeeds without a download or hosted fallback |
| Audio boundary | Test empty input, <300 ms, NaN/Inf, stereo and 44.1/48 kHz capture | Invalid engine input fails; app resampling/downmix produces verified 16 kHz mono |
| Silence | Test silence and quiet/noisy rooms using current app VAD | Empty/no-speech behavior is acceptable; hallucinations are counted |
| Duration | Natural recordings at 0.3 s, 2 s, 14.9 s, 15.1 s, 30 s, 60 s and several minutes | Complete beginning/end transcription; no silent truncation or chunk-boundary loss |
| Lifecycle | Rapid sequential recordings, concurrent attempt, cancellation, switching models, unload/reload and background/foreground | Fresh state, visible busy/error handling, no stale result pasted and no cross-recording text |
| Performance | At least 30 recordings per duration/device, balanced candidate order | Report cold install, first load, warm request p50/p95, audio length and real-time factor separately |
| Memory/thermal | Long recording and repeated requests under Instruments, with foreground/background transitions | No jetsam/crash, within agreed memory budget; record thermal state and sustained latency |
| English | Run v2, v3 and Orukeet on the identical consented English dictation set | Word errors, punctuation/proper names and latency support the intended selector change |
| Multilingual | Run v3 and Orukeet on all 25 languages using sealed identical recordings | Per-language WER/CER and wrong-language/script cases meet predeclared criteria |
| Known regressions | Include Greek final-sigma cases and Slovenian Latin/Cyrillic cases, plus code switching | Inspect raw text as well as normalized metrics; do not hide script changes through normalization |
| Attribution | Inspect app-distributed bundle and acknowledgments | Model license and NVIDIA/Fluid Inference notices remain accessible |

The published preview's existing Mac evidence is a starting regression reference.
It cannot fill any physical iPhone gate. Retain the English v2 option and the
previous multilingual model until the appropriate gates pass.

For accuracy, seal recording IDs, audio hashes and references before running
candidates. Report fixed-reference word/character denominators, per-language
results, empty outputs, failures and latency. Keep the app's decoding and text
cleanup identical across candidates; score raw ASR separately from cleanup.

Do not retune on the acceptance set. Any precision/profile change needs a new
held-out check and a new artifact identity. A successful short synthetic input
does not qualify long-form transcription or dictation quality.
