# Physical iPhone deployment follow-up

The root Swift package integrates one Orukeet r3 INT8 model for English and all
25 supported languages. These checks guide rollout in OpenWhispr's actual app;
physical-device measurements are follow-up work, not a prerequisite for merging
the reusable package. Start from the installation and preparation flow in the
[integration guide](README.md).

Record the app commit, FluidAudio revision, archive SHA-256, iPhone model, RAM,
iOS version, available storage, compute-unit policy and chunk concurrency with
every run. Use a physical device for performance and memory; simulator results
only test integration. Start with the oldest supported iPhone and one recent
iPhone. Set acceptable recording-to-text latency and peak-memory limits with the
application owner before evaluating. No universal budget is inferred from Mac
measurements.

| Check | Procedure | Expected result |
|---|---|---|
| iOS build | Add the repository-root `OrukeetCoreML` product and build the actual app target | App compiles with one pinned FluidAudio fork dependency and the selected SDK |
| Minimum OS | Install and run on iOS 17 as well as the app's current supported iOS | Installation and transcription work on the declared minimum OS |
| Install | Hash-check, extract, compile on phone, load offline | All four components and exact vocabulary load; model is identified as Orukeet |
| Interrupted install | Cancel/fail during download and compilation, then retry | Existing selection remains usable; retry succeeds; no partial cache selected |
| Airplane mode | Relaunch installed app and transcribe after networking is disabled | Local transcription succeeds without a download or hosted fallback |
| Audio boundary | Test empty input, <300 ms, NaN/Inf, stereo and 44.1/48 kHz capture | Invalid engine input fails; app resampling/downmix produces verified 16 kHz mono |
| Silence | Test silence and quiet/noisy rooms using current app VAD | Empty/no-speech behavior is acceptable; hallucinations are counted |
| Duration | Natural recordings at 0.3 s, 2 s, 14.9 s, 15.1 s, 30 s, 60 s and several minutes | Complete beginning/end transcription; no silent truncation or chunk-boundary loss |
| Lifecycle | Await `prepare()`, then test sequential recordings, concurrent attempts, cancellation, unload during preparation/inference, reload and background/foreground | Fresh state, visible busy/error handling, deferred unload, no stale result pasted and no cross-recording text |
| Performance | At least 30 recordings per duration/device; balance order when comparing runtime settings | Report installation, model load, preparation and warm request p50/p95 separately, with audio length and real-time factor |
| Memory/thermal | Long recording and repeated requests under Instruments, with foreground/background transitions | No jetsam/crash, within agreed memory budget; record thermal state and sustained latency |
| English | Run the same Orukeet INT8 engine on a sealed, consented English dictation set | Word errors, punctuation/proper names and latency meet the app's criteria |
| Multilingual | Run that engine on all 25 languages using sealed recordings | Per-language WER/CER and wrong-language/script cases meet predeclared criteria |
| Known regressions | Include Greek final-sigma cases and Slovenian Latin/Cyrillic cases, plus code switching | Inspect raw text as well as normalized metrics; do not hide script changes through normalization |
| Attribution | Inspect app-distributed bundle and acknowledgments | Model license and NVIDIA/Fluid Inference notices remain accessible |

The recorded Mac measurements and simulator checks are starting references.
Measure iPhone latency, memory and thermals on physical hardware. Use the same
Orukeet model and v3 decoder/vocabulary for English and multilingual requests;
there is no language-dependent model switch in this integration.

For accuracy, seal recording IDs, audio hashes and references before running
candidates or runtime configurations. Report fixed-reference word/character denominators, per-language
results, empty outputs, failures and latency. Keep the app's decoding and text
cleanup identical across candidates; score raw ASR separately from cleanup.

Do not retune on the acceptance set. Any precision/profile change needs a new
held-out check and a new artifact identity. A successful short synthetic input
does not qualify long-form transcription or dictation quality.
