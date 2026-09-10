# Optimized Orukeet ONNX: cross-platform verification

[CI run](https://github.com/Oruk-AI/openwhispr/actions/runs/34519797164) · App source `f2d1fa16eda2ba54992941bb7a5a8aa137a826dc` · QA source `518aa16931d1c1cfe9c86b3240ce8d14c479372f`

This comparison uses the original and optimized Orukeet ONNX archives at the same model checkpoint. Both run through OpenWhispr’s production ParakeetManager with stock sherpa-onnx 1.13.4 and its CPU provider.

All four runner configurations passed 64 qualification checks. Transcripts matched exactly in all 16 duration/platform comparisons. The optimized archive was faster in 15 of 16 measured pairs.

Each result is the median of three warm calls. Server warmup and one transcription at the measured duration run first. Artifact order alternates between durations, with one model resident per block. The app selects its normal CPU thread count on each machine.

| Platform                                                                                         | Runner CPU                               | Logical CPUs | App threads |
| ------------------------------------------------------------------------------------------------ | ---------------------------------------- | -----------: | ----------: |
| [Linux x64](https://github.com/Oruk-AI/openwhispr/actions/runs/34519797164/job/103014173244)     | AMD EPYC 9V74 80-Core Processor          |            4 |           3 |
| [Windows x64](https://github.com/Oruk-AI/openwhispr/actions/runs/34519797164/job/103014172838)   | INTEL(R) XEON(R) PLATINUM 8573C          |            4 |           3 |
| [Apple Silicon](https://github.com/Oruk-AI/openwhispr/actions/runs/34519797164/job/103014174819) | Apple M1 (Virtual)                       |            3 |           2 |
| [Intel Mac](https://github.com/Oruk-AI/openwhispr/actions/runs/34519797164/job/103014173382)     | Intel(R) Core(TM) i7-8700B CPU @ 3.20GHz |            4 |           3 |

| Platform      | Audio | Original Orukeet | Optimized Orukeet | Speedup |
| ------------- | ----: | ---------------: | ----------------: | ------: |
| Linux x64     | 1.5 s |         291.0 ms |          258.6 ms |   1.13× |
| Linux x64     |   3 s |         439.9 ms |          403.4 ms |   1.09× |
| Linux x64     |   6 s |         776.3 ms |          710.4 ms |   1.09× |
| Linux x64     |  11 s |        1344.2 ms |         1211.5 ms |   1.11× |
| Windows x64   | 1.5 s |         592.9 ms |          471.9 ms |   1.26× |
| Windows x64   |   3 s |         953.1 ms |          592.8 ms |   1.61× |
| Windows x64   |   6 s |        1143.6 ms |         1124.9 ms |   1.02× |
| Windows x64   |  11 s |        1555.1 ms |         1555.6 ms |   1.00× |
| Apple Silicon | 1.5 s |         747.5 ms |          520.0 ms |   1.44× |
| Apple Silicon |   3 s |         979.2 ms |          733.0 ms |   1.34× |
| Apple Silicon |   6 s |        1606.1 ms |         1456.5 ms |   1.10× |
| Apple Silicon |  11 s |        2721.5 ms |         2687.4 ms |   1.01× |
| Intel Mac     | 1.5 s |         245.4 ms |          175.6 ms |   1.40× |
| Intel Mac     |   3 s |        1404.3 ms |         1098.1 ms |   1.28× |
| Intel Mac     |   6 s |        2569.1 ms |         1232.8 ms |   2.08× |
| Intel Mac     |  11 s |        2672.0 ms |         1979.2 ms |   1.35× |

Windows at 11 seconds measured 1555.13 ms for the original graph and 1555.57 ms for the optimized graph: a 0.44 ms difference (0.028%), treated as tied at this sample size. These are three-repeat warm CI checks of the listed runner configurations.

Each archive was downloaded anonymously and verified before extraction. Installed graph and token hashes, repeated-process reuse, concurrent transcription, cancellation, and shutdown/restart passed for both artifacts on every runner.

[Machine-readable receipts](cross-platform-ci.json) contain all warm measurements, hardware details, artifact hashes, binary hashes, and transcript equality hashes.
