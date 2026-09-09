# Historical application benchmark

This directory preserves the file-transcription benchmark used for the
report's 111 ms Metal and 52 ms CUDA medians. It ran in the Qt-based
[Knuckles92/OpenWhisper v2.6.0](https://github.com/Knuckles92/OpenWhisper/releases/tag/v2.6.0)
fork, not the Electron-based OpenWhispr/openwhispr application.

`benchmark_oruk_native.py` depends on that historical application's modified
backend. It records the exact model/runtime paths and all warm samples.
The [technical report](../../docs/technical-report.md) states the measured
boundaries and links the original receipts. These results are not a benchmark
of the separate OpenWhispr integration.

For a standalone result using the released package, run the
[replayable proof](../../demos/README.md). Its receipt measures a separate
181 ms warm file call on M5 Max/Metal. General file, batch and application
worker examples are in [ASR usage](../../docs/usage.md).
