# iOS Simulator model-runtime check

The `Core ML iOS validation` workflow includes a separate runtime job in addition
to the unsigned iOS library build. This job uses Xcode 16.4 and an iPhone 16 /
iOS 18.5 simulator on a GitHub-hosted macOS runner. It tests Core ML model
compilation and inference inside the simulated iOS process, rather than loading
a previously compiled Mac cache.

The job downloads the existing public greedy inference ZIP at Hugging Face
revision `43142dd1897f9ddadcd70173fcb5ff45c08aa951` onto the ephemeral runner.
It authenticates the pinned archive and all payload files before extraction.
It does not download a NeMo/training checkpoint, create new trained weights, or
copy model files back to a developer's computer. Uploaded evidence contains only
the archive receipt, runtime JSON and test log.

`OrukeetPortableRuntimeTests` compiles all four portable model components inside
the iOS test process using the public installer. It then uses the public engine
to transcribe the repository's English JFK, French, Spanish and Latvian WAV
fixtures. It checks independent recording state, a repeated 33-second English
input, and unload/reload behavior. The existing audio provenance and licenses
remain under `demos/fixtures` and `demos/multilingual`.

Success requires both `xcodebuild test` success and a complete runtime report
with the expected fixture count and passing lifecycle assertions. A missing
environment variable that causes the model test to be skipped cannot produce a
passing workflow: the report gate fails. The test also requires the iOS simulator
compilation environment, so a Mac-only execution cannot satisfy that gate.

To reproduce using existing portable model files on an Xcode-equipped Mac, run
from `export/coreml/benchmark` with absolute paths:

```sh
export TEST_RUNNER_ORUKEET_PORTABLE_TEST_MODELS='/path/to/orukeet-r3-coreml-greedy'
export TEST_RUNNER_ORUKEET_PORTABLE_TEST_REPO='/path/to/orukeet-checkout'
export TEST_RUNNER_ORUKEET_PORTABLE_TEST_REPORT='/path/to/runtime.json'
xcodebuild test -scheme OrukeetCoreML -configuration Release \
  -destination 'platform=iOS Simulator,name=iPhone 16,OS=18.5' \
  -parallel-testing-enabled NO -only-testing:OrukeetCoreMLTests \
  IPHONEOS_DEPLOYMENT_TARGET=17.0 CODE_SIGNING_ALLOWED=NO
```

Xcode strips the [`TEST_RUNNER_` prefix](https://developer.apple.com/documentation/xcode/environment-variable-reference)
when passing these variables to the test process. Simulator processes can access
the host filesystem; their home directories are separate. This is described in
Apple's [Getting the Most out of Simulator](https://developer.apple.com/videos/play/wwdc2019/418/).
The model compiler's destination is a new directory in the simulator process's
temporary directory and is removed after the check.

This is a functional runtime check on reused speech fixtures. It does not
measure physical iPhone performance, memory headroom, Neural Engine execution,
battery use, minimum-iOS-17 runtime behavior, or full multilingual accuracy.
The simulated device name is not the host hardware. Its timing diagnostics must
not be quoted as iPhone latency. OpenWhispr's actual app and the
[physical-device protocol](device-qualification.md) remain separate gates.
