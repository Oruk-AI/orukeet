# iOS Simulator model-runtime check

The workflow checks the root `OrukeetCoreML` package with the same Orukeet INT8
bundle used for English and multilingual recordings. Consult the
[validation record](../../../evidence/coreml-openwhispr-20260920/README.md) for the
source revision, runtime pin and model identity covered by each completed run.

The earlier [run 35571858702](https://github.com/Oruk-AI/orukeet/actions/runs/35571858702)
passed at source commit `65199e54f1509acc7ad9d5d3897813da7f2fabee`. Its
[runtime receipt](../../../evidence/coreml-openwhispr-20260920/simulator-runtime.json)
covers the prior LUT6 bundle and runtime. It does not establish that the new
INT8 bundle, buffer backport or preparation API passed.

The `Core ML iOS validation` workflow includes a separate runtime job in addition
to the unsigned iOS library build. This job uses Xcode 16.4 and an iPhone 16 /
iOS 18.5 simulator on a GitHub-hosted macOS runner. It tests Core ML model
compilation and inference inside the simulated iOS process, rather than loading
a previously compiled Mac cache.

The CI setup explicitly inventories the installed runtimes, creates an iPhone 16
under the exact iOS 18.5 runtime, boots it, and selects its UDID. It does not rely
on a preexisting named device. Setup fails with an inventory receipt if that
runtime is absent; it never silently substitutes another iOS version.

The job downloads the portable INT8 inference ZIP at the immutable Hugging Face
revision recorded in the workflow and `OrukeetBundle.int8` onto the ephemeral runner.
It authenticates the pinned archive and all payload files before extraction.
It does not download a NeMo/training checkpoint, create new trained weights, or
copy model files back to a developer's computer. Uploaded evidence contains only
the archive receipt, runtime JSON and test log.

`OrukeetPortableRuntimeTests` verifies the ZIP again with the public Swift archive
verifier and compiles all four portable model components inside the iOS test
process using the public installer. It calls `prepare()` before using the public engine
to transcribe the repository's English JFK, French, Spanish and Latvian WAV
fixtures. It checks independent recording state, a repeated 33-second English
input, and unload/reload with preparation. The existing audio provenance and licenses
remain under `demos/fixtures` and `demos/multilingual`.

Success requires both `xcodebuild test` success and a complete runtime report
with the expected fixture count and passing lifecycle assertions. A missing
environment variable that causes the model test to be skipped cannot produce a
passing workflow: the report gate fails. The test also requires the iOS simulator
compilation environment, so a Mac-only execution cannot satisfy that gate.

To reproduce using existing portable model files on an Xcode-equipped Mac, run
from the repository root with absolute paths. Keep the authenticated archive as
`int8.zip` beside its extracted `orukeet-r3-coreml-int8sym-encoder-only-experimental-20260920`
directory: the test requires both. This manual command assumes
an iPhone 16 / iOS 18.5 simulator already exists in Xcode's Devices and Simulators
window; create it there first if needed. The CI-only setup helper creates its own
device and uses its UDID instead of this manual name-based destination.

```sh
export TEST_RUNNER_ORUKEET_PORTABLE_TEST_MODELS='/path/to/orukeet-r3-coreml-int8sym-encoder-only-experimental-20260920'
export TEST_RUNNER_ORUKEET_PORTABLE_TEST_REPO='/path/to/orukeet-checkout'
export TEST_RUNNER_ORUKEET_PORTABLE_TEST_REPORT='/path/to/runtime.json'
xcodebuild -list
xcodebuild test -scheme Orukeet -configuration Release \
  -destination 'platform=iOS Simulator,name=iPhone 16,OS=18.5' \
  -parallel-testing-enabled NO -only-testing:OrukeetCoreMLTests \
  ENABLE_TESTABILITY=YES \
  IPHONEOS_DEPLOYMENT_TARGET=17.0 CODE_SIGNING_ALLOWED=NO
```

The root package contains the library and its tests. The macOS command-line
benchmark lives in a separate nested package and is outside this iOS build graph.
`ENABLE_TESTABILITY=YES` permits the unit suite's `@testable` imports in this
Release test build; the separate iOS library build retains its ordinary settings.

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
[physical-device protocol](device-qualification.md) are deployment follow-ups.
