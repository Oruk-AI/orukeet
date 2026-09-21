# iOS Simulator model-runtime check

**Passed at source commit `65199e54f1509acc7ad9d5d3897813da7f2fabee` in
[run 35571858702](https://github.com/Oruk-AI/orukeet/actions/runs/35571858702).**
The [runtime receipt](../../../evidence/coreml-openwhispr-20260920/simulator-runtime.json)
contains all four speech results and the passing long-form and lifecycle checks.

The `Core ML iOS validation` workflow includes a separate runtime job in addition
to the unsigned iOS library build. This job uses Xcode 16.4 and an iPhone 16 /
iOS 18.5 simulator on a GitHub-hosted macOS runner. It tests Core ML model
compilation and inference inside the simulated iOS process, rather than loading
a previously compiled Mac cache.

The CI setup explicitly inventories the installed runtimes, creates an iPhone 16
under the exact iOS 18.5 runtime, boots it, and selects its UDID. It does not rely
on a preexisting named device. Setup fails with an inventory receipt if that
runtime is absent; it never silently substitutes another iOS version.

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
from `export/coreml/benchmark` with absolute paths. This manual command assumes
an iPhone 16 / iOS 18.5 simulator already exists in Xcode's Devices and Simulators
window; create it there first if needed. The CI-only setup helper creates its own
device and uses its UDID instead of this manual name-based destination.

```sh
export TEST_RUNNER_ORUKEET_PORTABLE_TEST_MODELS='/path/to/orukeet-r3-coreml-greedy'
export TEST_RUNNER_ORUKEET_PORTABLE_TEST_REPO='/path/to/orukeet-checkout'
export TEST_RUNNER_ORUKEET_PORTABLE_TEST_REPORT='/path/to/runtime.json'
xcodebuild test -scheme OrukeetCoreMLBenchmark-Package -configuration Release \
  -destination 'platform=iOS Simulator,name=iPhone 16,OS=18.5' \
  -parallel-testing-enabled NO -only-testing:OrukeetCoreMLTests \
  ENABLE_TESTABILITY=YES \
  OTHER_SWIFT_FLAGS='$(inherited) -parse-as-library' \
  IPHONEOS_DEPLOYMENT_TARGET=17.0 CODE_SIGNING_ALLOWED=NO
```

The package-wide scheme includes its command-line benchmark, whose `@main`
entry point needs `-parse-as-library` under Xcode's generated build settings.
SwiftPM's ordinary `swift build` already supplies that flag. The library-only
scheme remains appropriate for the separate generic iOS build.
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
[physical-device protocol](device-qualification.md) remain separate gates.
