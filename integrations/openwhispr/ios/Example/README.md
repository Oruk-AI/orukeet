# Orukeet iOS reference app

Open `OrukeetExample.xcodeproj` in Xcode 16.4 or newer, select the
`OrukeetExample` scheme and run on an iOS 17+ device. Select your signing team
for a physical device. No project generator or additional package setup is
needed: the project links the repository-root `OrukeetCoreML` product and its
single pinned FluidAudio dependency.

Tap **Download and prepare Orukeet**, then **Record** and **Stop and transcribe**.
The SDK downloads, authenticates, extracts and compiles the INT8 model once;
the app keeps its prepared engine alive across recordings. English and all 25
supported languages use the same Orukeet model. **Import audio** runs existing
audio files through the same SDK conversion and batch-transcription path.
After installation, transcription works offline.

The recorder requests microphone access and creates a temporary mono 16 kHz CAF.
The completed file is deleted after transcription or cancellation. Imported
files remain untouched, with security-scoped access retained until reading
finishes. Microphone denial offers Settings or file import. An interruption or
input-device change cancels the recording; backgrounding cancels work and
unloads models after the operation finishes. Returning to the app exposes
**Prepare Orukeet** again. Core ML may finish an in-flight prediction before
cancellation is observed; the app waits and discards that result.

The sources are intentionally small:

- `ExampleModel.swift` sequences the public SDK calls, progress, cancellation and errors.
- `PCMRecorder.swift` owns microphone permission, the audio session and recording files.
- `OrukeetExampleApp.swift` provides the SwiftUI controls and system-event handling.

The recorder uses Apple's [recording permission API](https://developer.apple.com/documentation/avfaudio/avaudioapplication/requestrecordpermission(completionhandler:))
and [AVAudioRecorder](https://developer.apple.com/documentation/avfaudio/avaudiorecorder).
App integration and physical-device follow-up are described in the
[integration guide](../README.md) and [device checklist](../device-qualification.md).

## Simulator app test

The checked-in UI test launches the actual app, taps installation, waits for
preparation, and transcribes `demos/fixtures/jfk.wav` twice. It then relaunches
without an archive argument, prepares the installed cache, and checks the same
transcript again. It requires the real
INT8 archive and asserts a nonempty JFK transcript and identical repeat output.
It does not access a microphone or replace the model/installer with a mock.
The ZIP remains on the ephemeral CI runner; the test creates an isolated install
directory beside the host archive and removes it afterward. A required JSON
receipt records installation, repeated transcription and offline relaunch.

With the already-authenticated ZIP available on an Xcode-equipped machine, run
from the repository root:

```sh
export TEST_RUNNER_ORUKEET_EXAMPLE_TEST_ARCHIVE='/path/to/int8.zip'
export TEST_RUNNER_ORUKEET_EXAMPLE_TEST_FIXTURE="$PWD/demos/fixtures/jfk.wav"
export TEST_RUNNER_ORUKEET_EXAMPLE_TEST_REPORT='/path/to/evidence/example-app.json'
xcodebuild test \
  -project integrations/openwhispr/ios/Example/OrukeetExample.xcodeproj \
  -scheme OrukeetExample -configuration Release \
  -destination 'platform=iOS Simulator,name=iPhone 16,OS=18.5' \
  -parallel-testing-enabled NO -only-testing:OrukeetExampleUITests \
  CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO
```

The test forwards these paths through `--orukeet-archive`,
`--orukeet-fixture` and `--orukeet-test-root` launch arguments. Without those
arguments, the app uses its ordinary download flow and Application Support
cache. Simulator results establish app wiring and runtime behavior; physical
iPhone latency, memory and thermal measurements remain deployment follow-up.
