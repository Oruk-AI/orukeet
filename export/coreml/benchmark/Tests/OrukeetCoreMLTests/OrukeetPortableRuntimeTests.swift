@preconcurrency import AVFoundation
import CryptoKit
import Foundation
import OrukeetCoreML
import Testing
#if os(iOS)
import UIKit
#endif

/// Opt-in integration evidence for the iOS Simulator runtime. CI verifies the
/// portable archive before supplying it; this test compiles it in that runtime.
/// Absolute host fixture paths are simulator-only and are not an app install API.
struct OrukeetPortableRuntimeTests {
    private struct Fixture {
        let path: String
        let samples: [Float]
        let sha256: String

        var seconds: Double { Double(samples.count) / 16_000 }
    }

    private func fixture(at relativePath: String, in repository: URL) throws -> Fixture {
        let url = repository.appendingPathComponent(relativePath)
        let file = try AVAudioFile(forReading: url)
        try #require(file.processingFormat.sampleRate == 16_000,
                     "The fixture must already contain 16 kHz PCM")
        try #require(file.processingFormat.channelCount == 1,
                     "The fixture must already be mono")
        try #require(file.length >= 4_800 && file.length <= AVAudioFramePosition(UInt32.max))
        let buffer = try #require(AVAudioPCMBuffer(
            pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        try #require(AVAudioFramePosition(buffer.frameLength) == file.length)
        let channel = try #require(buffer.floatChannelData)[0]
        let samples = Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
        // Evaluate outside the macro: Swift 6.1 treats a key-path predicate
        // forwarded through #require's generated closure as throwing.
        let finiteSamples = samples.allSatisfy(\.isFinite)
        try #require(finiteSamples)
        return Fixture(
            path: relativePath, samples: samples,
            sha256: SHA256.hash(data: try Data(contentsOf: url))
                .map { String(format: "%02x", $0) }.joined())
    }

    private func words(in text: String) -> [Substring] {
        text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber })
    }

    // Setting any portable-test variable opts in. A partially configured run
    // must fail rather than silently skip a required integration check.
    @Test(.enabled(if: ["ORUKEET_PORTABLE_TEST_MODELS", "ORUKEET_PORTABLE_TEST_REPO",
                        "ORUKEET_PORTABLE_TEST_REPORT"].contains {
        ProcessInfo.processInfo.environment[$0] != nil
    }))
    func portableBundleCompilesAndTranscribes() async throws {
        #if os(iOS) && targetEnvironment(simulator)
        let environment = ProcessInfo.processInfo.environment
        let source = URL(fileURLWithPath: try #require(environment["ORUKEET_PORTABLE_TEST_MODELS"]))
        let repository = URL(fileURLWithPath: try #require(environment["ORUKEET_PORTABLE_TEST_REPO"]))
        let reportURL = URL(fileURLWithPath: try #require(environment["ORUKEET_PORTABLE_TEST_REPORT"]))
        let files = FileManager.default
        let temporary = files.temporaryDirectory.appendingPathComponent("orukeet-portable-\(UUID().uuidString)")
        try files.createDirectory(at: temporary, withIntermediateDirectories: false)
        defer { try? files.removeItem(at: temporary) }
        let installed = temporary.appendingPathComponent("installed", isDirectory: true)

        let beforeCompile = Date()
        try OrukeetLocalModels.compilePackages(from: source, to: installed)
        let compileMs = Date().timeIntervalSince(beforeCompile) * 1_000
        let requiredComponents = ["Preprocessor", "Encoder", "Decoder", "JointDecisionv3"]
        for component in requiredComponents {
            var isDirectory: ObjCBool = false
            let componentExists = files.fileExists(
                atPath: installed.appendingPathComponent("\(component).mlmodelc").path,
                isDirectory: &isDirectory)
            try #require(componentExists && isDirectory.boolValue)
        }
        let sidecars = ["parakeet_vocab.json", "bundle.json", "LICENSE-WEIGHTS", "NOTICE.md", "COREML-NOTICE.txt"]
        for sidecar in sidecars {
            let expected = try Data(contentsOf: source.appendingPathComponent(sidecar))
            try #require(!expected.isEmpty)
            let actual = try Data(contentsOf: installed.appendingPathComponent(sidecar))
            try #require(actual == expected,
                         "Device-local installation must preserve vocabulary, identity and attribution")
        }

        let fixturePaths = [
            "demos/fixtures/jfk.wav",
            "demos/multilingual/audio/fr_fr.playback.wav",
            "demos/multilingual/audio/es_419.playback.wav",
            "demos/multilingual/audio/lv_lv.playback.wav",
        ]
        let fixtures = try fixturePaths.map { try fixture(at: $0, in: repository) }
        let english = try #require(fixtures.first)
        try #require(english.samples.count <= 240_000)
        let engine = OrukeetEngine(modelDirectory: installed, batchConcurrency: 1)
        let beforeLoad = Date()
        try await engine.ensureLoaded()
        let loadMs = Date().timeIntervalSince(beforeLoad) * 1_000

        var fixtureResults: [[String: Any]] = []
        var firstEnglish = ""
        for (index, fixture) in fixtures.enumerated() {
            let output = try await engine.transcribe(
                samples: fixture.samples, sampleRate: 16_000, channels: 1)
            try #require(!output.text.isEmpty)
            try #require(output.processingMs.isFinite && output.processingMs >= 0)
            if index == 0 { firstEnglish = output.text }
            fixtureResults.append([
                "audio": fixture.path,
                "audio_sha256": fixture.sha256,
                "samples": fixture.samples.count,
                "audio_seconds": fixture.seconds,
                "sample_rate_hz": 16_000,
                "channels": 1,
                "text": output.text,
                "processing_ms_diagnostic": output.processingMs,
            ])
        }

        let afterMultilingual = try await engine.transcribe(samples: english.samples)
        try #require(afterMultilingual.text == firstEnglish,
                     "Separate multilingual recordings must not leak decoder state")
        let longSamples = english.samples + english.samples + english.samples
        try #require(longSamples.count > 480_000, "The lifecycle fixture must exceed 30 seconds")
        let long = try await engine.transcribe(samples: longSamples)
        try #require(!long.text.isEmpty && long.processingMs.isFinite && long.processingMs >= 0)
        let firstWindow = try await engine.transcribe(samples: Array(longSamples.prefix(240_000)))
        try #require(!firstWindow.text.isEmpty)
        let longWordCount = words(in: long.text).count
        let firstWindowWordCount = words(in: firstWindow.text).count
        let englishWordCount = words(in: firstEnglish).count
        try #require(englishWordCount > 5 && firstWindowWordCount > 5)
        let longExceedsFirstWindow = longWordCount > firstWindowWordCount
            && longWordCount >= englishWordCount * 2
        try #require(longExceedsFirstWindow,
                     "The long recording must emit substantially more than one inference window")
        let afterLong = try await engine.transcribe(samples: english.samples)
        try #require(afterLong.text == firstEnglish,
                     "Long-form chunking must not leak decoder state to the next recording")

        await engine.unload()
        let beforeReload = Date()
        try await engine.ensureLoaded()
        let reloadMs = Date().timeIntervalSince(beforeReload) * 1_000
        let reloaded = try await engine.transcribe(samples: english.samples)
        try #require(reloaded.text == firstEnglish)
        await engine.unload()

        let device = await MainActor.run {
            let device = UIDevice.current
            return (device.model, device.systemName, device.systemVersion)
        }
        let report: [String: Any] = [
            "scope": "iOS Simulator runtime smoke",
            "all_assertions_passed": true,
            "recorded_at": ISO8601DateFormatter().string(from: Date()),
            "device_model": device.0,
            "system_name": device.1,
            "system_version": device.2,
            "runtime_version": ProcessInfo.processInfo.operatingSystemVersionString,
            "simulator_model_identifier": environment["SIMULATOR_MODEL_IDENTIFIER"] ?? "unavailable",
            "fluid_audio_version": "0.15.5",
            "requested_encoder_compute_units": "cpuAndNeuralEngine",
            "compute_placement": "Not measured; requested compute units do not prove Neural Engine execution",
            "batch_concurrency": 1,
            "compiled_components": requiredComponents,
            "preserved_sidecars": sidecars,
            "compile_ms_diagnostic": compileMs,
            "load_ms_diagnostic": loadMs,
            "reload_ms_diagnostic": reloadMs,
            "fixture_results": fixtureResults,
            "repeat_after_multilingual_identical": afterMultilingual.text == firstEnglish,
            "repeat_after_long_identical": afterLong.text == firstEnglish,
            "post_reload_identical": reloaded.text == firstEnglish,
            "long_exceeds_first_window": longExceedsFirstWindow,
            "long_fixture": [
                "construction": "Three consecutive copies of demos/fixtures/jfk.wav",
                "samples": longSamples.count,
                "audio_seconds": Double(longSamples.count) / 16_000,
                "text": long.text,
                "word_count": longWordCount,
                "first_window_text": firstWindow.text,
                "first_window_word_count": firstWindowWordCount,
                "processing_ms_diagnostic": long.processingMs,
            ],
            "limitations": [
                "Simulator compatibility and lifecycle evidence only; no physical iPhone qualification",
                "Diagnostic timings do not measure iPhone latency, memory, battery or thermal behavior",
                "Nonempty multilingual output and repeated audio are smoke checks, not accuracy evaluation",
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: reportURL, options: .atomic)
        print("Orukeet portable Core ML simulator smoke passed; report: \(reportURL.path)")
        #else
        Issue.record("Portable runtime smoke must execute in the iOS Simulator, not on macOS or a physical device")
        #endif
    }
}
