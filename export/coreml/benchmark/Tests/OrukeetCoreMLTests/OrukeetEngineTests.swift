@preconcurrency import AVFoundation
import Foundation
import Testing
@testable import OrukeetCoreML

private let recording = [Float](repeating: 0, count: 4_800)

private actor SuspendedTranscriber: OrukeetBatchTranscribing {
    private var started = false
    private var startedWaiters: [CheckedContinuation<Void, Never>] = []
    private var pending: CheckedContinuation<Void, Never>?
    private(set) var calls = 0

    func transcribe(_ samples: [Float]) async throws -> OrukeetEngine.Output {
        calls += 1
        if calls == 1 {
            // Intentionally ignore cancellation: the engine must still discard a
            // result when the caller cancels during a Core ML prediction.
            await withCheckedContinuation { continuation in
                pending = continuation
                started = true
                startedWaiters.forEach { $0.resume() }
                startedWaiters.removeAll()
            }
        }
        return .init(text: "recording", processingMs: 1)
    }

    func waitUntilStarted() async {
        if started { return }
        await withCheckedContinuation { startedWaiters.append($0) }
    }

    func finish() {
        pending?.resume()
        pending = nil
    }
}

private enum DeliberateFailure: Error { case inference }

private actor FailOnceTranscriber: OrukeetBatchTranscribing {
    private var failed = false
    func transcribe(_ samples: [Float]) async throws -> OrukeetEngine.Output {
        if !failed {
            failed = true
            throw DeliberateFailure.inference
        }
        return .init(text: "recovered", processingMs: 1)
    }
}

struct OrukeetEngineTests {
    private func missingModelsEngine(batchConcurrency: Int = 1) -> OrukeetEngine {
        OrukeetEngine(
            modelDirectory: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString),
            batchConcurrency: batchConcurrency)
    }

    @Test func inputErrorsPrecedeModelLoading() async {
        let engine = missingModelsEngine()
        await #expect(throws: OrukeetModelError.emptyAudio) {
            try await engine.transcribe(samples: [])
        }
        await #expect(throws: OrukeetModelError.invalidAudioFormat(sampleRate: 48_000, channels: 1)) {
            try await engine.transcribe(samples: recording, sampleRate: 48_000)
        }
        await #expect(throws: OrukeetModelError.invalidAudioFormat(sampleRate: 16_000, channels: 2)) {
            try await engine.transcribe(samples: recording, channels: 2)
        }
        for invalid in [Float.nan, Float.infinity, -Float.infinity] {
            var samples = recording
            samples[31] = invalid
            await #expect(throws: OrukeetModelError.nonFiniteSample(index: 31)) {
                try await engine.transcribe(samples: samples)
            }
        }
        await #expect(throws: OrukeetModelError.audioTooShort(minimumSamples: 4_800)) {
            try await engine.transcribe(samples: [0])
        }
    }

    @Test func invalidConcurrencyPrecedesModelLoading() async {
        let engine = missingModelsEngine(batchConcurrency: 0)
        await #expect(throws: OrukeetModelError.invalidBatchConcurrency) { try await engine.ensureLoaded() }
    }

    @Test func cancelledRequestDoesNotLoadModels() async {
        let engine = missingModelsEngine()
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await engine.transcribe(samples: recording)
        }
        await #expect(throws: CancellationError.self) { try await task.value }
    }

    @Test func overlappingRecordingsAreRejectedAndEngineRemainsReusable() async throws {
        let backend = SuspendedTranscriber()
        let engine = OrukeetEngine(transcriber: backend)
        let first = Task { try await engine.transcribe(samples: recording) }
        await backend.waitUntilStarted()
        await #expect(throws: OrukeetModelError.busy) { try await engine.transcribe(samples: recording) }
        await backend.finish()
        #expect(try await first.value.text == "recording")
        #expect(try await engine.transcribe(samples: recording).text == "recording")
        #expect(await backend.calls == 2)
    }

    @Test func cancellationDiscardsLateResultsAndReleasesAdmission() async throws {
        let backend = SuspendedTranscriber()
        let engine = OrukeetEngine(transcriber: backend)
        let first = Task { try await engine.transcribe(samples: recording) }
        await backend.waitUntilStarted()
        first.cancel()
        await #expect(throws: OrukeetModelError.busy) { try await engine.transcribe(samples: recording) }
        await backend.finish()
        await #expect(throws: CancellationError.self) { try await first.value }
        #expect(try await engine.transcribe(samples: recording).text == "recording")
    }

    @Test func unloadDuringRecordingIsDeferredAndCannotCreateASecondManager() async throws {
        let backend = SuspendedTranscriber()
        let engine = OrukeetEngine(transcriber: backend)
        let first = Task { try await engine.transcribe(samples: recording) }
        await backend.waitUntilStarted()
        await engine.unload()
        // The first manager remains retained until its inference completes.
        try await engine.ensureLoaded()
        await #expect(throws: OrukeetModelError.busy) { try await engine.transcribe(samples: recording) }
        await backend.finish()
        #expect(try await first.value.text == "recording")
        // A subsequent call now reloads, proving that the deferred unload ran.
        await #expect(throws: OrukeetModelError.missingComponent("/unused-test-models/parakeet_vocab.json")) {
            try await engine.transcribe(samples: recording)
        }
        #expect(await backend.calls == 1)
    }

    @Test func prepareWarmsOnceAndPreservesFreshRecordingAdmission() async throws {
        let backend = SuspendedTranscriber()
        let engine = OrukeetEngine(transcriber: backend)
        let warmup = Task { try await engine.prepare() }
        await backend.waitUntilStarted()
        await #expect(throws: OrukeetModelError.busy) {
            try await engine.transcribe(samples: recording)
        }
        await backend.finish()
        try await warmup.value
        try await engine.prepare()
        #expect(await backend.calls == 1)
        #expect(try await engine.transcribe(samples: recording).text == "recording")
        #expect(await backend.calls == 2)
        await engine.unload()
        await #expect(throws: OrukeetModelError.missingComponent("/unused-test-models/parakeet_vocab.json")) {
            try await engine.prepare()
        }
    }

    @Test func unloadDuringPreparationDoesNotLeaveEngineMarkedReady() async throws {
        let backend = SuspendedTranscriber()
        let engine = OrukeetEngine(transcriber: backend)
        let warmup = Task { try await engine.prepare() }
        await backend.waitUntilStarted()
        await engine.unload()
        // Keep the warming manager alive until its active prediction completes.
        try await engine.ensureLoaded()
        await #expect(throws: OrukeetModelError.busy) { try await engine.prepare() }
        await backend.finish()
        try await warmup.value
        // A fresh preparation must reload, not return through a stale ready flag.
        await #expect(throws: OrukeetModelError.missingComponent("/unused-test-models/parakeet_vocab.json")) {
            try await engine.prepare()
        }
        #expect(await backend.calls == 1)
    }

    @Test func inferenceFailureReleasesAdmission() async throws {
        let engine = OrukeetEngine(transcriber: FailOnceTranscriber())
        await #expect(throws: DeliberateFailure.self) { try await engine.transcribe(samples: recording) }
        #expect(try await engine.transcribe(samples: recording).text == "recovered")
    }

    /// Opt in with existing compiled models and an existing 16 kHz mono WAV.
    /// This does not download weights and is macOS host evidence, not iPhone evidence.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ORUKEET_TEST_MODELS"] != nil
                  && ProcessInfo.processInfo.environment["ORUKEET_TEST_AUDIO"] != nil))
    func realBatchShortLongAndRepeat() async throws {
        let env = ProcessInfo.processInfo.environment
        let modelURL = URL(fileURLWithPath: try #require(env["ORUKEET_TEST_MODELS"]))
        let audioURL = URL(fileURLWithPath: try #require(env["ORUKEET_TEST_AUDIO"]))
        let file = try AVAudioFile(forReading: audioURL)
        #expect(file.processingFormat.sampleRate == 16_000)
        #expect(file.processingFormat.channelCount == 1)
        let buffer = try #require(AVAudioPCMBuffer(
            pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        let samples = Array(UnsafeBufferPointer(
            start: try #require(buffer.floatChannelData)[0], count: Int(buffer.frameLength)))
        #expect(samples.count >= 4_800 && samples.count <= 240_000)
        let engine = OrukeetEngine(modelDirectory: modelURL, batchConcurrency: 1)
        let first = try await engine.transcribe(samples: samples, sampleRate: 16_000, channels: 1)
        #expect(!first.text.isEmpty && first.processingMs.isFinite)
        let repeatResult = try await engine.transcribe(samples: samples)
        #expect(first.text == repeatResult.text)
        var longSamples: [Float] = []
        while longSamples.count < 496_000 { longSamples.append(contentsOf: samples) }
        let long = try await engine.transcribe(samples: longSamples)
        #expect(!long.text.isEmpty && long.processingMs.isFinite)
        let final = try await engine.transcribe(samples: samples)
        #expect(final.text == first.text)
        await engine.unload()
        print("OrukeetEngine host smoke: short=\(samples.count) samples, long=\(longSamples.count) samples, shortMs=\(first.processingMs), longMs=\(long.processingMs), text=\(first.text)")
    }
}
