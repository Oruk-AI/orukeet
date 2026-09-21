@preconcurrency import AVFoundation
import Foundation
import Testing
@testable import OrukeetCoreML

private struct TemporaryRecording {
    let directory: URL

    init() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    }

    func remove() { try? FileManager.default.removeItem(at: directory) }

    func write(rate: Double, channels: Int, frames: Int, extension suffix: String = "caf",
               sample: (Int, Int) -> Float) throws -> URL {
        let url = directory.appendingPathComponent("recording.\(suffix)")
        let format = try #require(AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate,
                                              channels: AVAudioChannelCount(channels), interleaved: false))
        var settings: [String: Any] = suffix == "m4a" ? [
            AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: rate,
            AVNumberOfChannelsKey: channels, AVEncoderBitRateKey: 128_000,
        ] : format.settings
        if suffix != "m4a" { settings[AVLinearPCMIsNonInterleaved] = false }
        let file = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32,
                                   interleaved: false)
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 2_047))
        let data = try #require(buffer.floatChannelData)
        var offset = 0
        while offset < frames {
            let count = min(2_047, frames - offset)
            for channel in 0..<channels {
                for index in 0..<count { data[channel][index] = sample(offset + index, channel) }
            }
            buffer.frameLength = AVAudioFrameCount(count)
            try file.write(from: buffer)
            offset += count
        }
        return url
    }
}

struct OrukeetAudioTests {
    @Test func mono16kPreservesSamplesAndClipsOutsidePCMRange() throws {
        let recording = try TemporaryRecording()
        defer { recording.remove() }
        let url = try recording.write(rate: 16_000, channels: 1, frames: 8_211) { frame, _ in
            frame == 5 ? 1.25 : (frame == 6 ? -1.25 : Float(frame % 31) / 64)
        }
        let result = try OrukeetAudio.readMono16k(from: url)
        #expect(result.count == 8_211)
        #expect(result[5] == 1 && result[6] == -1)
        #expect(result[8_210] == Float(8_210 % 31) / 64)
        #expect(result.allSatisfy(\.isFinite))
    }

    @Test(arguments: [44_100.0, 48_000.0])
    func stereoAveragesBothChannelsAndPreservesFrequency(rate: Double) throws {
        let recording = try TemporaryRecording()
        defer { recording.remove() }
        let url = try recording.write(rate: rate, channels: 2, frames: Int(rate), extension: "wav") { frame, channel in
            let frequency = channel == 0 ? 500.0 : 1_000.0
            let amplitude = channel == 0 ? 0.3 : 0.5
            return Float(amplitude * sin(2 * Double.pi * frequency * Double(frame) / rate))
        }
        let result = try OrukeetAudio.readMono16k(from: url)
        #expect(abs(result.count - 16_000) <= 1)
        // Different tones in each channel detect channel dropping, incorrect
        // gain, resampling twice, and replay of an input buffer.
        let error = (200..<15_800).reduce(0.0) { total, index in
            let t = Double(index) / 16_000
            let expected = 0.15 * sin(2 * Double.pi * 500 * t) + 0.25 * sin(2 * Double.pi * 1_000 * t)
            return total + pow(Double(result[index]) - expected, 2)
        } / 15_600
        #expect(sqrt(error) < 0.002)
    }

    @Test(arguments: [44_100.0, 48_000.0])
    func drainsFinalPartialBufferAndDoesNotTruncate(rate: Double) throws {
        let recording = try TemporaryRecording()
        defer { recording.remove() }
        let frames = 66_001
        let url = try recording.write(rate: rate, channels: 1, frames: frames) { frame, _ in
            frame >= frames - 1_000 ? 0.5 : 0.1
        }
        let result = try OrukeetAudio.readMono16k(from: url)
        #expect(abs(result.count - Int((Double(frames) * 16_000 / rate).rounded())) <= 1)
        #expect(abs(result[result.count - 50] - 0.5) < 0.002)
        #expect(abs(result[200] - 0.1) < 0.002)
    }

    @Test func decodesAACRecording() throws {
        let recording = try TemporaryRecording()
        defer { recording.remove() }
        let url = try recording.write(rate: 48_000, channels: 2, frames: 48_000, extension: "m4a") { frame, _ in
            Float(0.4 * sin(2 * Double.pi * 700 * Double(frame) / 48_000))
        }
        let result = try OrukeetAudio.readMono16k(from: url)
        #expect(abs(result.count - 16_000) < 400)
        #expect(result.allSatisfy { $0.isFinite && abs($0) <= 1 })
        let rms = sqrt(result.dropFirst(300).dropLast(300).reduce(0.0) { $0 + Double($1 * $1) }
                       / Double(result.count - 600))
        #expect(abs(rms - 0.4 / sqrt(2)) < 0.025)
    }

    @Test func rejectsEmptyMalformedAndNonFiniteAudio() throws {
        let recording = try TemporaryRecording()
        defer { recording.remove() }
        let empty = try recording.write(rate: 48_000, channels: 1, frames: 0) { _, _ in 0 }
        #expect(throws: OrukeetAudioError.emptyRecording) { try OrukeetAudio.readMono16k(from: empty) }
        let invalid = recording.directory.appendingPathComponent("invalid.wav")
        try Data("not audio".utf8).write(to: invalid)
        #expect(throws: (any Error).self) { try OrukeetAudio.readMono16k(from: invalid) }
        for value in [Float.nan, .infinity, -.infinity] {
            let file = try recording.write(rate: 48_000, channels: 1, frames: 9_001) { frame, _ in
                frame == 8_700 ? value : 0
            }
            #expect(throws: OrukeetAudioError.nonFiniteSample) { try OrukeetAudio.readMono16k(from: file) }
        }
    }

    @Test func observesCancellationBeforeOpeningFile() async {
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try OrukeetAudio.readMono16k(from: URL(fileURLWithPath: "/unused-cancelled-recording.wav"))
        }
        await #expect(throws: CancellationError.self) { try await task.value }
    }
}
