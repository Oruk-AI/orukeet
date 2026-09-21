@preconcurrency import AVFoundation
import Foundation

public enum OrukeetAudioError: Error, LocalizedError, Equatable {
    case unsupportedFormat
    case emptyRecording
    case nonFiniteSample
    case conversionFailed

    public var errorDescription: String? {
        switch self {
        case .unsupportedFormat: "The recording has an unsupported audio format"
        case .emptyRecording: "The recording contains no audio samples"
        case .nonFiniteSample: "The recording contains a non-finite audio sample"
        case .conversionFailed: "The recording could not be converted to 16 kHz mono audio"
        }
    }
}

/// Decodes local recordings for batch transcription without changing their gain.
public enum OrukeetAudio {
    /// Reads a mono or stereo AVFoundation recording (including WAV, CAF, and M4A)
    /// into normalized Float32 PCM at 16 kHz mono. Stereo channels are mixed;
    /// finite values outside PCM's -1...1 range are clipped, not gain-normalized.
    ///
    /// Decoding and conversion use bounded buffers; only the returned mono audio
    /// grows with recording duration. Call off the main thread. Cancellation is
    /// checked before opening the file and throughout conversion.
    public static func readMono16k(from url: URL) throws -> [Float] {
        try Task.checkCancellation()
        let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        guard file.length > 0 else { throw OrukeetAudioError.emptyRecording }
        let format = file.processingFormat
        guard format.sampleRate.isFinite, format.sampleRate > 0,
              (1...2).contains(format.channelCount),
              let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                  sampleRate: 16_000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: format, to: outputFormat),
              let inputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 8_192),
              let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: 4_096) else {
            throw OrukeetAudioError.unsupportedFormat
        }
        // AVAudioConverter defaults to channel remapping. Explicit downmix is
        // necessary to retain both channels of stereo recordings.
        converter.downmix = true
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        let input = FileInput(file: file, buffer: inputBuffer)
        var samples: [Float] = []
        var stalledCalls = 0
        while true {
            try Task.checkCancellation()
            outputBuffer.frameLength = 0
            var conversionError: NSError?
            let framesBefore = input.framesRead
            let status = converter.convert(to: outputBuffer, error: &conversionError) { requested, status in
                input.next(requested: requested, status: status)
            }
            if let error = input.error { throw error }
            if status == .error { throw conversionError ?? OrukeetAudioError.conversionFailed }
            guard let channel = outputBuffer.floatChannelData?[0] else {
                throw OrukeetAudioError.conversionFailed
            }
            for index in 0..<Int(outputBuffer.frameLength) {
                let sample = channel[index]
                guard sample.isFinite else { throw OrukeetAudioError.nonFiniteSample }
                samples.append(min(1, max(-1, sample)))
            }
            if status == .endOfStream { break }
            // A complete local file never needs an indefinite "no data now"
            // wait. Detect a broken conversion instead of spinning forever.
            stalledCalls = outputBuffer.frameLength == 0 && input.framesRead == framesBefore ? stalledCalls + 1 : 0
            guard stalledCalls < 8 else { throw OrukeetAudioError.conversionFailed }
        }
        try Task.checkCancellation()
        guard !samples.isEmpty else { throw OrukeetAudioError.emptyRecording }
        return samples
    }
}

// The converter invokes its input block synchronously during convert(...).
// This box bridges the SDK's @Sendable callback without capturing mutable local
// variables; it is never shared with another conversion or accessed concurrently.
private final class FileInput: @unchecked Sendable {
    let file: AVAudioFile
    let buffer: AVAudioPCMBuffer
    var framesRead: Int64 = 0
    var error: (any Error)?
    private var ended = false

    init(file: AVAudioFile, buffer: AVAudioPCMBuffer) {
        self.file = file
        self.buffer = buffer
    }

    func next(requested: AVAudioPacketCount, status: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioBuffer? {
        guard !ended, error == nil else {
            status.pointee = .endOfStream
            return nil
        }
        do {
            try Task.checkCancellation()
            guard requested > 0 else { throw OrukeetAudioError.conversionFailed }
            let remaining = file.length - file.framePosition
            guard remaining > 0 else {
                ended = true
                status.pointee = .endOfStream
                return nil
            }
            buffer.frameLength = 0
            let count = AVAudioFrameCount(min(Int64(min(requested, buffer.frameCapacity)), remaining))
            try file.read(into: buffer, frameCount: count)
            guard buffer.frameLength > 0 else { throw OrukeetAudioError.conversionFailed }
            guard let channels = buffer.floatChannelData else { throw OrukeetAudioError.unsupportedFormat }
            for channel in 0..<Int(buffer.format.channelCount) {
                for index in 0..<Int(buffer.frameLength) where !channels[channel][index].isFinite {
                    throw OrukeetAudioError.nonFiniteSample
                }
            }
            framesRead += Int64(buffer.frameLength)
            status.pointee = .haveData
            return buffer
        } catch {
            self.error = error
            status.pointee = .endOfStream
            return nil
        }
    }
}
