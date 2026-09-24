@preconcurrency import AVFoundation
import Darwin
import Foundation
import OrukeetCoreML

@main
struct EngineSmoke {
    static func main() async throws {
        guard CommandLine.arguments.count == 5 else {
            fatalError("Usage: EngineSmoke COMPILED_MODELS SHORT_16K_MONO_WAV LONG_16K_MONO_WAV OUTPUT_JSON")
        }
        let modelPath = CommandLine.arguments[1]
        let audioPath = CommandLine.arguments[2]
        let longAudioPath = CommandLine.arguments[3]
        let outputPath = CommandLine.arguments[4]
        let file = try AVAudioFile(forReading: URL(fileURLWithPath: audioPath))
        guard file.processingFormat.sampleRate == 16_000, file.processingFormat.channelCount == 1,
              file.length >= 4800, file.length <= 240000,
              let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))
        else { fatalError("Invalid smoke fixture") }
        try file.read(into: buffer)
        let samples = Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
        let engine = OrukeetEngine(modelDirectory: URL(fileURLWithPath: modelPath), batchConcurrency: 1)
        let beforeLoad = Date()
        try await engine.ensureLoaded()
        let loadMs = Date().timeIntervalSince(beforeLoad) * 1000
        let first = try await engine.transcribe(samples: samples, sampleRate: 16000, channels: 1)
        let second = try await engine.transcribe(samples: samples)
        guard !first.text.isEmpty, first.text == second.text, first.processingMs.isFinite else {
            fatalError("Invalid or inconsistent short transcription")
        }
        let longFile = try AVAudioFile(forReading: URL(fileURLWithPath: longAudioPath))
        guard longFile.processingFormat.sampleRate == 16000, longFile.processingFormat.channelCount == 1,
              let longBuffer = AVAudioPCMBuffer(pcmFormat: longFile.processingFormat, frameCapacity: AVAudioFrameCount(longFile.length))
        else { fatalError("Invalid long fixture") }
        try longFile.read(into: longBuffer)
        let longSamples = Array(UnsafeBufferPointer(start: longBuffer.floatChannelData![0], count: Int(longBuffer.frameLength)))
        guard longSamples.count > 240000 else { fatalError("Long fixture must exercise chunking") }
        let long = try await engine.transcribe(samples: longSamples)
        let final = try await engine.transcribe(samples: samples)
        guard !long.text.isEmpty, long.processingMs.isFinite, final.text == first.text else {
            fatalError("Long recording failed or leaked decoder state")
        }
        let prefix = try await engine.transcribe(samples: Array(longSamples.prefix(240000)))
        let fullWords = long.text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber })
        let prefixWords = prefix.text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber })
        let newSuffix = fullWords.suffix(8).joined(separator: " ")
        let containsNewSuffix = !prefixWords.joined(separator: " ").contains(newSuffix)
        guard fullWords.count > prefixWords.count, containsNewSuffix,
              Array(fullWords.prefix(8)) == Array(prefixWords.prefix(8)) else {
            fatalError("Long transcription appears truncated to its first window")
        }
        await engine.unload()
        let beforeReload = Date()
        try await engine.ensureLoaded()
        let reloadMs = Date().timeIntervalSince(beforeReload) * 1000
        let reloaded = try await engine.transcribe(samples: samples)
        guard reloaded.text == first.text else { fatalError("Reload changed the short transcription") }
        await engine.unload()
        var usage = rusage()
        let usageOK = getrusage(RUSAGE_SELF, &usage) == 0
        let report: [String: Any] = [
            "host": ProcessInfo.processInfo.operatingSystemVersionString,
            "scope": "macOS host smoke, not iOS device validation",
            "short_audio": URL(fileURLWithPath: audioPath).lastPathComponent,
            "long_audio": URL(fileURLWithPath: longAudioPath).lastPathComponent,
            "models": URL(fileURLWithPath: modelPath).deletingLastPathComponent().lastPathComponent
                + "/" + URL(fileURLWithPath: modelPath).lastPathComponent,
            "batch_concurrency": 1,
            "load_ms": loadMs, "reload_ms": reloadMs,
            "short_samples": samples.count, "long_samples": longSamples.count,
            "short_seconds": Double(samples.count) / 16000,
            "long_seconds": Double(longSamples.count) / 16000,
            "short_processing_ms": first.processingMs, "repeat_processing_ms": second.processingMs,
            "long_processing_ms": long.processingMs,
            "short_text": first.text, "long_text": long.text,
            "first_15s_text": prefix.text,
            "first_15s_word_count": prefixWords.count, "long_word_count": fullWords.count,
            "long_includes_content_beyond_first_15s": containsNewSuffix && fullWords.count > prefixWords.count,
            "post_reload_identical": reloaded.text == first.text,
            "repeat_and_post_long_identical": final.text == first.text && second.text == first.text,
            "peak_rss_bytes": usageOK ? usage.ru_maxrss : -1
        ]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outputPath))
        print(String(decoding: data, as: UTF8.self))
    }
}
