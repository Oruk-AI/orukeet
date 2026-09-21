@preconcurrency import AVFoundation
import Darwin
import Foundation
import OrukeetCoreML

private struct ParityFixture: Codable {
    let id: String
    let path: String
    let language: String
    let sha256: String
    let samples: Int
}

private struct ParityConfiguration: Codable {
    let runtime: String
    let modelDirectory: String
    let fixtures: [ParityFixture]
    let outputPath: String
}

@main
struct CorpusParity {
    static func main() async throws {
        guard CommandLine.arguments.count == 2 else { fatalError("Usage: CorpusParity CONFIG_JSON") }
        let config = try JSONDecoder().decode(ParityConfiguration.self,
            from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
        guard config.fixtures.count == 400,
              Set(config.fixtures.map(\.sha256)).count == 400,
              Set(config.fixtures.map(\.language)).count == 25 else { fatalError("Expected sealed 400-clip corpus") }
        let engine = OrukeetEngine(modelDirectory: URL(fileURLWithPath: config.modelDirectory), batchConcurrency: 1)
        try await engine.ensureLoaded()
        var records: [[String: Any]] = []
        for fixture in config.fixtures {
            let file = try AVAudioFile(forReading: URL(fileURLWithPath: fixture.path))
            guard file.processingFormat.sampleRate == 16000, file.processingFormat.channelCount == 1,
                  let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                               frameCapacity: AVAudioFrameCount(file.length)) else { fatalError("Invalid audio") }
            try file.read(into: buffer)
            guard Int(buffer.frameLength) == fixture.samples, let channel = buffer.floatChannelData?[0] else {
                fatalError("Decoded audio does not match manifest")
            }
            let samples = Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
            let result = try await engine.transcribe(samples: samples)
            records.append(["id": fixture.id, "language": fixture.language, "audio_sha256": fixture.sha256,
                            "samples": fixture.samples, "text": result.text])
            if records.count % 50 == 0 { print("\(config.runtime): \(records.count)/400") }
        }
        await engine.unload()
        var usage = rusage()
        let rss = getrusage(RUSAGE_SELF, &usage) == 0 ? Int64(usage.ru_maxrss) : -1
        let report: [String: Any] = ["runtime": config.runtime, "model_profile": "int8sym-encoder-only",
                                   "batch_concurrency": 1, "encoder_requested_compute_units": "cpuAndNeuralEngine",
                                   "records": records, "process_peak_rss_bytes": rss,
                                   "scope": "Exact runtime transcript regression with fixed existing INT8 weights; not an accuracy comparison or iPhone qualification"]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: config.outputPath), options: .atomic)
    }
}
