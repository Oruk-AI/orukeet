@preconcurrency import AVFoundation
@preconcurrency import CoreML
import Darwin
import Foundation
import OrukeetCoreML

private struct Fixture: Codable {
    let id: String
    let path: String
    let sha256: String
}

private struct Configuration: Codable {
    let label: String
    let round: Int
    let runtime: String
    let modelProfile: String
    let modelDirectory: String
    let encoderUnits: String
    let concurrency: Int
    let warmups: Int
    let repetitions: Int
    let fixtures: [Fixture]
    let outputPath: String
}

private func peakRSS() -> Int64 {
    var usage = rusage()
    return getrusage(RUSAGE_SELF, &usage) == 0 ? Int64(usage.ru_maxrss) : -1
}

private func readAudio(_ fixture: Fixture) throws -> [Float] {
    let file = try AVAudioFile(forReading: URL(fileURLWithPath: fixture.path))
    guard file.processingFormat.sampleRate == 16000,
          file.processingFormat.channelCount == 1,
          file.length >= 4800,
          let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                        frameCapacity: AVAudioFrameCount(file.length)) else {
        throw NSError(domain: "Benchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid audio: \(fixture.id)"])
    }
    try file.read(into: buffer)
    guard let channel = buffer.floatChannelData?[0] else {
        throw NSError(domain: "Benchmark", code: 2)
    }
    return Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
}

@main
struct EngineBenchmark {
    static func main() async throws {
        guard CommandLine.arguments.count == 2 else {
            fatalError("Usage: EngineBenchmark CONFIG_JSON")
        }
        let config = try JSONDecoder().decode(Configuration.self, from: Data(contentsOf:
            URL(fileURLWithPath: CommandLine.arguments[1])))
        guard config.repetitions >= 15, config.warmups >= 3,
              ["ane", "gpu"].contains(config.encoderUnits) else { fatalError("Invalid protocol") }
        let audio = try config.fixtures.map(readAudio)
        let rssBeforeLoad = peakRSS()
        let engine = OrukeetEngine(modelDirectory: URL(fileURLWithPath: config.modelDirectory),
                                  encoderComputeUnits: config.encoderUnits == "gpu" ? .cpuAndGPU : .cpuAndNeuralEngine,
                                  batchConcurrency: config.concurrency)
        let loadStart = DispatchTime.now().uptimeNanoseconds
        try await engine.ensureLoaded()
        let loadMs = Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1_000_000
        let rssAfterLoad = peakRSS()
        var records: [[String: Any]] = []
        for repetition in 0..<(config.warmups + config.repetitions) {
            // Rotate clips within each repetition; reverse their base order in
            // round two. All configurations receive every fixture equally.
            let order = Array(config.fixtures.indices)
            let shifted = Array(order.dropFirst(repetition % order.count)) + Array(order.prefix(repetition % order.count))
            for index in shifted {
                let fixture = config.fixtures[index]
                let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
                let start = DispatchTime.now().uptimeNanoseconds
                let result = try await engine.transcribe(samples: audio[index])
                let wallMs = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
                guard wallMs.isFinite, result.processingMs.isFinite, !result.text.isEmpty else {
                    throw NSError(domain: "Benchmark", code: 3,
                                  userInfo: [NSLocalizedDescriptionKey: "Invalid result for \(fixture.id)"])
                }
                records.append([
                    "fixture": fixture.id, "audio_sha256": fixture.sha256,
                    "samples": audio[index].count, "audio_seconds": Double(audio[index].count) / 16000,
                    "repetition": repetition - config.warmups,
                    "warmup": repetition < config.warmups,
                    "wall_ms": wallMs, "processing_ms": result.processingMs,
                    "text": result.text, "thermal_before": thermalBefore,
                    "thermal_after": ProcessInfo.processInfo.thermalState.rawValue,
                    "process_peak_rss_bytes": peakRSS()
                ])
            }
        }
        let finalRSS = peakRSS()
        await engine.unload()
        let report: [String: Any] = [
            "label": config.label, "round": config.round,
            "runtime": config.runtime, "model_profile": config.modelProfile,
            "encoder_requested_compute_units": config.encoderUnits == "gpu" ? "cpuAndGPU" : "cpuAndNeuralEngine",
            "decoder_joint_requested_compute_units": "cpuAndNeuralEngine",
            "preprocessor_requested_compute_units": "cpuOnly",
            "batch_concurrency": config.concurrency, "warmups_per_fixture": config.warmups,
            "timed_repetitions_per_fixture": config.repetitions,
            "host_os": ProcessInfo.processInfo.operatingSystemVersionString,
            "host_physical_memory_bytes": ProcessInfo.processInfo.physicalMemory,
            "load_ms": loadMs, "peak_rss_bytes_before_load": rssBeforeLoad,
            "peak_rss_bytes_after_load": rssAfterLoad, "peak_rss_bytes_final": finalRSS,
            "records": records,
            "limitations": [
                "One Apple silicon Mac; not physical iPhone performance or exclusive ANE execution.",
                "Wall time surrounds the public engine call; model load and file decoding are excluded.",
                "Peak RSS is the whole process maximum and includes runtime/model/audio overhead.",
                "Load timings have uncontrolled Core ML cache history and are not cold-start guarantees."
            ]
        ]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: config.outputPath), options: .atomic)
        print("\(config.label) round \(config.round): \(records.count) calls, peak RSS \(finalRSS), output \(config.outputPath)")
    }
}
