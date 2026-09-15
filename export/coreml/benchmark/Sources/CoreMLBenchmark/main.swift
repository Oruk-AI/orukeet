@preconcurrency import AVFoundation
@preconcurrency import CoreML
import CryptoKit
import Darwin
import FluidAudio
import Foundation
import OrukeetCoreML

struct BenchmarkError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

struct Options {
    var models: [(String, URL)] = []
    var audio: [URL] = []
    var repetitions = 10
    var warmup = 2
    var mode = "batch"
    var encoderUnits = "ane"
    var output = URL(fileURLWithPath: "benchmark.json")
    var paced = false
    var concurrency = 4

    init(_ args: [String]) throws {
        var i = 0
        while i < args.count {
            let flag = args[i]
            if flag == "--paced" {
                paced = true
                i += 1
                continue
            }
            guard i + 1 < args.count else { throw BenchmarkError("Missing value for \(flag)") }
            let value = args[i + 1]
            switch flag {
            case "--models":
                let parts = value.split(separator: "=", maxSplits: 1).map(String.init)
                guard parts.count == 2 else { throw BenchmarkError("Use --models label=/path/to/models") }
                models.append((parts[0], URL(fileURLWithPath: parts[1])))
            case "--audio": audio.append(URL(fileURLWithPath: value))
            case "--repeat":
                guard let n = Int(value), n > 0 else { throw BenchmarkError("--repeat must be positive") }
                repetitions = n
            case "--warmup":
                guard let n = Int(value), n >= 0 else { throw BenchmarkError("--warmup cannot be negative") }
                warmup = n
            case "--concurrency":
                guard let n = Int(value), n > 0 else { throw BenchmarkError("--concurrency must be positive") }
                concurrency = n
            case "--mode": mode = value
            case "--encoder-units": encoderUnits = value
            case "--output": output = URL(fileURLWithPath: value)
            default: throw BenchmarkError("Unknown option \(flag)")
            }
            i += 2
        }
        guard !models.isEmpty, !audio.isEmpty else {
            throw BenchmarkError("Required: --models label=/path --audio /path/to/16k-mono.wav")
        }
        guard Set(models.map(\.0)).count == models.count else { throw BenchmarkError("Model labels must be unique") }
        guard ["batch", "streaming"].contains(mode) else { throw BenchmarkError("Invalid --mode") }
        guard ["ane", "gpu", "cpu", "all"].contains(encoderUnits) else {
            throw BenchmarkError("Invalid --encoder-units")
        }
        guard !paced || mode == "streaming" else { throw BenchmarkError("--paced requires streaming mode") }
    }
}

struct AudioFixture {
    let url: URL
    let samples: [Float]
    let sha256: String
    init(_ url: URL) throws {
        self.url = url
        let file = try AVAudioFile(forReading: url)
        guard file.processingFormat.sampleRate == 16000, file.processingFormat.channelCount == 1,
            file.length > 0, file.length <= Int64(UInt32.max),
            let buffer = AVAudioPCMBuffer(
                pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))
        else { throw BenchmarkError("Expected nonempty 16 kHz mono audio: \(url.path)") }
        try file.read(into: buffer)
        guard let channel = buffer.floatChannelData?[0] else { throw BenchmarkError("Cannot read float audio") }
        samples = Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
        sha256 = SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
    }
}

struct Loaded {
    let label: String
    let directory: String
    let models: AsrModels
    let manager: AsrManager
    let loadMs: Double
}

func seconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
}

func chipName() -> String {
    var size = 0
    guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0 else { return "unknown" }
    var buffer = [CChar](repeating: 0, count: size)
    guard sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0) == 0 else { return "unknown" }
    return String(decoding: buffer.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

func load(_ label: String, _ directory: URL, _ options: Options) async throws -> Loaded {
    let start = ContinuousClock.now
    let units: MLComputeUnits =
        switch options.encoderUnits {
        case "cpu": .cpuOnly
        case "gpu": .cpuAndGPU
        case "all": .all
        default: .cpuAndNeuralEngine
        }
    let models = try OrukeetLocalModels.load(from: directory, encoderComputeUnits: units)
    let manager = AsrManager(config: ASRConfig(parallelChunkConcurrency: options.concurrency))
    try await manager.loadModels(models)
    return Loaded(
        label: label, directory: directory.path, models: models, manager: manager,
        loadMs: seconds(start.duration(to: .now)) * 1000)
}

struct Update: Codable {
    let elapsedMs: Double
    let text: String
    let confirmed: Bool
}

struct Measurement: Codable {
    let model: String
    let audio: String
    let audioSHA256: String
    let audioSeconds: Double
    let repetition: Int
    let wallMs: Double
    let engineMs: Double?
    let finalizationMs: Double?
    let firstTextMs: Double?
    let text: String
    let tokenIDs: [Int]?
    let updates: [Update]?
    let thermalState: Int
}

func audioChunk(_ samples: [Float], offset: Int, count: Int) throws -> sending AVAudioPCMBuffer {
    guard let format = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1),
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count)),
        let destination = buffer.floatChannelData?[0]
    else { throw BenchmarkError("Cannot allocate audio chunk") }
    buffer.frameLength = AVAudioFrameCount(count)
    samples.withUnsafeBufferPointer { source in
        if let base = source.baseAddress { destination.update(from: base.advanced(by: offset), count: count) }
    }
    return buffer
}

func measure(
    _ loaded: Loaded, _ fixture: AudioFixture, _ options: Options, _ repetition: Int
) async throws -> Measurement {
    let thermalState = ProcessInfo.processInfo.thermalState.rawValue
    if options.mode == "batch" {
        var state = TdtDecoderState.make(decoderLayers: await loaded.manager.decoderLayerCount)
        let start = ContinuousClock.now
        let result = try await loaded.manager.transcribe(fixture.samples, decoderState: &state)
        let wallMs = seconds(start.duration(to: .now)) * 1000
        return Measurement(
            model: loaded.label, audio: fixture.url.path, audioSHA256: fixture.sha256,
            audioSeconds: Double(fixture.samples.count) / 16000, repetition: repetition,
            wallMs: wallMs, engineMs: result.processingTime * 1000, finalizationMs: nil,
            firstTextMs: nil, text: result.text, tokenIDs: result.tokenTimings?.map(\.tokenId),
            updates: nil, thermalState: thermalState)
    }
    let manager = SlidingWindowAsrManager(config: .streaming)
    try await manager.loadModels(loaded.models)
    let stream = await manager.transcriptionUpdates
    let start = ContinuousClock.now
    let collector = Task { () -> [Update] in
        var updates: [Update] = []
        for await update in stream {
            updates.append(
                Update(
                    elapsedMs: seconds(start.duration(to: .now)) * 1000,
                    text: update.text, confirmed: update.isConfirmed))
        }
        return updates
    }
    try await manager.startStreaming(source: .microphone)
    for offset in stride(from: 0, to: fixture.samples.count, by: 1600) {
        let count = min(1600, fixture.samples.count - offset)
        if options.paced {
            try await ContinuousClock().sleep(until: start.advanced(by: .seconds(Double(offset + count) / 16000)))
        }
        await manager.streamAudio(try audioChunk(fixture.samples, offset: offset, count: count))
    }
    let endOfAudio = ContinuousClock.now
    let text = try await manager.finish()
    let end = ContinuousClock.now
    await manager.cleanup()
    collector.cancel()
    let updates = await collector.value
    return Measurement(
        model: loaded.label, audio: fixture.url.path, audioSHA256: fixture.sha256,
        audioSeconds: Double(fixture.samples.count) / 16000, repetition: repetition,
        wallMs: seconds(start.duration(to: end)) * 1000, engineMs: nil,
        finalizationMs: seconds(endOfAudio.duration(to: end)) * 1000,
        firstTextMs: updates.first(where: { !$0.text.isEmpty })?.elapsedMs,
        text: text, tokenIDs: nil, updates: updates, thermalState: thermalState)
}

struct ModelRecord: Codable {
    let label: String
    let directory: String
    let loadMs: Double
}
struct Report: Encodable {
    let schemaVersion = 1
    let fluidAudioVersion = "0.15.5"
    let createdAt: String
    let os: String
    let host: String
    let chip: String
    let processors: Int
    let memoryBytes: UInt64
    let mode: String
    let paced: Bool
    let encoderUnits: String
    let concurrency: Int
    let warmup: Int
    let repetitions: Int
    let models: [ModelRecord]
    let measurements: [Measurement]
}

@main struct Main {
    static func main() async throws {
        let options = try Options(Array(CommandLine.arguments.dropFirst()))
        let fixtures = try options.audio.map(AudioFixture.init)
        var models: [Loaded] = []
        for (label, directory) in options.models { models.append(try await load(label, directory, options)) }
        var measurements: [Measurement] = []
        for fixture in fixtures {
            for repetition in -options.warmup..<options.repetitions {
                // Rotate every model through every position, then reverse. This
                // avoids leaving the middle model fixed in three-model trials.
                let offset = (repetition % models.count + models.count) % models.count
                let rotated = Array(models[offset...]) + Array(models[..<offset])
                let ordered = repetition.isMultiple(of: 2) ? rotated : Array(rotated.reversed())
                for model in ordered {
                    let row = try await measure(model, fixture, options, repetition)
                    if repetition >= 0 { measurements.append(row) }
                    FileHandle.standardError.write(
                        Data(
                            "\(model.label) \(fixture.url.lastPathComponent) run=\(repetition) \(String(format: "%.1f", row.wallMs)) ms\n"
                                .utf8))
                }
            }
        }
        let info = ProcessInfo.processInfo
        let report = Report(
            createdAt: ISO8601DateFormatter().string(from: Date()),
            os: info.operatingSystemVersionString, host: info.hostName, chip: chipName(),
            processors: info.processorCount, memoryBytes: info.physicalMemory,
            mode: options.mode, paced: options.paced, encoderUnits: options.encoderUnits,
            concurrency: options.concurrency, warmup: options.warmup, repetitions: options.repetitions,
            models: models.map { ModelRecord(label: $0.label, directory: $0.directory, loadMs: $0.loadMs) },
            measurements: measurements)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: options.output, options: .atomic)
    }
}
