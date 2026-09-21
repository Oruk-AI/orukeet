@preconcurrency import AVFoundation
@preconcurrency import CoreML
import CryptoKit
import Darwin
import FluidAudio
import Foundation
import OrukeetCoreML

struct ComparisonError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

struct Options {
    let v2: URL
    let packages: URL
    let compiled: URL
    let audio: URL
    let fixtures: URL
    let output: URL
    let encoderUnits: String

    init(_ arguments: [String]) throws {
        guard arguments.count % 2 == 0 else { throw ComparisonError("Arguments require values") }
        var values: [String: String] = [:]
        let valid = Set(["--v2", "--orukeet-packages", "--orukeet-compiled", "--audio", "--fixtures", "--output", "--encoder-units"])
        for offset in stride(from: 0, to: arguments.count, by: 2) {
            let key = arguments[offset]
            guard valid.contains(key), values[key] == nil else { throw ComparisonError("Unknown or duplicate argument: \(key)") }
            values[key] = arguments[offset + 1]
        }
        func path(_ key: String) throws -> URL {
            guard let value = values[key] else { throw ComparisonError("Missing \(key)") }
            return URL(fileURLWithPath: value)
        }
        v2 = try path("--v2")
        packages = try path("--orukeet-packages")
        compiled = try path("--orukeet-compiled")
        audio = try path("--audio")
        fixtures = try path("--fixtures")
        output = try path("--output")
        encoderUnits = values["--encoder-units"] ?? "ane"
        guard ["cpu", "ane"].contains(encoderUnits) else { throw ComparisonError("Encoder units must be cpu or ane") }
    }
}

struct Fixture: Decodable {
    let path: String
    let sha256: String
    let samples: Int
    let reference: String
}
struct Fixtures: Decodable { let fixtures: [Fixture] }

struct Measurement: Codable {
    let model: String
    let audio: String
    let audioSHA256: String
    let audioSeconds: Double
    let text: String?
    let wallMs: Double?
    let engineMs: Double?
    let tokenIDs: [Int]?
    let error: String?
}

struct Report: Encodable {
    let qualification = "Tiny reused English regression diagnostic; not a release benchmark or an iPhone qualification."
    let fluidAudioVersion = "0.15.5"
    let fluidAudioRevision = "19600a485baa4998812e4654b70d2bab8f2c9949"
    let v2Revision = "ee09c569f73759e6d44c9bd16766f477b2b36d39"
    let orukeetRevision = "43142dd1897f9ddadcd70173fcb5ff45c08aa951"
    let orukeetArchiveSHA256 = "beccdc6f18c4b10527a764f6e3ab12e3e11b969220c0cee175b3bb7eaa94290e"
    let sampleRate = 16000
    let channels = 1
    let mode = "batch"
    let parallelChunkConcurrency = 1
    let decoderState = "Fresh TdtDecoderState for every recording; model-specific blank ID from AsrModels.version"
    let preprocessorUnits = "cpuOnly"
    let decoderAndJointUnits = "cpuAndNeuralEngine"
    let platform = ProcessInfo.processInfo.operatingSystemVersionString
    let timestamp = ISO8601DateFormatter().string(from: Date())
    let fixturesSHA256: String
    let encoderUnits: String
    var measurements: [Measurement] = []
    var modelErrors: [String: String] = [:]
}

func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

func readAudio(_ fixture: Fixture, from directory: URL) throws -> [Float] {
    guard fixture.path == URL(fileURLWithPath: fixture.path).lastPathComponent,
          fixture.samples > 0, fixture.samples <= 240000 else { throw ComparisonError("Invalid fixture path or duration") }
    let path = directory.appendingPathComponent(fixture.path)
    guard sha256(try Data(contentsOf: path)) == fixture.sha256 else { throw ComparisonError("Audio hash mismatch: \(fixture.path)") }
    let file = try AVAudioFile(forReading: path)
    guard file.processingFormat.sampleRate == 16000, file.processingFormat.channelCount == 1,
          file.length == Int64(fixture.samples),
          let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))
    else { throw ComparisonError("Expected sealed 16 kHz mono audio: \(fixture.path)") }
    try file.read(into: buffer)
    guard Int(buffer.frameLength) == fixture.samples, let channel = buffer.floatChannelData?[0]
    else { throw ComparisonError("Audio decode incomplete: \(fixture.path)") }
    let samples = Array(UnsafeBufferPointer(start: channel, count: fixture.samples))
    guard samples.allSatisfy(\.isFinite) else { throw ComparisonError("Nonfinite audio") }
    return samples
}

// All paths are local. Do not use AsrModels.load/ModelHub, whose recovery can download mutable assets.
func loadV2(_ directory: URL, encoderUnits: MLComputeUnits) throws -> AsrModels {
    let data = try Data(contentsOf: directory.appendingPathComponent("parakeet_vocab.json"))
    guard sha256(data) == "57019fe3c745772ca83a1b048a4bb951cd51329504ea33d4d83316b96e279a97" else {
        throw ComparisonError("V2 vocabulary differs from pinned publication")
    }
    let raw = try JSONDecoder().decode([String: String].self, from: data)
    var vocabulary: [Int: String] = [:]
    for (key, token) in raw {
        guard let id = Int(key), key == String(id), id >= 0 else { throw ComparisonError("Invalid vocabulary index") }
        vocabulary[id] = token
    }
    func component(_ name: String, units: MLComputeUnits) throws -> MLModel {
        let path = directory.appendingPathComponent("\(name).mlmodelc")
        guard FileManager.default.fileExists(atPath: path.path) else { throw ComparisonError("Missing local component: \(path.path)") }
        let config = MLModelConfiguration()
        config.computeUnits = units
        return try MLModel(contentsOf: path, configuration: config)
    }
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine
    return try AsrModels(
        encoder: component("Encoder", units: encoderUnits),
        preprocessor: component("Preprocessor", units: .cpuOnly),
        decoder: component("Decoder", units: .cpuAndNeuralEngine),
        joint: component("JointDecision", units: .cpuAndNeuralEngine),
        configuration: config, vocabulary: vocabulary, version: .v2)
}

func persist(_ report: Report, to path: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
    try encoder.encode(report).write(to: path, options: .atomic)
}

func executeModel(_ label: String, options: Options, fixtures: [Fixture], report: inout Report) async throws {
    let units: MLComputeUnits = options.encoderUnits == "cpu" ? .cpuOnly : .cpuAndNeuralEngine
    let models: AsrModels
    if label == "parakeet_v2" {
        models = try loadV2(options.v2, encoderUnits: units)
    } else {
        // A fresh target ensures this run cannot silently reuse compiled models with unknown provenance.
        try OrukeetLocalModels.compilePackages(from: options.packages, to: options.compiled)
        models = try OrukeetLocalModels.load(from: options.compiled, encoderComputeUnits: units)
    }
    guard models.preprocessor.configuration.computeUnits == .cpuOnly,
          models.encoder?.configuration.computeUnits == units,
          models.decoder.configuration.computeUnits == .cpuAndNeuralEngine,
          models.joint.configuration.computeUnits == .cpuAndNeuralEngine else {
        throw ComparisonError("Actual model compute policy differs from the matched diagnostic policy")
    }
    let manager = AsrManager(config: ASRConfig(parallelChunkConcurrency: 1))
    try await manager.loadModels(models)
    for fixture in fixtures {
        do {
            let samples = try readAudio(fixture, from: options.audio)
            var state = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
            let start = ContinuousClock.now
            let result = try await manager.transcribe(samples, decoderState: &state)
            let elapsed = start.duration(to: .now)
            let wallMs = Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15
            report.measurements.append(Measurement(
                model: label, audio: fixture.path, audioSHA256: fixture.sha256,
                audioSeconds: Double(samples.count) / 16000, text: result.text,
                wallMs: wallMs, engineMs: result.processingTime * 1000,
                tokenIDs: result.tokenTimings?.map(\.tokenId), error: nil))
        } catch {
            report.measurements.append(Measurement(
                model: label, audio: fixture.path, audioSHA256: fixture.sha256,
                audioSeconds: Double(fixture.samples) / 16000, text: nil,
                wallMs: nil, engineMs: nil, tokenIDs: nil, error: String(describing: error)))
        }
        try persist(report, to: options.output)
    }
    await manager.cleanup()
}

do {
    let options = try Options(Array(CommandLine.arguments.dropFirst()))
    let fixtureBytes = try Data(contentsOf: options.fixtures)
    guard sha256(fixtureBytes) == "46d5db4c0a5b92557bf378bfeb9a7b150029714d449dc462d1a1b945475428d2" else {
        throw ComparisonError("Fixture manifest is not the sealed English16 set")
    }
    let fixtures = try JSONDecoder().decode(Fixtures.self, from: fixtureBytes).fixtures
    guard fixtures.count == 16, Set(fixtures.map(\.path)).count == 16 else { throw ComparisonError("Require exactly 16 unique fixtures") }
    var report = Report(fixturesSHA256: sha256(fixtureBytes), encoderUnits: options.encoderUnits)
    try persist(report, to: options.output)
    for label in ["parakeet_v2", "orukeet"] {
        do { try await executeModel(label, options: options, fixtures: fixtures, report: &report) }
        catch {
            let message = String(describing: error)
            report.modelErrors[label] = message
            let present = Set(report.measurements.filter { $0.model == label }.map(\.audio))
            for fixture in fixtures where !present.contains(fixture.path) {
                report.measurements.append(Measurement(
                    model: label, audio: fixture.path, audioSHA256: fixture.sha256,
                    audioSeconds: Double(fixture.samples) / 16000, text: nil,
                    wallMs: nil, engineMs: nil, tokenIDs: nil,
                    error: "Model load or execution failed: \(message)"))
            }
        }
        try persist(report, to: options.output)
    }
    guard report.modelErrors.isEmpty, report.measurements.count == 32,
          report.measurements.allSatisfy({ $0.error == nil }) else {
        throw ComparisonError("Comparison incomplete; retained exact errors and partial transcripts in \(options.output.path)")
    }
    print("Completed 32 paired transcriptions: \(options.output.path)")
} catch {
    FileHandle.standardError.write(Data("EnglishComparison: \(error)\n".utf8))
    exit(1)
}
