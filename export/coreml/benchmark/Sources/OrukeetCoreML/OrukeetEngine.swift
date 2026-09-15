@preconcurrency import CoreML
import FluidAudio
import Foundation

public enum OrukeetModelError: Error, LocalizedError {
    case missingComponent(String)
    case invalidVocabulary

    public var errorDescription: String? {
        switch self {
        case .missingComponent(let path): "Missing Orukeet Core ML component: \(path)"
        case .invalidVocabulary: "Orukeet requires the complete 8192-token v3 vocabulary"
        }
    }
}

/// Loads an already-installed bundle directly, preserving Orukeet's identity.
/// FluidAudio's NVIDIA downloader/cache layout must not be used for this bundle.
public enum OrukeetLocalModels {
    /// Compile packages on the target device during installation, outside the
    /// transcription timer. Do not assume a macOS 26 cache works on macOS 14.
    public static func compilePackages(from source: URL, to destination: URL) throws {
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        for name in ["Preprocessor", "Encoder", "Decoder", "JointDecisionv3"] {
            let target = destination.appendingPathComponent("\(name).mlmodelc")
            guard !FileManager.default.fileExists(atPath: target.path) else {
                throw CocoaError(.fileWriteFileExists)
            }
            let compiled = try MLModel.compileModel(at: source.appendingPathComponent("\(name).mlpackage"))
            try FileManager.default.copyItem(at: compiled, to: target)
        }
        try FileManager.default.copyItem(
            at: source.appendingPathComponent("parakeet_vocab.json"),
            to: destination.appendingPathComponent("parakeet_vocab.json"))
    }

    public static func load(
        from directory: URL,
        encoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) throws -> AsrModels {
        func component(_ name: String, _ units: MLComputeUnits) throws -> MLModel {
            let path = directory.appendingPathComponent("\(name).mlmodelc")
            guard FileManager.default.fileExists(atPath: path.path) else {
                throw OrukeetModelError.missingComponent(path.path)
            }
            let config = MLModelConfiguration()
            config.computeUnits = units
            return try MLModel(contentsOf: path, configuration: config)
        }
        let data = try Data(contentsOf: directory.appendingPathComponent("parakeet_vocab.json"))
        let raw = try JSONDecoder().decode([String: String].self, from: data)
        var vocabulary: [Int: String] = [:]
        for (key, value) in raw {
            guard let id = Int(key), id >= 0, id < 8192, vocabulary[id] == nil else {
                throw OrukeetModelError.invalidVocabulary
            }
            vocabulary[id] = value
        }
        guard vocabulary.count == 8192 else { throw OrukeetModelError.invalidVocabulary }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        return try AsrModels(
            encoder: component("Encoder", encoderComputeUnits),
            preprocessor: component("Preprocessor", .cpuOnly),
            decoder: component("Decoder", .cpuAndNeuralEngine),
            joint: component("JointDecisionv3", .cpuAndNeuralEngine),
            configuration: config, vocabulary: vocabulary, version: .v3
        )
    }
}

/// TapTalk-compatible local engine. Input is mono 16 kHz PCM, as in ParakeetEngine.
/// The greedy bundle supports this API's unconditioned decoding. Use the baseline
/// bundle when integrating language hints or top-K vocabulary reranking.
public actor OrukeetEngine {
    private let modelDirectory: URL
    private let encoderComputeUnits: MLComputeUnits
    private var models: AsrModels?
    private var manager: AsrManager?

    public init(modelDirectory: URL, encoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine) {
        self.modelDirectory = modelDirectory
        self.encoderComputeUnits = encoderComputeUnits
    }

    public func ensureLoaded() throws {
        guard manager == nil else { return }
        let loaded = try OrukeetLocalModels.load(from: modelDirectory, encoderComputeUnits: encoderComputeUnits)
        models = loaded
        manager = AsrManager(config: .default, models: loaded)
    }

    public func unload() {
        manager = nil
        models = nil
    }

    public struct Output: Sendable {
        public let text: String
        public let processingMs: Double
    }

    public func transcribe(samples: [Float]) async throws -> Output {
        try ensureLoaded()
        guard let manager else { throw OrukeetModelError.missingComponent(modelDirectory.path) }
        var state = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
        let result = try await manager.transcribe(samples, decoderState: &state)
        return Output(
            text: result.text.trimmingCharacters(in: .whitespacesAndNewlines),
            processingMs: result.processingTime * 1000)
    }

    /// Sliding-window TDT streaming, with the same model and policy as FluidAudio.
    /// This is distinct from TapTalk's separate 120M EOU live-typing model.
    public func makeStreamingManager(
        config: SlidingWindowAsrConfig = .streaming
    ) async throws -> SlidingWindowAsrManager {
        try ensureLoaded()
        guard let models else { throw OrukeetModelError.missingComponent(modelDirectory.path) }
        let streaming = SlidingWindowAsrManager(config: config)
        try await streaming.loadModels(models)
        return streaming
    }
}
