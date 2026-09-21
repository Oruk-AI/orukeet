@preconcurrency import CoreML
import FluidAudio
import Foundation

public enum OrukeetModelError: Error, LocalizedError, Equatable {
    case missingComponent(String)
    case invalidVocabulary
    case invalidAudioFormat(sampleRate: Int, channels: Int)
    case emptyAudio
    case audioTooShort(minimumSamples: Int)
    case nonFiniteSample(index: Int)
    case invalidBatchConcurrency
    case busy

    public var errorDescription: String? {
        switch self {
        case .missingComponent(let path): "Missing Orukeet Core ML component: \(path)"
        case .invalidVocabulary: "Orukeet requires the complete 8192-token v3 vocabulary"
        case .invalidAudioFormat(let rate, let channels):
            "Orukeet requires 16000 Hz mono PCM; received \(rate) Hz and \(channels) channels"
        case .emptyAudio: "The recording contains no audio samples"
        case .audioTooShort(let minimum): "Orukeet requires at least \(minimum) samples (300 ms at 16 kHz)"
        case .nonFiniteSample(let index): "Audio sample \(index) is not finite"
        case .invalidBatchConcurrency: "Batch concurrency must be at least one"
        case .busy: "This Orukeet engine is already transcribing a recording"
        }
    }
}

/// Loads an already-installed bundle directly, preserving Orukeet's identity.
/// FluidAudio's NVIDIA downloader/cache layout must not be used for this bundle.
public enum OrukeetLocalModels {
    static let componentNames = ["Preprocessor", "Encoder", "Decoder", "JointDecisionv3"]

    /// Compile verified packages on the target device during installation, outside
    /// the transcription timer and off the UI thread. This synchronous call stages
    /// all components beside the destination, then publishes the complete directory.
    /// The destination must not exist: use a versioned path for upgrades and switch
    /// the app's selected path only after this method succeeds. Existing installs
    /// are never overwritten. Compilation caches are specific to the target OS.
    public static func compilePackages(from source: URL, to destination: URL) throws {
        try compilePackages(from: source, to: destination, compiler: { try MLModel.compileModel(at: $0) })
    }

    // Compiler injection lets tests exercise failed installations without model weights.
    static func compilePackages(
        from source: URL, to destination: URL, compiler: (URL) throws -> URL
    ) throws {
        try Task.checkCancellation()
        let files = FileManager.default
        guard !files.fileExists(atPath: destination.path) else {
            throw CocoaError(.fileWriteFileExists)
        }
        _ = try vocabulary(in: source)
        for name in componentNames {
            let package = source.appendingPathComponent("\(name).mlpackage")
            var isDirectory: ObjCBool = false
            guard files.fileExists(atPath: package.path, isDirectory: &isDirectory), isDirectory.boolValue else {
                throw OrukeetModelError.missingComponent(package.path)
            }
        }
        let parent = destination.deletingLastPathComponent()
        try files.createDirectory(at: parent, withIntermediateDirectories: true)
        let staging = parent.appendingPathComponent(".orukeet-install-\(UUID().uuidString)", isDirectory: true)
        try files.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? files.removeItem(at: staging) }
        for name in componentNames {
            try Task.checkCancellation()
            let compiled = try compiler(source.appendingPathComponent("\(name).mlpackage"))
            // Core ML owns no persistent copy here; compilation returns a temporary URL.
            defer { try? files.removeItem(at: compiled) }
            try Task.checkCancellation()
            try files.copyItem(at: compiled, to: staging.appendingPathComponent("\(name).mlmodelc"))
        }
        try files.copyItem(
            at: source.appendingPathComponent("parakeet_vocab.json"),
            to: staging.appendingPathComponent("parakeet_vocab.json"))
        // Keep attribution and source identity available in an offline app cache.
        // Older local bundles may lack these; portable distribution validation is
        // performed before this device-local compilation step.
        for name in ["bundle.json", "LICENSE-WEIGHTS", "NOTICE.md", "COREML-NOTICE.txt"] {
            let sidecar = source.appendingPathComponent(name)
            if files.fileExists(atPath: sidecar.path) {
                try files.copyItem(at: sidecar, to: staging.appendingPathComponent(name))
            }
        }
        try Task.checkCancellation()
        // Same-parent rename publishes one complete install. moveItem fails if a
        // concurrent installer has already created the destination.
        try files.moveItem(at: staging, to: destination)
    }

    static func vocabulary(in directory: URL) throws -> [Int: String] {
        let path = directory.appendingPathComponent("parakeet_vocab.json")
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw OrukeetModelError.missingComponent(path.path)
        }
        let data = try Data(contentsOf: path)
        guard let raw = try? JSONDecoder().decode([String: String].self, from: data), raw.count == 8192 else {
            throw OrukeetModelError.invalidVocabulary
        }
        var vocabulary: [Int: String] = [:]
        for (key, value) in raw {
            guard let id = Int(key), key == String(id), id >= 0, id < 8192 else {
                throw OrukeetModelError.invalidVocabulary
            }
            vocabulary[id] = value
        }
        return vocabulary
    }

    public static func load(
        from directory: URL,
        encoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) throws -> AsrModels {
        try Task.checkCancellation()
        let vocabulary = try vocabulary(in: directory)
        func component(_ name: String, _ units: MLComputeUnits) throws -> MLModel {
            try Task.checkCancellation()
            let path = directory.appendingPathComponent("\(name).mlmodelc")
            guard FileManager.default.fileExists(atPath: path.path) else {
                throw OrukeetModelError.missingComponent(path.path)
            }
            let config = MLModelConfiguration()
            config.computeUnits = units
            return try MLModel(contentsOf: path, configuration: config)
        }
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

// Each recording receives fresh decoder state; the engine protects the manager
// across suspension points, since actor isolation alone does not serialize awaits.
protocol OrukeetBatchTranscribing: Sendable {
    func transcribe(_ samples: [Float]) async throws -> OrukeetEngine.Output
}

private struct FluidAudioBatchTranscriber: OrukeetBatchTranscribing {
    let manager: AsrManager

    func transcribe(_ samples: [Float]) async throws -> OrukeetEngine.Output {
        var state = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
        try Task.checkCancellation()
        let result = try await manager.transcribe(samples, decoderState: &state)
        return OrukeetEngine.Output(
            text: result.text.trimmingCharacters(in: .whitespacesAndNewlines),
            processingMs: result.processingTime * 1000)
    }
}

/// Local record-then-transcribe engine for iOS 17+ and macOS 14+.
/// Input is finite, mono 16 kHz PCM containing at least 300 ms of audio. Format
/// parameters describe the supplied samples; conversion/resampling is the caller's
/// responsibility. One active recording per engine is permitted; overlapping calls
/// throw `OrukeetModelError.busy`. Cancel the calling Task to cancel transcription.
/// Core ML work already executing may finish before cancellation is observed.
/// The greedy bundle supports unconditioned decoding. Use the baseline bundle
/// when integrating language hints or top-K vocabulary reranking.
public actor OrukeetEngine {
    private let modelDirectory: URL
    private let encoderComputeUnits: MLComputeUnits
    private let batchConcurrency: Int
    private var models: AsrModels?
    private var transcriber: (any OrukeetBatchTranscribing)?
    private var isTranscribing = false
    private var unloadWhenIdle = false

    /// One long-form chunk at a time limits mobile working memory. Applications
    /// may increase this after profiling their supported devices.
    public init(
        modelDirectory: URL,
        encoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        batchConcurrency: Int = 1
    ) {
        self.modelDirectory = modelDirectory
        self.encoderComputeUnits = encoderComputeUnits
        self.batchConcurrency = batchConcurrency
    }

    init(transcriber: any OrukeetBatchTranscribing) {
        self.modelDirectory = URL(fileURLWithPath: "/unused-test-models")
        self.encoderComputeUnits = .cpuOnly
        self.batchConcurrency = 1
        self.transcriber = transcriber
    }

    public func ensureLoaded() throws {
        try Task.checkCancellation()
        guard batchConcurrency > 0 else { throw OrukeetModelError.invalidBatchConcurrency }
        guard transcriber == nil else { return }
        let loaded = try OrukeetLocalModels.load(from: modelDirectory, encoderComputeUnits: encoderComputeUnits)
        try Task.checkCancellation()
        models = loaded
        transcriber = FluidAudioBatchTranscriber(
            manager: AsrManager(config: ASRConfig(parallelChunkConcurrency: batchConcurrency), models: loaded))
    }

    /// Release models when idle. During transcription, release is deferred until
    /// that request finishes (or observes cancellation); it never replaces an
    /// active manager with a second one.
    public func unload() {
        guard !isTranscribing else {
            unloadWhenIdle = true
            return
        }
        transcriber = nil
        models = nil
        unloadWhenIdle = false
    }

    public struct Output: Sendable {
        public let text: String
        public let processingMs: Double
    }

    public func transcribe(
        samples: [Float], sampleRate: Int = 16_000, channels: Int = 1
    ) async throws -> Output {
        try Task.checkCancellation()
        guard sampleRate == 16_000, channels == 1 else {
            throw OrukeetModelError.invalidAudioFormat(sampleRate: sampleRate, channels: channels)
        }
        guard !samples.isEmpty else { throw OrukeetModelError.emptyAudio }
        if let index = samples.firstIndex(where: { !$0.isFinite }) {
            throw OrukeetModelError.nonFiniteSample(index: index)
        }
        let minimum = ASRConstants.minimumRequiredSamples(forSampleRate: sampleRate)
        guard samples.count >= minimum else { throw OrukeetModelError.audioTooShort(minimumSamples: minimum) }
        guard !isTranscribing else { throw OrukeetModelError.busy }
        isTranscribing = true
        defer {
            isTranscribing = false
            if unloadWhenIdle { unload() }
        }
        try ensureLoaded()
        guard let transcriber else { throw OrukeetModelError.missingComponent(modelDirectory.path) }
        try Task.checkCancellation()
        let output = try await transcriber.transcribe(samples)
        try Task.checkCancellation()
        return output
    }

    /// Sliding-window TDT streaming, with the same model and policy as FluidAudio.
    /// This returned manager has its own lifecycle. Use a separate engine instance
    /// if an application also runs batch transcription; batch admission and unload
    /// cannot control a streaming manager once it has been returned to the caller.
    public func makeStreamingManager(
        config: SlidingWindowAsrConfig = .streaming
    ) async throws -> SlidingWindowAsrManager {
        try Task.checkCancellation()
        guard !isTranscribing else { throw OrukeetModelError.busy }
        try ensureLoaded()
        guard let models else { throw OrukeetModelError.missingComponent(modelDirectory.path) }
        let streaming = SlidingWindowAsrManager(config: config)
        try await streaming.loadModels(models)
        try Task.checkCancellation()
        return streaming
    }
}
