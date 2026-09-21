@preconcurrency import CoreML
import Foundation

protocol OrukeetModelInstalling: Sendable {
    func installedDirectory() async throws -> URL?
    func install(progress: (@Sendable (OrukeetModelStore.State) -> Void)?) async throws -> URL
    func install(fromArchive: URL, progress: (@Sendable (OrukeetModelStore.State) -> Void)?) async throws -> URL
}

extension OrukeetModelStore: OrukeetModelInstalling {}

protocol OrukeetServing: Sendable {
    func prepare() async throws
    func transcribe(samples: [Float], sampleRate: Int, channels: Int) async throws -> OrukeetEngine.Output
    func unload() async
}

extension OrukeetEngine: OrukeetServing {}

/// The application-facing record-then-transcribe service. Install explicitly
/// once, prepare when opening the recorder, and reuse across completed recordings.
/// Preparation and transcription never start a download. One active operation is
/// admitted; cancel and await it before submitting another recording.
public actor OrukeetTranscriber {
    public enum ServiceError: Error, LocalizedError, Equatable {
        case modelNotInstalled
        case busy

        public var errorDescription: String? {
            switch self {
            case .modelNotInstalled: "Install the Orukeet model before transcribing."
            case .busy: "Orukeet is already preparing or processing a recording."
            }
        }
    }

    private let store: any OrukeetModelInstalling
    private let modelDirectory: URL?
    private let makeEngine: @Sendable (URL) -> any OrukeetServing
    private let readAudio: @Sendable (URL) async throws -> [Float]
    private var engine: (any OrukeetServing)?
    private var isWorking = false
    private var unloadWhenIdle = false

    /// Omit modelDirectory to use the managed, verified installation. A supplied
    /// directory opts into an existing compiled cache; its provenance is the
    /// caller's responsibility. It is never replaced or downloaded by this service.
    public init(
        modelDirectory: URL? = nil,
        store: OrukeetModelStore = OrukeetModelStore(),
        encoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        batchConcurrency: Int = 4
    ) {
        self.modelDirectory = modelDirectory
        self.store = store
        self.makeEngine = { directory in
            OrukeetEngine(modelDirectory: directory, encoderComputeUnits: encoderComputeUnits,
                          batchConcurrency: batchConcurrency)
        }
        self.readAudio = Self.decodeRecording
    }

    init(
        store: any OrukeetModelInstalling,
        makeEngine: @escaping @Sendable (URL) -> any OrukeetServing,
        readAudio: @escaping @Sendable (URL) async throws -> [Float] = OrukeetTranscriber.decodeRecording
    ) {
        self.store = store
        self.modelDirectory = nil
        self.makeEngine = makeEngine
        self.readAudio = readAudio
    }

    /// Check the local installation without network access.
    public func installedDirectory() async throws -> URL? {
        try Task.checkCancellation()
        if let modelDirectory {
            let files = FileManager.default
            for component in OrukeetLocalModels.componentNames {
                var directory: ObjCBool = false
                guard files.fileExists(atPath: modelDirectory.appendingPathComponent("\(component).mlmodelc").path,
                                       isDirectory: &directory), directory.boolValue else { return nil }
            }
            _ = try OrukeetLocalModels.vocabulary(in: modelDirectory)
            return modelDirectory
        }
        return try await store.installedDirectory()
    }

    @discardableResult
    public func install(progress: (@Sendable (OrukeetModelStore.State) -> Void)? = nil) async throws -> URL {
        try await perform {
            if modelDirectory != nil { return try await requireInstalledDirectory() }
            return try await store.install(progress: progress)
        }
    }

    /// Import an already-downloaded archive through the same verified installer.
    /// The caller retains ownership of the archive, including on cancellation.
    @discardableResult
    public func install(
        fromArchive archive: URL,
        progress: (@Sendable (OrukeetModelStore.State) -> Void)? = nil
    ) async throws -> URL {
        try await perform {
            if modelDirectory != nil { return try await requireInstalledDirectory() }
            return try await store.install(fromArchive: archive, progress: progress)
        }
    }

    public func prepare() async throws {
        try await perform {
            let engine = try await currentEngine()
            try await engine.prepare()
        }
    }

    /// Decode/downmix/resample a completed recording (for example CAF, WAV or
    /// M4A) off the main actor, then transcribe its complete 16 kHz mono PCM.
    public func transcribe(fileURL: URL) async throws -> OrukeetEngine.Output {
        try await perform {
            let engine = try await currentEngine()
            let samples = try await readAudio(fileURL)
            try Task.checkCancellation()
            return try await engine.transcribe(samples: samples, sampleRate: 16_000, channels: 1)
        }
    }

    /// Preserve an application's existing normalized 16 kHz mono capture path.
    public func transcribe(
        samples: [Float], sampleRate: Int = 16_000, channels: Int = 1
    ) async throws -> OrukeetEngine.Output {
        try await perform {
            let engine = try await currentEngine()
            return try await engine.transcribe(samples: samples, sampleRate: sampleRate, channels: channels)
        }
    }

    /// Keep the verified installation on disk, and release only loaded models.
    /// Active operations finish or observe cancellation before release occurs.
    public func unload() async {
        guard !isWorking else {
            unloadWhenIdle = true
            return
        }
        isWorking = true
        unloadWhenIdle = true
        await finishOperation()
    }

    private func requireInstalledDirectory() async throws -> URL {
        guard let directory = try await installedDirectory() else { throw ServiceError.modelNotInstalled }
        return directory
    }

    private func currentEngine() async throws -> any OrukeetServing {
        // The service's model identity is immutable. Reuse its loaded engine
        // without reparsing vocabulary or walking the disk cache per recording.
        if let engine { return engine }
        let directory = try await requireInstalledDirectory()
        try Task.checkCancellation()
        let current = makeEngine(directory)
        engine = current
        return current
    }

    private func perform<T: Sendable>(_ operation: () async throws -> T) async throws -> T {
        try Task.checkCancellation()
        guard !isWorking else { throw ServiceError.busy }
        isWorking = true
        do {
            let result = try await operation()
            try Task.checkCancellation()
            await finishOperation()
            return result
        } catch {
            await finishOperation()
            throw error
        }
    }

    private func finishOperation() async {
        if unloadWhenIdle {
            let previous = engine
            engine = nil
            if let previous { await previous.unload() }
            unloadWhenIdle = false
        }
        isWorking = false
    }

    private nonisolated static func decodeRecording(_ url: URL) async throws -> [Float] {
        let reader = Task.detached(priority: .userInitiated) {
            try OrukeetAudio.readMono16k(from: url)
        }
        return try await withTaskCancellationHandler {
            let samples = try await reader.value
            try Task.checkCancellation()
            return samples
        } onCancel: {
            reader.cancel()
        }
    }
}
