import Foundation
import Testing
@testable import OrukeetCoreML

private actor ServiceStore: OrukeetModelInstalling {
    var directory: URL?
    private(set) var downloads = 0
    private(set) var imports = 0
    private(set) var checks = 0
    init(installed: Bool) {
        directory = installed ? URL(fileURLWithPath: "/service-test-models") : nil
    }
    func installedDirectory() -> URL? { checks += 1; return directory }
    func install(progress: (@Sendable (OrukeetModelStore.State) -> Void)?) -> URL {
        downloads += 1
        directory = URL(fileURLWithPath: "/service-test-models")
        return directory!
    }
    func install(fromArchive: URL, progress: (@Sendable (OrukeetModelStore.State) -> Void)?) -> URL {
        imports += 1
        directory = URL(fileURLWithPath: "/service-test-models")
        return directory!
    }
}

private actor ServiceEngine: OrukeetServing {
    private(set) var prepareCalls = 0
    private(set) var transcribeCalls = 0
    private(set) var unloadCalls = 0
    private(set) var lastSamples: [Float] = []
    private(set) var lastRate = 0
    private(set) var lastChannels = 0
    func prepare() { prepareCalls += 1 }
    func transcribe(samples: [Float], sampleRate: Int, channels: Int) -> OrukeetEngine.Output {
        transcribeCalls += 1
        lastSamples = samples
        lastRate = sampleRate
        lastChannels = channels
        return .init(text: "complete recording", processingMs: 2)
    }
    func unload() { unloadCalls += 1 }
}

private actor AudioReadGate {
    private var pending: CheckedContinuation<[Float], Never>?
    private var entered = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func read(_ url: URL) async -> [Float] {
        await withCheckedContinuation { continuation in
            pending = continuation
            entered = true
            waiters.forEach { $0.resume() }
            waiters.removeAll()
        }
    }
    func waitForRead() async {
        if entered { return }
        await withCheckedContinuation { waiters.append($0) }
    }
    func finish() {
        pending?.resume(returning: [Float](repeating: 0.25, count: 4_800))
        pending = nil
    }
}

struct OrukeetTranscriberTests {
    @Test func offlineRequestsNeverTriggerAnImplicitDownload() async {
        let store = ServiceStore(installed: false)
        let backend = ServiceEngine()
        let service = OrukeetTranscriber(store: store, makeEngine: { _ in backend })
        await #expect(throws: OrukeetTranscriber.ServiceError.modelNotInstalled) {
            try await service.prepare()
        }
        await #expect(throws: OrukeetTranscriber.ServiceError.modelNotInstalled) {
            try await service.transcribe(samples: [Float](repeating: 0, count: 4_800))
        }
        await #expect(throws: OrukeetTranscriber.ServiceError.modelNotInstalled) {
            try await service.transcribe(fileURL: URL(fileURLWithPath: "/not-read.wav"))
        }
        #expect(await store.downloads == 0)
        #expect(await backend.transcribeCalls == 0)
    }

    @Test func explicitInstallThenIndependentRecordingsReusePreparedBackend() async throws {
        let store = ServiceStore(installed: false)
        let backend = ServiceEngine()
        let samples: [Float] = [0.25, -0.5, 0.75]
        let service = OrukeetTranscriber(store: store, makeEngine: { _ in backend }, readAudio: { _ in samples })
        _ = try await service.install()
        try await service.prepare()
        #expect(try await service.transcribe(fileURL: URL(fileURLWithPath: "/recording.caf")).text == "complete recording")
        #expect(try await service.transcribe(samples: samples).text == "complete recording")
        #expect(await backend.lastSamples == samples)
        #expect(await backend.lastRate == 16_000)
        #expect(await backend.lastChannels == 1)
        #expect(await backend.prepareCalls == 1)
        #expect(await backend.transcribeCalls == 2)
        #expect(await store.downloads == 1)
        #expect(await store.checks == 1)
    }

    @Test func importedArchiveDoesNotUseDownloader() async throws {
        let store = ServiceStore(installed: false)
        let backend = ServiceEngine()
        let service = OrukeetTranscriber(store: store, makeEngine: { _ in backend })
        _ = try await service.install(fromArchive: URL(fileURLWithPath: "/caller-owned.zip"))
        try await service.prepare()
        #expect(await store.imports == 1)
        #expect(await store.downloads == 0)
    }

    @Test func cancellationDuringDecodeDiscardsSamplesAndDefersUnload() async throws {
        let store = ServiceStore(installed: true)
        let backend = ServiceEngine()
        let gate = AudioReadGate()
        let service = OrukeetTranscriber(store: store, makeEngine: { _ in backend }, readAudio: { await gate.read($0) })
        let request = Task { try await service.transcribe(fileURL: URL(fileURLWithPath: "/recording.caf")) }
        await gate.waitForRead()
        await service.unload()
        #expect(await backend.unloadCalls == 0)
        await #expect(throws: OrukeetTranscriber.ServiceError.busy) { try await service.prepare() }
        await #expect(throws: OrukeetTranscriber.ServiceError.busy) { try await service.install() }
        request.cancel()
        await gate.finish()
        await #expect(throws: CancellationError.self) { try await request.value }
        #expect(await backend.transcribeCalls == 0)
        #expect(await backend.unloadCalls == 1)
        try await service.prepare()
        #expect(try await service.transcribe(samples: [0]).text == "complete recording")
    }

    @Test func audioFailureReleasesAdmissionAndAllowsRetry() async throws {
        let store = ServiceStore(installed: true)
        let backend = ServiceEngine()
        let service = OrukeetTranscriber(store: store, makeEngine: { _ in backend }, readAudio: { _ in
            throw CocoaError(.fileReadCorruptFile)
        })
        await #expect(throws: CocoaError.self) {
            try await service.transcribe(fileURL: URL(fileURLWithPath: "/bad.wav"))
        }
        #expect(try await service.transcribe(samples: [0]).text == "complete recording")
        #expect(await backend.transcribeCalls == 1)
    }
}
