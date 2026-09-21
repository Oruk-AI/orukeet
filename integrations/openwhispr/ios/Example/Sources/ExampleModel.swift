import Combine
import Foundation
import OrukeetCoreML

@MainActor
final class ExampleModel: ObservableObject {
    enum Activity { case idle, restoring, installing, preparing, requestingPermission, recording, transcribing, cancelling }

    @Published private(set) var activity: Activity = .idle
    @Published private(set) var isInstalled = false
    @Published private(set) var isPrepared = false
    @Published private(set) var status = "Download Orukeet once to transcribe locally."
    @Published private(set) var progress: Double?
    @Published private(set) var transcript = ""
    @Published private(set) var completedRecordings = 0
    @Published private(set) var errorMessage: String?
    @Published private(set) var microphoneDenied = false
    @Published private(set) var hasRetryableRecording = false

    private let transcriber: OrukeetTranscriber
    private let recorder = PCMRecorder()
    private let testArchive: URL?
    private let testFixture: URL?
    private var operation: Task<Void, Never>?
    private var restored = false
    private var releaseAfterOperation = false
    private var cancellationMessage = "Cancelled."
    private var retainedRecording: URL?

    var isBusy: Bool { activity != .idle && activity != .recording }
    var canRecord: Bool { activity == .idle && isPrepared }
    var hasTestFixture: Bool { testFixture != nil }

    init(arguments: [String] = ProcessInfo.processInfo.arguments) {
        func url(after flag: String) -> URL? {
            guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else { return nil }
            return URL(fileURLWithPath: arguments[index + 1])
        }
        // These explicit launch arguments let CI exercise the real installer and
        // inference with a preauthenticated ZIP. Production uses the SDK defaults.
        testArchive = url(after: "--orukeet-archive")
        testFixture = url(after: "--orukeet-fixture")
        let store = OrukeetModelStore(rootDirectory: url(after: "--orukeet-test-root"))
        transcriber = OrukeetTranscriber(store: store)
    }

    func restore() {
        guard !restored else { return }
        restored = true
        perform(.restoring, status: "Checking installed model…") { [self] in
            isInstalled = try await transcriber.installedDirectory() != nil
            if isInstalled {
                activity = .preparing
                status = "Preparing Orukeet…"
                try await transcriber.prepare()
                try Task.checkCancellation()
                isPrepared = true
                status = "Ready. Audio stays on this device."
            } else {
                status = "Download Orukeet once to transcribe locally."
            }
        }
    }

    func installAndPrepare() {
        perform(.installing, status: "Installing Orukeet…") { [self] in
            let update: @Sendable (OrukeetModelStore.State) -> Void = { [weak self] state in
                Task { @MainActor [weak self] in
                    guard let self, self.activity == .installing else { return }
                    self.progress = state.fraction
                    switch state.phase {
                    case .checkingCache: self.status = "Checking installed model…"
                    case .downloading: self.status = "Downloading Orukeet…"
                    case .verifying: self.status = "Verifying model…"
                    case .extracting: self.status = "Extracting model…"
                    case .compiling: self.status = "Compiling for this device…"
                    case .ready: self.status = "Model installed."
                    }
                }
            }
            if let testArchive {
                _ = try await transcriber.install(fromArchive: testArchive, progress: update)
            } else {
                _ = try await transcriber.install(progress: update)
            }
            try Task.checkCancellation()
            isInstalled = true
            activity = .preparing
            progress = nil
            status = "Preparing Orukeet…"
            try await transcriber.prepare()
            try Task.checkCancellation()
            isPrepared = true
            status = "Ready. Audio stays on this device."
        }
    }

    func startRecording() {
        guard canRecord else { return }
        discardRetainedRecording()
        perform(.requestingPermission, status: "Opening microphone…") { [self] in
            try await recorder.start()
            try Task.checkCancellation()
            activity = .recording
            status = "Recording. Tap Stop to transcribe."
        }
    }

    func stopAndTranscribe() {
        guard activity == .recording else { return }
        do {
            let file = try recorder.stop()
            activity = .idle
            transcribe(file, ownedRecording: true)
        } catch {
            recorder.cancel()
            activity = .idle
            report(error)
        }
    }

    func transcribeImportedFile(_ file: URL) { transcribe(file, ownedRecording: false) }

    func retryRecording() {
        guard let retainedRecording, canRecord else { return }
        transcribe(retainedRecording, ownedRecording: true)
    }

    func transcribeTestFixture() {
        if let testFixture { transcribe(testFixture, ownedRecording: false) }
    }

    private func transcribe(_ file: URL, ownedRecording: Bool) {
        guard canRecord else {
            if ownedRecording { try? FileManager.default.removeItem(at: file) }
            return
        }
        if ownedRecording {
            if retainedRecording != file { discardRetainedRecording() }
            retainedRecording = file
            hasRetryableRecording = false
        }
        perform(.transcribing, status: "Transcribing on device…") { [self] in
            let scoped = file.startAccessingSecurityScopedResource()
            defer {
                if scoped { file.stopAccessingSecurityScopedResource() }
            }
            do {
                let result = try await transcriber.transcribe(fileURL: file)
                try Task.checkCancellation()
                transcript = result.text
                completedRecordings += 1
                if ownedRecording { discardRetainedRecording() }
                status = result.text.isEmpty ? "No speech detected. Ready to record again." : "Transcription complete. Ready to record again."
            } catch {
                if ownedRecording && !Task.isCancelled && !(error is CancellationError) {
                    hasRetryableRecording = true
                }
                throw error
            }
        }
    }

    func cancel(reason: String = "Cancelled. You can try again.", releaseModels: Bool = false) {
        cancellationMessage = reason
        recorder.cancel()
        errorMessage = nil
        if let operation {
            releaseAfterOperation = releaseAfterOperation || releaseModels
            activity = .cancelling
            status = "Cancelling…"
            operation.cancel()
        } else {
            discardRetainedRecording()
            activity = .idle
            status = reason
            if releaseModels {
                isPrepared = false
                perform(.cancelling, status: reason) { [self] in await transcriber.unload() }
            }
        }
    }

    func cancelRecordingForSystemEvent(_ reason: String) {
        guard activity == .recording || activity == .requestingPermission else { return }
        cancel(reason: reason)
    }

    func didEnterBackground() {
        cancel(reason: "Paused in background. Prepare Orukeet to continue.", releaseModels: true)
    }

    func report(_ error: Error) {
        errorMessage = error.localizedDescription
        status = "Try again when ready."
    }

    private func discardRetainedRecording() {
        if let retainedRecording { try? FileManager.default.removeItem(at: retainedRecording) }
        retainedRecording = nil
        hasRetryableRecording = false
    }

    private func perform(_ activity: Activity, status: String, body: @escaping @MainActor () async throws -> Void) {
        guard operation == nil, self.activity != .recording else { return }
        self.activity = activity
        self.status = status
        errorMessage = nil
        microphoneDenied = false
        progress = nil
        operation = Task { [self] in
            do {
                try await body()
            } catch is CancellationError {
                recorder.cancel()
                self.status = cancellationMessage
            } catch {
                if case PCMRecorder.RecordingError.microphoneDenied = error { microphoneDenied = true }
                report(error)
            }
            // Wait for file reading/inference to finish before deleting a
            // cancelled recording. Failed imports never replace this owned file.
            if Task.isCancelled { discardRetainedRecording() }
            if releaseAfterOperation {
                await transcriber.unload()
                isPrepared = false
                releaseAfterOperation = false
            }
            if self.activity != .recording { self.activity = .idle }
            progress = nil
            operation = nil
        }
    }
}
