import AVFAudio
import Foundation

/// A foreground, file-backed recorder. The SDK reads and validates the completed
/// file, so no microphone buffer or mutable decoder state crosses actors.
@MainActor
final class PCMRecorder {
    enum RecordingError: LocalizedError {
        case microphoneDenied
        case couldNotStart
        case notRecording

        var errorDescription: String? {
            switch self {
            case .microphoneDenied:
                "Microphone access is disabled. Allow it in Settings, or import an audio file."
            case .couldNotStart: "The microphone could not start. Check your audio device and try again."
            case .notRecording: "There is no active recording."
            }
        }
    }

    private var recorder: AVAudioRecorder?
    private var fileURL: URL?

    func start() async throws {
        try Task.checkCancellation()
        let granted = await AVAudioApplication.requestRecordPermission()
        try Task.checkCancellation()
        guard granted else {
            throw RecordingError.microphoneDenied
        }
        let session = AVAudioSession.sharedInstance()
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("orukeet-recording-\(UUID().uuidString).caf")
        do {
            try session.setCategory(.record, mode: .measurement, options: [.allowBluetooth])
            try session.setActive(true)
            let candidate = try AVAudioRecorder(url: file, settings: [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: 16_000.0,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsBigEndianKey: false,
                AVLinearPCMIsFloatKey: false,
            ])
            guard candidate.prepareToRecord(), candidate.record() else {
                candidate.stop()
                throw RecordingError.couldNotStart
            }
            recorder = candidate
            fileURL = file
        } catch {
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            try? FileManager.default.removeItem(at: file)
            throw error
        }
    }

    /// Transfers ownership of the completed file to the caller, which deletes it
    /// after transcription. stop() closes the CAF before the SDK opens it.
    func stop() throws -> URL {
        guard let recorder, let fileURL else { throw RecordingError.notRecording }
        recorder.stop()
        self.recorder = nil
        self.fileURL = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        return fileURL
    }

    func cancel() {
        recorder?.stop()
        recorder = nil
        if let fileURL { try? FileManager.default.removeItem(at: fileURL) }
        fileURL = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }
}
