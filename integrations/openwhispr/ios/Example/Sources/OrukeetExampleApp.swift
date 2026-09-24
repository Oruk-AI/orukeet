import AVFAudio
import Combine
import SwiftUI
import UIKit
import UniformTypeIdentifiers

@main
struct OrukeetExampleApp: App {
    var body: some Scene { WindowGroup { RecorderView() } }
}

@MainActor
private struct RecorderView: View {
    @StateObject private var model = ExampleModel()
    @Environment(\.scenePhase) private var scenePhase
    @State private var importing = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("Record, then transcribe locally in English or any of Orukeet’s 25 languages.")
                        .foregroundStyle(.secondary)
                    Text(model.status).accessibilityIdentifier("status")
                    if model.isBusy {
                        if let progress = model.progress {
                            ProgressView(value: progress).accessibilityIdentifier("installProgress")
                        } else { ProgressView() }
                    }
                    if let error = model.errorMessage {
                        Text(error).foregroundStyle(.red).accessibilityIdentifier("errorMessage")
                    }
                    if model.microphoneDenied {
                        Button("Open microphone settings") {
                            if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
                        }
                    }
                    if !model.isPrepared {
                        Button(model.isInstalled ? "Prepare Orukeet" : "Download and prepare Orukeet") {
                            model.installAndPrepare()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.isBusy)
                        .accessibilityIdentifier("installButton")
                    }
                    Button(model.activity == .recording ? "Stop and transcribe" : "Record") {
                        if model.activity == .recording { model.stopAndTranscribe() }
                        else { model.startRecording() }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!model.canRecord && model.activity != .recording)
                    .accessibilityIdentifier("recordButton")
                    HStack {
                        Button("Import audio") { importing = true }
                            .disabled(!model.canRecord)
                            .accessibilityIdentifier("importButton")
                        if model.isBusy || model.activity == .recording {
                            Button("Cancel", role: .cancel) { model.cancel() }
                                .disabled(model.activity == .cancelling)
                                .accessibilityIdentifier("cancelButton")
                        }
                    }
                    if model.hasTestFixture {
                        Button("Transcribe test fixture") { model.transcribeTestFixture() }
                            .disabled(!model.canRecord)
                            .accessibilityIdentifier("fixtureButton")
                    }
                    if model.hasRetryableRecording {
                        HStack {
                            Button("Retry recording") { model.retryRecording() }
                                .accessibilityIdentifier("retryRecordingButton")
                            Button("Discard recording", role: .destructive) {
                                model.cancel(reason: "Recording discarded.")
                            }
                        }
                        .disabled(!model.canRecord)
                    }
                    Divider()
                    Text("Transcript").font(.headline)
                    if model.transcript.isEmpty {
                        Text("Your transcript will appear here.").foregroundStyle(.secondary)
                    } else {
                        Text(model.transcript).textSelection(.enabled)
                            .accessibilityIdentifier("transcriptText")
                        Button("Copy transcript") { UIPasteboard.general.string = model.transcript }
                    }
                    Text("Completed recordings: \(model.completedRecordings)")
                        .font(.caption).foregroundStyle(.secondary)
                        .accessibilityIdentifier("completedRecordings")
                }
                .padding()
            }
            .navigationTitle("Orukeet")
            .fileImporter(isPresented: $importing, allowedContentTypes: [.audio]) { result in
                switch result {
                case .success(let file): model.transcribeImportedFile(file)
                case .failure(let error): model.report(error)
                }
            }
            .task { model.restore() }
            .onChange(of: scenePhase) { _, phase in
                if phase == .background { model.didEnterBackground() }
            }
            .onReceive(NotificationCenter.default.publisher(for: AVAudioSession.interruptionNotification).receive(on: RunLoop.main)) { note in
                if let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
                   AVAudioSession.InterruptionType(rawValue: raw) == .began {
                    model.cancelRecordingForSystemEvent("Recording interrupted. Tap Record to start again.")
                }
            }
            .onReceive(NotificationCenter.default.publisher(for: AVAudioSession.routeChangeNotification).receive(on: RunLoop.main)) { note in
                if let raw = note.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt,
                   let reason = AVAudioSession.RouteChangeReason(rawValue: raw),
                   reason == .oldDeviceUnavailable || reason == .newDeviceAvailable {
                    model.cancelRecordingForSystemEvent("Audio device changed. Tap Record to start again.")
                }
            }
            .onReceive(NotificationCenter.default.publisher(for: AVAudioSession.mediaServicesWereResetNotification).receive(on: RunLoop.main)) { _ in
                model.cancelRecordingForSystemEvent("Audio services restarted. Tap Record to start again.")
            }
        }
    }
}
