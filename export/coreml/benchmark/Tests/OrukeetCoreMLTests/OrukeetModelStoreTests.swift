import CryptoKit
import Foundation
import Testing
import ZIPFoundation
@testable import OrukeetCoreML

private enum StoreTestFailure: Error, Equatable { case unexpectedDownload, compilation }

private final class StoreProgressLog: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [OrukeetModelStore.State] = []
    func append(_ state: OrukeetModelStore.State) {
        lock.lock()
        defer { lock.unlock() }
        storage.append(state)
    }
    var phases: [OrukeetModelStore.State.Phase] {
        lock.lock()
        defer { lock.unlock() }
        return storage.map(\.phase)
    }
}

private actor SuspendedStoreDownload {
    let archive: URL
    private var started = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private var pending: CheckedContinuation<Void, Never>?
    init(archive: URL) { self.archive = archive }
    func download(_ session: URLSession, _ url: URL) async throws -> (URL, URLResponse) {
        await withCheckedContinuation { continuation in
            pending = continuation
            started = true
            waiters.forEach { $0.resume() }
            waiters.removeAll()
        }
        let temporary = archive.deletingLastPathComponent().appendingPathComponent(UUID().uuidString)
        try FileManager.default.copyItem(at: archive, to: temporary)
        return (temporary, HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)!)
    }
    func waitUntilStarted() async {
        if started { return }
        await withCheckedContinuation { waiters.append($0) }
    }
    func finish() { pending?.resume(); pending = nil }
}

struct OrukeetModelStoreTests {
    private struct Fixture {
        let root: URL
        let archive: URL
        let cache: URL
        let bundle: OrukeetBundle
    }

    private func fixture(extraEntries: [(String, Entry.EntryType, Data)] = []) throws -> Fixture {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("orukeet-store-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        let archiveURL = root.appendingPathComponent("synthetic.zip")
        let archive = try Archive(url: archiveURL, accessMode: .create)
        var entries: [(String, Entry.EntryType, Data)] = []
        let vocabulary = Dictionary(uniqueKeysWithValues: (0..<8192).map { (String($0), "token\($0)") })
        entries.append(("synthetic/parakeet_vocab.json", .file, try JSONEncoder().encode(vocabulary)))
        for name in ["bundle.json", "LICENSE-WEIGHTS", "NOTICE.md", "COREML-NOTICE.txt"] {
            entries.append(("synthetic/\(name)", .file, Data("synthetic fixture \(name)".utf8)))
        }
        for name in OrukeetLocalModels.componentNames {
            entries.append(("synthetic/\(name).mlpackage/model", .file, Data("synthetic component \(name)".utf8)))
        }
        for (path, type, data) in entries + extraEntries {
            try archive.addEntry(with: path, type: type, uncompressedSize: Int64(data.count),
                                 compressionMethod: .deflate) { position, size in
                data.subdata(in: Int(position)..<Int(position) + size)
            }
        }
        let data = try Data(contentsOf: archiveURL)
        let bundle = OrukeetBundle(url: URL(string: "https://example.invalid/orukeet.zip")!,
                                   bytes: Int64(data.count),
                                   sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
                                   archiveRoot: "synthetic")
        return Fixture(root: root, archive: archiveURL, cache: root.appendingPathComponent("cache"), bundle: bundle)
    }

    private static func compileSynthetic(_ source: URL, _ destination: URL) throws {
        try OrukeetLocalModels.compilePackages(from: source, to: destination) { package in
            let temporary = destination.deletingLastPathComponent().appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false)
            try FileManager.default.copyItem(at: package.appendingPathComponent("model"),
                                            to: temporary.appendingPathComponent("synthetic.bin"))
            return temporary
        }
    }

    private func store(_ fixture: Fixture, cacheKey: String = "test-os",
                       download: @escaping OrukeetModelStore.Downloader = { _, _ in throw StoreTestFailure.unexpectedDownload },
                       compile: @escaping OrukeetModelStore.Compiler = OrukeetModelStoreTests.compileSynthetic) -> OrukeetModelStore {
        OrukeetModelStore(rootDirectory: fixture.cache, bundle: fixture.bundle, cacheKey: cacheKey,
                          download: download, compile: compile)
    }

    @Test func installsArchiveAndReusesVerifiedCacheOfflineAcrossStoreInstances() async throws {
        let f = try fixture()
        defer { try? FileManager.default.removeItem(at: f.root) }
        let states = StoreProgressLog()
        let initial = store(f)
        #expect(try await initial.installedDirectory() == nil)
        let installed = try await initial.install(fromArchive: f.archive, progress: { states.append($0) })
        #expect(FileManager.default.fileExists(atPath: f.archive.path))
        #expect(states.phases.first == .checkingCache && states.phases.last == .ready)
        #expect(states.phases.contains(.verifying) && states.phases.contains(.extracting) && states.phases.contains(.compiling))
        #expect(!states.phases.contains(.downloading))
        let receipt = installed.appendingPathComponent("orukeet-installation.json")
        let before = try Data(contentsOf: receipt)
        #expect(try f.cache.resourceValues(forKeys: [.isExcludedFromBackupKey]).isExcludedFromBackup == true)
        // Every install/reuse repairs the backup flag, including older roots.
        var cache = f.cache
        var values = URLResourceValues()
        values.isExcludedFromBackup = false
        try cache.setResourceValues(values)
        let offline = store(f)
        #expect(try await offline.install() == installed)
        let refreshedCacheURL = URL(fileURLWithPath: f.cache.path)
        #expect(try refreshedCacheURL.resourceValues(forKeys: [.isExcludedFromBackupKey]).isExcludedFromBackup == true)
        #expect(try await offline.installedDirectory() == installed)
        #expect(try Data(contentsOf: receipt) == before)
        for name in ["bundle.json", "LICENSE-WEIGHTS", "NOTICE.md", "COREML-NOTICE.txt"] {
            #expect(try String(contentsOf: installed.appendingPathComponent(name), encoding: .utf8) == "synthetic fixture \(name)")
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: f.cache.path) == [installed.lastPathComponent])
    }

    @Test func downloadRejectsHTTPErrorAndRemovesOnlyItsTemporaryFiles() async throws {
        let f = try fixture()
        defer { try? FileManager.default.removeItem(at: f.root) }
        let temporary = f.root.appendingPathComponent("download-temporary")
        let installer = store(f, download: { _, url in
            try Data("404".utf8).write(to: temporary)
            return (temporary, HTTPURLResponse(url: url, statusCode: 404, httpVersion: nil, headerFields: nil)!)
        })
        await #expect(throws: OrukeetModelStore.StoreError.httpStatus(404)) { try await installer.install() }
        #expect(!FileManager.default.fileExists(atPath: temporary.path))
        #expect(FileManager.default.fileExists(atPath: f.archive.path))
        #expect(try FileManager.default.contentsOfDirectory(atPath: f.cache.path).isEmpty)
    }

    @Test func changedArchiveIsRejectedBeforeExtractionAndPreservesCallerInput() async throws {
        let f = try fixture()
        defer { try? FileManager.default.removeItem(at: f.root) }
        var altered = try Data(contentsOf: f.archive)
        altered[0] ^= 1
        try altered.write(to: f.archive)
        let installer = store(f)
        await #expect(throws: OrukeetBundle.VerificationError.wrongChecksum) {
            try await installer.install(fromArchive: f.archive)
        }
        #expect(try Data(contentsOf: f.archive) == altered)
        #expect(try FileManager.default.contentsOfDirectory(atPath: f.cache.path).isEmpty)
    }

    @Test func extractionRejectsCRCDisagreementEvenWhenZIPParsingSucceeds() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let archiveURL = root.appendingPathComponent("crc.zip")
        let marker = Data("synthetic_crc_payload_0123456789".utf8)
        do {
            let archive = try Archive(url: archiveURL, accessMode: .create)
            try archive.addEntry(with: "synthetic/file", type: .file, uncompressedSize: Int64(marker.count)) { offset, size in
                marker.subdata(in: Int(offset)..<Int(offset) + size)
            }
        }
        var bytes = try Data(contentsOf: archiveURL)
        let position = try #require(bytes.range(of: marker)).lowerBound
        bytes[position] ^= 1
        try bytes.write(to: archiveURL)
        #expect(throws: OrukeetModelStore.StoreError.unsafeArchive("entry size or checksum")) {
            try OrukeetModelStore.extract(archiveURL, into: root.appendingPathComponent("output"), archiveRoot: "synthetic")
        }
    }

    @Test func simultaneousRequestsAreRejectedWhileTheDownloadFinishesNormally() async throws {
        let f = try fixture()
        defer { try? FileManager.default.removeItem(at: f.root) }
        let backend = SuspendedStoreDownload(archive: f.archive)
        let installer = store(f, download: { try await backend.download($0, $1) })
        let first = Task { try await installer.install() }
        await backend.waitUntilStarted()
        await #expect(throws: OrukeetModelStore.StoreError.busy) { try await installer.install() }
        await backend.finish()
        let installed = try await first.value
        #expect(try await installer.installedDirectory() == installed)
        #expect(try FileManager.default.contentsOfDirectory(atPath: f.root.path).sorted() == ["cache", "synthetic.zip"])
    }

    @Test func cancellationDuringExtractionCleansTransactionAndPreservesInputForRetry() async throws {
        let f = try fixture()
        defer { try? FileManager.default.removeItem(at: f.root) }
        let installer = store(f)
        let task = Task {
            try await installer.install(fromArchive: f.archive) { state in
                if state.phase == .extracting, let fraction = state.fraction, fraction > 0 {
                    withUnsafeCurrentTask { $0?.cancel() }
                }
            }
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(FileManager.default.fileExists(atPath: f.archive.path))
        #expect(try FileManager.default.contentsOfDirectory(atPath: f.cache.path).isEmpty)
        let installed = try await installer.install(fromArchive: f.archive)
        #expect(try await installer.installedDirectory() == installed)
    }

    @Test func failedCompilationLeavesNoPublishedCacheAndCanRetry() async throws {
        let f = try fixture()
        defer { try? FileManager.default.removeItem(at: f.root) }
        let broken = store(f, compile: { _, destination in
            try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: false)
            try Data("partial".utf8).write(to: destination.appendingPathComponent("partial"))
            throw StoreTestFailure.compilation
        })
        await #expect(throws: StoreTestFailure.compilation) { try await broken.install(fromArchive: f.archive) }
        #expect(try FileManager.default.contentsOfDirectory(atPath: f.cache.path).isEmpty)
        let retry = store(f)
        _ = try await retry.install(fromArchive: f.archive)
        #expect(try await retry.installedDirectory() != nil)
    }

    @Test func corruptExistingCacheIsReportedAndNeverDeletedOrRedownloaded() async throws {
        let f = try fixture()
        defer { try? FileManager.default.removeItem(at: f.root) }
        let installer = store(f)
        let installed = try await installer.install(fromArchive: f.archive)
        let license = installed.appendingPathComponent("LICENSE-WEIGHTS")
        try Data("modified".utf8).write(to: license)
        await #expect(throws: OrukeetModelStore.StoreError.self) { try await installer.install() }
        #expect(try String(contentsOf: license, encoding: .utf8) == "modified")
        #expect(FileManager.default.fileExists(atPath: f.archive.path))
    }

    @Test func missingCompiledComponentAndMismatchedReceiptAreRejected() async throws {
        for corruption in ["component", "receipt"] {
            let f = try fixture()
            defer { try? FileManager.default.removeItem(at: f.root) }
            let installer = store(f)
            let installed = try await installer.install(fromArchive: f.archive)
            if corruption == "component" {
                try FileManager.default.removeItem(at: installed.appendingPathComponent("Encoder.mlmodelc"))
            } else {
                try Data("{}".utf8).write(to: installed.appendingPathComponent("orukeet-installation.json"))
            }
            await #expect(throws: OrukeetModelStore.StoreError.self) { try await installer.installedDirectory() }
        }
    }

    @Test func osCacheKeysKeepExistingInstallationsSeparate() async throws {
        let f = try fixture()
        defer { try? FileManager.default.removeItem(at: f.root) }
        let old = try await store(f, cacheKey: "previous-os").install(fromArchive: f.archive)
        let nextStore = store(f, cacheKey: "new-os")
        #expect(try await nextStore.installedDirectory() == nil)
        let next = try await nextStore.install(fromArchive: f.archive)
        #expect(old != next)
        #expect(FileManager.default.fileExists(atPath: old.path))
    }

    @Test func unsafeArchiveEntriesAreRejectedBeforeExtraction() async throws {
        let cases: [[(String, Entry.EntryType, Data)]] = [
            [("synthetic/../escape", .file, Data([1]))],
            [("/absolute", .file, Data([1]))],
            [("synthetic/link", .symlink, Data("../../escape".utf8))],
            [("synthetic/NOTICE.md", .file, Data([1]))],
            [("synthetic/notice.md", .file, Data([1]))],
            [("synthetic/file", .file, Data([1])), ("synthetic/file/child", .file, Data([1]))],
            [("synthetic/a\\b", .file, Data([1]))],
        ]
        for extra in cases {
            let f = try fixture(extraEntries: extra)
            defer { try? FileManager.default.removeItem(at: f.root) }
            let installer = store(f)
            await #expect(throws: OrukeetModelStore.StoreError.self) { try await installer.install(fromArchive: f.archive) }
            #expect(try FileManager.default.contentsOfDirectory(atPath: f.cache.path).isEmpty)
            #expect(FileManager.default.fileExists(atPath: f.archive.path))
        }
    }
}
