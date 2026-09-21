import CryptoKit
import Foundation
import ZIPFoundation

/// Downloads, authenticates and installs Orukeet once, then reuses it offline.
/// A cache is specific to the archive, platform and OS build. Installation never
/// replaces an existing directory or deletes a caller-provided archive.
public actor OrukeetModelStore {
    public struct State: Sendable, Equatable {
        public enum Phase: String, Sendable {
            case checkingCache, downloading, verifying, extracting, compiling, ready
        }
        public let phase: Phase
        public let fraction: Double?

        public init(phase: Phase, fraction: Double? = nil) {
            self.phase = phase
            self.fraction = fraction
        }
    }

    public enum StoreError: Error, LocalizedError, Equatable {
        case busy
        case applicationSupportUnavailable
        case invalidHTTPResponse
        case httpStatus(Int)
        case invalidInstallation(String)
        case unsafeArchive(String)

        public var errorDescription: String? {
            switch self {
            case .busy: "An Orukeet installation is already in progress"
            case .applicationSupportUnavailable: "Application Support is unavailable"
            case .invalidHTTPResponse: "The model download did not return an HTTP response"
            case .httpStatus(let status): "The model download returned HTTP \(status)"
            case .invalidInstallation(let detail): "The installed Orukeet cache is invalid: \(detail)"
            case .unsafeArchive(let detail): "The Orukeet archive cannot be extracted safely: \(detail)"
            }
        }
    }

    typealias Downloader = @Sendable (URLSession, URL) async throws -> (URL, URLResponse)
    typealias Compiler = @Sendable (URL, URL) throws -> Void
    private let configuredRoot: URL?
    private let session: URLSession
    private let bundle: OrukeetBundle
    private let cacheKey: String
    private let download: Downloader
    private let compile: Compiler
    private var installing = false
    private static let receiptName = "orukeet-installation.json"
    private static let sidecars = ["parakeet_vocab.json", "bundle.json", "LICENSE-WEIGHTS", "NOTICE.md", "COREML-NOTICE.txt"]

    /// A custom root must be a dedicated model-cache directory. Model caches are
    /// excluded from backups because they can be downloaded and compiled again.
    public init(rootDirectory: URL? = nil, session: URLSession = .shared) {
        self.configuredRoot = rootDirectory
        self.session = session
        self.bundle = .int8
        self.cacheKey = Self.currentCacheKey
        self.download = Self.downloadArchive
        self.compile = { try OrukeetLocalModels.compilePackages(from: $0, to: $1) }
    }

    // Synthetic archives and compiler outputs exercise the complete transaction
    // without downloading or copying real model weights in unit tests.
    init(rootDirectory: URL, session: URLSession = .shared, bundle: OrukeetBundle,
         cacheKey: String = "test-os", download: @escaping Downloader = OrukeetModelStore.downloadArchive,
         compile: @escaping Compiler) {
        self.configuredRoot = rootDirectory
        self.session = session
        self.bundle = bundle
        self.cacheKey = cacheKey
        self.download = download
        self.compile = compile
    }

    /// Returns a structurally validated, identity-matched installation without
    /// accessing the network. Invalid cache metadata or structure is reported,
    /// not replaced. Compiled file sizes are checked without rehashing weights.
    public func installedDirectory() throws -> URL? {
        try Task.checkCancellation()
        let directory = try destination()
        guard try Self.exists(directory) else { return nil }
        try validateInstallation(directory)
        return directory
    }

    /// Download the pinned public archive when no valid local cache exists.
    /// Progress callbacks run on the store's executor, not the main actor.
    public func install(progress: (@Sendable (State) -> Void)? = nil) async throws -> URL {
        try await install(archive: nil, progress: progress)
    }

    /// Install an existing archive in place. This never modifies or removes it.
    public func install(fromArchive archive: URL,
                        progress: (@Sendable (State) -> Void)? = nil) async throws -> URL {
        try await install(archive: archive, progress: progress)
    }

    private func install(archive suppliedArchive: URL?,
                         progress: (@Sendable (State) -> Void)?) async throws -> URL {
        try Task.checkCancellation()
        guard !installing else { throw StoreError.busy }
        installing = true
        defer { installing = false }
        progress?(State(phase: .checkingCache))
        var root = try rootDirectory()
        let files = FileManager.default
        try files.createDirectory(at: root, withIntermediateDirectories: true)
        try Self.requireDirectory(root)
        var resourceValues = URLResourceValues()
        resourceValues.isExcludedFromBackup = true
        try root.setResourceValues(resourceValues)
        if let installed = try installedDirectory() {
            progress?(State(phase: .ready, fraction: 1))
            return installed
        }
        let workspace = root.appendingPathComponent(".orukeet-work-\(UUID().uuidString)", isDirectory: true)
        try files.createDirectory(at: workspace, withIntermediateDirectories: false)
        defer { try? files.removeItem(at: workspace) }
        do {
            let archive: URL
            if let suppliedArchive {
                archive = suppliedArchive
            } else {
                progress?(State(phase: .downloading))
                let (temporary, response) = try await download(session, bundle.url)
                // URLSession's temporary download belongs to this request only.
                defer { try? files.removeItem(at: temporary) }
                try Task.checkCancellation()
                guard let http = response as? HTTPURLResponse else { throw StoreError.invalidHTTPResponse }
                guard http.statusCode == 200 else { throw StoreError.httpStatus(http.statusCode) }
                archive = workspace.appendingPathComponent("download.zip")
                try files.moveItem(at: temporary, to: archive)
            }
            progress?(State(phase: .verifying))
            try bundle.verifyArchive(at: archive)
            let extracted = workspace.appendingPathComponent("extracted", isDirectory: true)
            progress?(State(phase: .extracting, fraction: 0))
            try Self.extract(archive, into: extracted, archiveRoot: bundle.archiveRoot, progress: progress)
            let source = extracted.appendingPathComponent(bundle.archiveRoot, isDirectory: true)
            // Attribution is mandatory in this pinned portable distribution.
            _ = try Self.sidecarHashes(in: source)
            let compiled = workspace.appendingPathComponent("compiled", isDirectory: true)
            progress?(State(phase: .compiling))
            try Task.checkCancellation()
            try compile(source, compiled)
            try Task.checkCancellation()
            let receipt = Receipt(formatVersion: 1, archiveSHA256: bundle.sha256,
                                  archiveBytes: bundle.bytes, cacheKey: cacheKey,
                                  archiveURL: bundle.url.absoluteString,
                                  installedAt: ISO8601DateFormatter().string(from: Date()),
                                  sidecarSHA256: try Self.sidecarHashes(in: compiled),
                                  compiledFiles: try Self.componentInventory(in: compiled))
            try JSONEncoder().encode(receipt).write(
                to: compiled.appendingPathComponent(Self.receiptName), options: .atomic)
            try validateInstallation(compiled)
            try Task.checkCancellation()
            let destination = try destination()
            do {
                try files.moveItem(at: compiled, to: destination)
            } catch {
                // A second store may have completed the identical transaction.
                // Reuse its verified result; never remove or replace it.
                guard let winner = try installedDirectory() else { throw error }
                progress?(State(phase: .ready, fraction: 1))
                return winner
            }
            progress?(State(phase: .ready, fraction: 1))
            return destination
        } catch {
            if Task.isCancelled { throw CancellationError() }
            throw error
        }
    }

    private static func downloadArchive(_ session: URLSession, _ url: URL) async throws -> (URL, URLResponse) {
        var request = URLRequest(url: url)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        return try await session.download(for: request)
    }

    private func rootDirectory() throws -> URL {
        if let configuredRoot { return configuredRoot }
        guard let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            throw StoreError.applicationSupportUnavailable
        }
        return support.appendingPathComponent("Orukeet/models", isDirectory: true)
    }

    private func destination() throws -> URL {
        try rootDirectory().appendingPathComponent("int8-\(bundle.sha256)-\(cacheKey)", isDirectory: true)
    }

    private static var currentCacheKey: String {
        #if targetEnvironment(simulator)
        let platform = "ios-simulator"
        #elseif os(iOS)
        let platform = "ios"
        #else
        let platform = "macos"
        #endif
        #if arch(arm64)
        let architecture = "arm64"
        #else
        let architecture = "x86_64"
        #endif
        let osBuild = SHA256.hash(data: Data(ProcessInfo.processInfo.operatingSystemVersionString.utf8))
            .map { String(format: "%02x", $0) }.joined().prefix(16)
        return "v1-\(platform)-\(architecture)-\(osBuild)"
    }

    private struct Receipt: Codable {
        let formatVersion: Int
        let archiveSHA256: String
        let archiveBytes: Int64
        let cacheKey: String
        let archiveURL: String
        let installedAt: String
        let sidecarSHA256: [String: String]
        let compiledFiles: [String: Int64]
    }

    private func validateInstallation(_ directory: URL) throws {
        do {
            try Self.requireDirectory(directory)
            let path = directory.appendingPathComponent(Self.receiptName)
            try Self.requireRegularFile(path, maximumBytes: 1_048_576)
            let receipt = try JSONDecoder().decode(Receipt.self, from: Data(contentsOf: path))
            guard receipt.formatVersion == 1, receipt.archiveSHA256 == bundle.sha256,
                  receipt.archiveBytes == bundle.bytes, receipt.cacheKey == cacheKey,
                  receipt.archiveURL == bundle.url.absoluteString,
                  receipt.sidecarSHA256 == (try Self.sidecarHashes(in: directory)),
                  receipt.compiledFiles == (try Self.componentInventory(in: directory)) else {
                throw StoreError.invalidInstallation("receipt or component mismatch")
            }
            _ = try OrukeetLocalModels.vocabulary(in: directory)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw StoreError.invalidInstallation(error.localizedDescription)
        }
    }

    private static func exists(_ url: URL) throws -> Bool {
        do {
            _ = try FileManager.default.attributesOfItem(atPath: url.path)
            return true
        } catch {
            let underlying = error as NSError
            if underlying.domain == NSCocoaErrorDomain,
               underlying.code == CocoaError.Code.fileReadNoSuchFile.rawValue { return false }
            throw error
        }
    }

    private static func requireDirectory(_ url: URL) throws {
        let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard values.isDirectory == true, values.isSymbolicLink != true else {
            throw StoreError.invalidInstallation("expected directory: \(url.lastPathComponent)")
        }
    }

    private static func requireRegularFile(_ url: URL, maximumBytes: Int) throws {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true,
              let bytes = values.fileSize, bytes > 0, bytes <= maximumBytes else {
            throw StoreError.invalidInstallation("expected nonempty file: \(url.lastPathComponent)")
        }
    }

    private static func sidecarHashes(in directory: URL) throws -> [String: String] {
        var hashes: [String: String] = [:]
        for name in sidecars {
            try Task.checkCancellation()
            let path = directory.appendingPathComponent(name)
            try requireRegularFile(path, maximumBytes: 8_388_608)
            hashes[name] = SHA256.hash(data: try Data(contentsOf: path)).map { String(format: "%02x", $0) }.joined()
        }
        return hashes
    }

    private static func componentInventory(in directory: URL) throws -> [String: Int64] {
        var inventory: [String: Int64] = [:]
        for component in OrukeetLocalModels.componentNames {
            let model = directory.appendingPathComponent("\(component).mlmodelc", isDirectory: true)
            try requireDirectory(model)
            // Foundation can enumerate /var URLs as /private/var. Carry relative
            // names through traversal so receipts survive the staging rename.
            var pending = [(model, "\(component).mlmodelc")]
            var fileCount = 0
            while let (parent, relativeParent) = pending.popLast() {
                try Task.checkCancellation()
                for path in try FileManager.default.contentsOfDirectory(
                    at: parent, includingPropertiesForKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]) {
                    let values = try path.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
                    guard values.isSymbolicLink != true else { throw StoreError.invalidInstallation("symbolic link in compiled model") }
                    let relative = relativeParent + "/" + path.lastPathComponent
                    if values.isDirectory == true { pending.append((path, relative)) }
                    else if values.isRegularFile == true, let size = values.fileSize, size > 0 {
                        inventory[relative] = Int64(size)
                        fileCount += 1
                    } else { throw StoreError.invalidInstallation("invalid compiled model payload") }
                }
            }
            guard fileCount > 0 else { throw StoreError.invalidInstallation("empty compiled model: \(component)") }
        }
        return inventory
    }

    // Authenticate first; this also rejects traversal, links, aliases, ambiguous
    // duplicate paths and decompression bombs before writing any archive entry.
    static func extract(_ url: URL, into directory: URL, archiveRoot: String,
                        progress: (@Sendable (State) -> Void)? = nil) throws {
        let archive = try Archive(url: url, accessMode: .read)
        var entries: [Entry] = []
        for entry in archive {
            try Task.checkCancellation()
            guard entries.count < 1_024 else { throw StoreError.unsafeArchive("entry count") }
            entries.append(entry)
        }
        guard !entries.isEmpty else { throw StoreError.unsafeArchive("entry count") }
        let maximumBytes: UInt64 = 1_500_000_000
        var total: UInt64 = 0
        var paths = Set<String>()
        var aliases: [String: String] = [:]
        var filePaths = Set<String>()
        for entry in entries {
            try Task.checkCancellation()
            guard entry.type != .symlink else { throw StoreError.unsafeArchive("symbolic link") }
            let path = entry.type == .directory && entry.path.hasSuffix("/") ? String(entry.path.dropLast()) : entry.path
            let parts = path.split(separator: "/", omittingEmptySubsequences: false)
            guard parts.first == Substring(archiveRoot), !parts.isEmpty, parts.count <= 32,
                  path.utf8.count <= 4_096, parts.allSatisfy({ $0.utf8.count <= 255 }),
                  !parts.contains(where: { $0.isEmpty || $0 == "." || $0 == ".." }),
                  !path.contains("\\"), !path.contains(":"),
                  path.rangeOfCharacter(from: .controlCharacters) == nil,
                  paths.insert(path).inserted else { throw StoreError.unsafeArchive("invalid or duplicate path") }
            for end in 1...parts.count {
                let prefix = parts.prefix(end).joined(separator: "/")
                let key = prefix.precomposedStringWithCanonicalMapping.lowercased()
                if let original = aliases[key], original != prefix { throw StoreError.unsafeArchive("aliased path") }
                aliases[key] = prefix
            }
            if entry.type == .file { filePaths.insert(path) }
            if entry.type == .directory, entry.uncompressedSize != 0 {
                throw StoreError.unsafeArchive("directory payload")
            }
            guard entry.uncompressedSize <= maximumBytes - total else { throw StoreError.unsafeArchive("expanded size") }
            total += entry.uncompressedSize
        }
        for path in paths {
            var parent = path
            while let slash = parent.lastIndex(of: "/") {
                parent = String(parent[..<slash])
                guard !filePaths.contains(parent) else { throw StoreError.unsafeArchive("file used as directory") }
            }
        }
        let files = FileManager.default
        try files.createDirectory(at: directory, withIntermediateDirectories: false)
        var extractedBytes: UInt64 = 0
        for entry in entries {
            try Task.checkCancellation()
            let target = directory.appendingPathComponent(entry.path)
            if entry.type == .directory {
                try files.createDirectory(at: target, withIntermediateDirectories: true)
                continue
            }
            try files.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: target, options: .withoutOverwriting)
            let output = try FileHandle(forWritingTo: target)
            defer { try? output.close() }
            var written: UInt64 = 0
            let checksum = try archive.extract(entry, bufferSize: 1_048_576) { data in
                try Task.checkCancellation()
                guard UInt64(data.count) <= entry.uncompressedSize - written else {
                    throw StoreError.unsafeArchive("entry exceeds declared size")
                }
                try output.write(contentsOf: data)
                written += UInt64(data.count)
                extractedBytes += UInt64(data.count)
                progress?(State(phase: .extracting, fraction: total == 0 ? 1 : Double(extractedBytes) / Double(total)))
            }
            guard written == entry.uncompressedSize, checksum == entry.checksum else {
                throw StoreError.unsafeArchive("entry size or checksum")
            }
            try output.close()
        }
    }
}
