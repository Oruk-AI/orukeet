import CryptoKit
import Foundation
import Testing
@testable import OrukeetCoreML

struct OrukeetBundleTests {
    @Test func checksumSpansMultipleFullChunksAndPartialTail() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        let data = Data((0..<(3 * 1_048_576 + 71)).map { UInt8(truncatingIfNeeded: $0) })
        let bundle = OrukeetBundle(url: file, bytes: Int64(data.count),
                                   sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
                                   archiveRoot: "test")
        try data.write(to: file)
        try bundle.verifyArchive(at: file)
        let stream = try FileHandle(forWritingTo: file)
        try stream.seek(toOffset: UInt64(data.count - 1))
        try stream.write(contentsOf: Data([0]))
        try stream.close()
        #expect(throws: OrukeetBundle.VerificationError.wrongChecksum) { try bundle.verifyArchive(at: file) }
    }

    @Test func trustedArchiveRejectsTruncationAndSameSizeCorruption() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        let bundle = OrukeetBundle(
            url: file, bytes: 3,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            archiveRoot: "test")
        try Data("abc".utf8).write(to: file)
        try bundle.verifyArchive(at: file)
        try Data("ab".utf8).write(to: file)
        #expect(throws: OrukeetBundle.VerificationError.wrongSize(expected: 3, actual: 2)) {
            try bundle.verifyArchive(at: file)
        }
        try Data("abcd".utf8).write(to: file)
        #expect(throws: OrukeetBundle.VerificationError.wrongSize(expected: 3, actual: 4)) {
            try bundle.verifyArchive(at: file)
        }
        try Data("abd".utf8).write(to: file)
        #expect(throws: OrukeetBundle.VerificationError.wrongChecksum) {
            try bundle.verifyArchive(at: file)
        }
    }
}
