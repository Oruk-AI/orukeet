import Foundation
import Testing
@testable import OrukeetCoreML

struct OrukeetBundleTests {
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
