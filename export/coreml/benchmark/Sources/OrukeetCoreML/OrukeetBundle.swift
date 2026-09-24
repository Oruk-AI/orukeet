import CryptoKit
import Foundation

/// Immutable portable model identity used by OrukeetModelStore. Applications
/// with their own installer may verify the archive here before extraction and
/// compilation. This descriptor itself never accesses the network.
public struct OrukeetBundle: Sendable {
    public let url: URL
    public let bytes: Int64
    public let sha256: String
    public let archiveRoot: String

    /// Orukeet r3 for English and all 25 supported languages: symmetric INT8
    /// encoder weights, greedy v3 joint, and unchanged FP16 decoder components.
    public static let int8 = OrukeetBundle(
        url: URL(string: "https://huggingface.co/oruk/orukeet/resolve/419d7f79e290127e202a0f610509868d314743eb/coreml/orukeet-r3-coreml-int8sym-encoder-only-experimental-20260920-deflated.zip?download=true")!,
        bytes: 554_985_744,
        sha256: "24df9ff76f00f86f9ae1fd601cbbcab1d1eac98c7e8107de67444a7858d88b8b",
        archiveRoot: "orukeet-r3-coreml-int8sym-encoder-only-experimental-20260920")

    public enum VerificationError: Error, Equatable {
        case wrongSize(expected: Int64, actual: Int64)
        case wrongChecksum
    }

    /// Bounded-memory SHA-256 verification. Run off the main actor before ZIP
    /// extraction; the embedded bundle.json alone is not a trusted checksum.
    public func verifyArchive(at file: URL) throws {
        try Task.checkCancellation()
        let stream = try FileHandle(forReadingFrom: file)
        defer { try? stream.close() }
        var hasher = SHA256()
        var actualBytes: Int64 = 0
        // FileHandle's NSData bridge may autorelease each read. Drain it per
        // chunk: releasing only after the loop retains an entire model archive
        // on some Apple Foundation versions, despite the 1 MiB read size.
        while try autoreleasepool(invoking: {
            try Task.checkCancellation()
            guard let chunk = try stream.read(upToCount: 1_048_576), !chunk.isEmpty else { return false }
            actualBytes += Int64(chunk.count)
            guard actualBytes <= bytes else {
                throw VerificationError.wrongSize(expected: bytes, actual: actualBytes)
            }
            hasher.update(data: chunk)
            return true
        }) {}
        guard actualBytes == bytes else {
            throw VerificationError.wrongSize(expected: bytes, actual: actualBytes)
        }
        guard hasher.finalize().map({ String(format: "%02x", $0) }).joined() == sha256 else {
            throw VerificationError.wrongChecksum
        }
    }
}
