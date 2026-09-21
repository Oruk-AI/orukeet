import Foundation
import Testing
@testable import OrukeetCoreML

private enum InstallFailure: Error { case compiler }

struct OrukeetInstallationTests {
    private func fixture() throws -> (root: URL, source: URL, destination: URL) {
        let files = FileManager.default
        let root = files.temporaryDirectory.appendingPathComponent("orukeet-test-\(UUID().uuidString)")
        let source = root.appendingPathComponent("source")
        try files.createDirectory(at: source, withIntermediateDirectories: true)
        for name in OrukeetLocalModels.componentNames {
            try files.createDirectory(at: source.appendingPathComponent("\(name).mlpackage"), withIntermediateDirectories: false)
        }
        let vocabulary = Dictionary(uniqueKeysWithValues: (0..<8192).map { (String($0), "token\($0)") })
        try JSONEncoder().encode(vocabulary).write(to: source.appendingPathComponent("parakeet_vocab.json"))
        return (root, source, root.appendingPathComponent("installed"))
    }

    @Test func successfulInstallPublishesCompleteBundleAndRemovesCompilerTemporaryFiles() throws {
        let (root, source, destination) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let sidecars = ["bundle.json", "LICENSE-WEIGHTS", "NOTICE.md", "COREML-NOTICE.txt"]
        for name in sidecars { try Data(name.utf8).write(to: source.appendingPathComponent(name)) }
        var compilerOutputs: [URL] = []
        try OrukeetLocalModels.compilePackages(from: source, to: destination) { package in
            #expect(!FileManager.default.fileExists(atPath: destination.path))
            let output = root.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
            try Data(package.lastPathComponent.utf8).write(to: output.appendingPathComponent("model"))
            compilerOutputs.append(output)
            return output
        }
        let installed = try FileManager.default.contentsOfDirectory(atPath: destination.path)
        #expect(Set(installed) == Set(OrukeetLocalModels.componentNames.map { "\($0).mlmodelc" } + ["parakeet_vocab.json"] + sidecars))
        for name in sidecars {
            #expect(try Data(contentsOf: destination.appendingPathComponent(name)) == Data(name.utf8))
        }
        #expect(compilerOutputs.allSatisfy { !FileManager.default.fileExists(atPath: $0.path) })
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).sorted() == ["installed", "source"])
        #expect(try OrukeetLocalModels.vocabulary(in: destination).count == 8192)
    }

    @Test func midCompilationFailureLeavesNoPartialInstallationAndCanRetry() throws {
        let (root, source, destination) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        var calls = 0
        #expect(throws: InstallFailure.self) {
            try OrukeetLocalModels.compilePackages(from: source, to: destination) { _ in
                calls += 1
                if calls == 3 { throw InstallFailure.compiler }
                let output = root.appendingPathComponent(UUID().uuidString)
                try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
                return output
            }
        }
        #expect(calls == 3)
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path) == ["source"])
        try OrukeetLocalModels.compilePackages(from: source, to: destination) { _ in
            let output = root.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
            return output
        }
        #expect(FileManager.default.fileExists(atPath: destination.appendingPathComponent("Encoder.mlmodelc").path))
    }

    @Test func existingInstallationIsPreservedWithoutInvokingCompiler() throws {
        let (root, source, destination) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: false)
        let marker = destination.appendingPathComponent("existing-install")
        try Data("unchanged".utf8).write(to: marker)
        #expect(throws: CocoaError(.fileWriteFileExists)) {
            try OrukeetLocalModels.compilePackages(from: source, to: destination) { _ in
                Issue.record("Compiler must not run when a destination already exists")
                throw InstallFailure.compiler
            }
        }
        #expect(try String(contentsOf: marker, encoding: .utf8) == "unchanged")
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).sorted() == ["installed", "source"])
    }

    @Test func allSourceComponentsAreCheckedBeforeCompilation() throws {
        let (root, source, destination) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let missing = source.appendingPathComponent("JointDecisionv3.mlpackage")
        try FileManager.default.removeItem(at: missing)
        #expect(throws: OrukeetModelError.missingComponent(missing.path)) {
            try OrukeetLocalModels.compilePackages(from: source, to: destination) { _ in
                Issue.record("Compiler must not run for an incomplete source")
                throw InstallFailure.compiler
            }
        }
        #expect(!FileManager.default.fileExists(atPath: destination.path))
    }

    @Test func malformedAndNonCanonicalVocabularyIsRejectedBeforeCompilation() throws {
        let (root, source, destination) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let path = source.appendingPathComponent("parakeet_vocab.json")
        let complete = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: path))
        var alternateKey = complete
        alternateKey["00"] = alternateKey.removeValue(forKey: "0")
        var outOfRange = complete
        outOfRange["8192"] = outOfRange.removeValue(forKey: "8191")
        let invalidData = try [Data("not JSON".utf8), JSONEncoder().encode(["0": "only one"]),
                               JSONEncoder().encode(alternateKey), JSONEncoder().encode(outOfRange)]
        for data in invalidData {
            try data.write(to: path)
            #expect(throws: OrukeetModelError.invalidVocabulary) {
                try OrukeetLocalModels.compilePackages(from: source, to: destination) { _ in
                    Issue.record("Compiler must not run with an invalid vocabulary")
                    throw InstallFailure.compiler
                }
            }
        }
    }

    @Test func cancellationAfterCompilationCleansStagingAndCompilerOutput() async throws {
        let (root, source, destination) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let task = Task {
            try OrukeetLocalModels.compilePackages(from: source, to: destination) { _ in
                let output = root.appendingPathComponent(UUID().uuidString)
                try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
                withUnsafeCurrentTask { $0?.cancel() }
                return output
            }
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path) == ["source"])
    }
}
