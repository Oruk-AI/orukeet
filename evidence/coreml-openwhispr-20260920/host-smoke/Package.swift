// swift-tools-version: 6.0
import PackageDescription
let package = Package(
    name: "OrukeetEngineSmoke",
    platforms: [.macOS(.v14)],
    dependencies: [.package(path: "../../../export/coreml/benchmark")],
    targets: [.executableTarget(name: "EngineSmoke", dependencies: [.product(name: "OrukeetCoreML", package: "benchmark")])]
)
