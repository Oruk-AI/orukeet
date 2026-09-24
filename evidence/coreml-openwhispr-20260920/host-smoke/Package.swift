// swift-tools-version: 6.0
import PackageDescription
let package = Package(
    name: "OrukeetEngineSmoke",
    platforms: [.macOS(.v14)],
    dependencies: [.package(name: "Orukeet", path: "../../..")],
    targets: [.executableTarget(name: "EngineSmoke", dependencies: [.product(name: "OrukeetCoreML", package: "Orukeet")])]
)
