// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Orukeet",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [.library(name: "OrukeetCoreML", targets: ["OrukeetCoreML"])],
    dependencies: [
        .package(url: "https://github.com/Oruk-AI/FluidAudio.git", exact: "0.15.5-orukeet.1")
    ],
    targets: [
        .target(
            name: "OrukeetCoreML",
            dependencies: [.product(name: "FluidAudio", package: "FluidAudio")],
            path: "export/coreml/benchmark/Sources/OrukeetCoreML"),
        .testTarget(
            name: "OrukeetCoreMLTests", dependencies: ["OrukeetCoreML"],
            path: "export/coreml/benchmark/Tests/OrukeetCoreMLTests"),
    ]
)
