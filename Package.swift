// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Orukeet",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [.library(name: "OrukeetCoreML", targets: ["OrukeetCoreML"])],
    dependencies: [
        .package(url: "https://github.com/Oruk-AI/FluidAudio.git", exact: "0.15.5-orukeet.1"),
        .package(url: "https://github.com/weichsel/ZIPFoundation.git", exact: "0.9.20")
    ],
    targets: [
        .target(
            name: "OrukeetCoreML",
            dependencies: [.product(name: "FluidAudio", package: "FluidAudio"),
                           .product(name: "ZIPFoundation", package: "ZIPFoundation")],
            path: "export/coreml/benchmark/Sources/OrukeetCoreML"),
        .testTarget(
            name: "OrukeetCoreMLTests",
            dependencies: ["OrukeetCoreML", .product(name: "ZIPFoundation", package: "ZIPFoundation")],
            path: "export/coreml/benchmark/Tests/OrukeetCoreMLTests"),
    ]
)
