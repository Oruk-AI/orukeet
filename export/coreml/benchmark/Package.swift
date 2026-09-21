// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "OrukeetCoreMLBenchmark",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "OrukeetCoreML", targets: ["OrukeetCoreML"]),
        .executable(name: "CoreMLBenchmark", targets: ["CoreMLBenchmark"]),
    ],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio.git", exact: "0.15.5")
    ],
    targets: [
        .target(name: "OrukeetCoreML", dependencies: [.product(name: "FluidAudio", package: "FluidAudio")]),
        .testTarget(name: "OrukeetCoreMLTests", dependencies: ["OrukeetCoreML"]),
        .executableTarget(
            name: "CoreMLBenchmark",
            dependencies: ["OrukeetCoreML", .product(name: "FluidAudio", package: "FluidAudio")]
        ),
    ]
)
