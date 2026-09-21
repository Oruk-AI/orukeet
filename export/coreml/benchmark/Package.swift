// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "OrukeetCoreMLBenchmark",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "CoreMLBenchmark", targets: ["CoreMLBenchmark"])],
    dependencies: [
        .package(name: "Orukeet", path: "../../.."),
        .package(url: "https://github.com/Oruk-AI/FluidAudio.git", exact: "0.15.5-orukeet.1")
    ],
    targets: [
        .executableTarget(
            name: "CoreMLBenchmark",
            dependencies: [.product(name: "OrukeetCoreML", package: "Orukeet"),
                           .product(name: "FluidAudio", package: "FluidAudio")])
    ]
)
