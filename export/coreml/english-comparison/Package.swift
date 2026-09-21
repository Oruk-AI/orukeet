// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "OrukeetEnglishComparison",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "EnglishComparison", targets: ["EnglishComparison"])],
    dependencies: [
        .package(path: "../benchmark"),
        .package(url: "https://github.com/FluidInference/FluidAudio.git", exact: "0.15.5"),
    ],
    targets: [
        .executableTarget(
            name: "EnglishComparison",
            dependencies: [
                .product(name: "OrukeetCoreML", package: "benchmark"),
                .product(name: "FluidAudio", package: "FluidAudio"),
            ])
    ]
)
