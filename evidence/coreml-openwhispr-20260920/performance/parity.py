#!/usr/bin/env python3
"""Check the exact runtime backport on the existing sealed 400-clip INT8 corpus."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from run import HERE, RUNTIMES, command, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--summarize-only", action="store_true", help="Summarize completed private results without inference")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    fixtures = manifest["fixtures"]
    if len(fixtures) != 400 or len({x["sha256"] for x in fixtures}) != 400:
        raise ValueError("Unexpected corpus membership")
    counts = collections.Counter(x["language"] for x in fixtures)
    if len(counts) != 25 or set(counts.values()) != {16}:
        raise ValueError("Unexpected language counts")
    inputs = []
    for fixture in fixtures:
        path = args.manifest.parent / fixture["path"]
        if sha(path) != fixture["sha256"]:
            raise ValueError(f"Audio hash mismatch: {path.name}")
        inputs.append({"id": path.name, "path": str(path), "language": fixture["language"],
                       "sha256": fixture["sha256"], "samples": fixture["samples"]})
    output = HERE / "parity"
    output.mkdir(exist_ok=True)
    scratch = Path(settings["scratch"])
    private_output = scratch / "parity-results"
    private_output.mkdir(parents=True, exist_ok=True)
    binaries = {}
    for runtime, revision in RUNTIMES.items():
        if args.summarize_only:
            continue
        dependency = settings["runtimes"][runtime]
        if command(["git", "-C", dependency, "rev-parse", "HEAD"]) != revision:
            raise ValueError("Runtime revision mismatch")
        if command(["git", "-C", dependency, "status", "--porcelain", "--untracked-files=no"]):
            raise ValueError("Dirty runtime checkout")
        package = scratch / (runtime + "-parity")
        (package / "Sources/OrukeetCoreML").mkdir(parents=True, exist_ok=True)
        (package / "Sources/CorpusParity").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / "source/OrukeetEngine.swift", package / "Sources/OrukeetCoreML/OrukeetEngine.swift")
        shutil.copyfile(HERE / "CorpusParity.swift", package / "Sources/CorpusParity/main.swift")
        (package / "Package.swift").write_text(f'''// swift-tools-version: 6.0
import PackageDescription
let package = Package(name: "OrukeetCorpusParity", platforms: [.macOS(.v14)],
    dependencies: [.package(name: "FluidAudio", path: {json.dumps(dependency)})],
    targets: [
        .target(name: "OrukeetCoreML", dependencies: [.product(name: "FluidAudio", package: "FluidAudio")]),
        .executableTarget(name: "CorpusParity", dependencies: ["OrukeetCoreML"])
    ])
''')
        # Reuse this runtime's already-built release objects, with a separate
        # SwiftPM product. No source/runtime or model files are changed.
        build_dir = scratch / runtime / ".build"
        log = scratch / f"parity-{runtime}.log"
        with log.open("w") as stream:
            result = subprocess.run(["swift", "build", "--package-path", str(package), "--scratch-path", str(build_dir),
                                     "-c", "release", "--product", "CorpusParity", "--jobs", "4"],
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text()[-10000:])
            raise SystemExit(result.returncode)
        if command(["git", "-C", dependency, "rev-parse", "HEAD"]) != revision or command(
            ["git", "-C", dependency, "status", "--porcelain", "--untracked-files=no"]
        ):
            raise ValueError("Runtime source changed during build; use isolated frozen worktrees")
        binaries[runtime] = build_dir / "release/CorpusParity"
        print(f"Built corpus runner: {runtime}", flush=True)
    for runtime, binary in binaries.items():
        result_path = private_output / f"{runtime}.json"
        if result_path.exists():
            raise ValueError(f"Refuse to overwrite existing result: {result_path}")
        config = {"runtime": runtime, "modelDirectory": settings["models"]["int8"],
                  "fixtures": inputs, "outputPath": str(result_path)}
        config_path = scratch / f"parity-{runtime}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        subprocess.run([str(binary), str(config_path)], check=True)
        result = json.loads(result_path.read_text())
        result.update({"runtime_commit": RUNTIMES[runtime], "binary_sha256": sha(binary),
                       "engine_source_sha256": sha(HERE / "source/OrukeetEngine.swift"),
                       "manifest_sha256": sha(args.manifest)})
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    reports = {runtime: json.loads((private_output / f"{runtime}.json").read_text()) for runtime in RUNTIMES}
    for runtime, report in reports.items():
        if report["runtime_commit"] != RUNTIMES[runtime] or report["manifest_sha256"] != sha(args.manifest):
            raise ValueError("Raw result provenance mismatch")
        if report["engine_source_sha256"] != sha(HERE / "source/OrukeetEngine.swift"):
            raise ValueError("Engine source mismatch")
    indexed = {runtime: {x["audio_sha256"]: x for x in report["records"]} for runtime, report in reports.items()}
    expected = {x["sha256"] for x in inputs}
    for runtime, rows in indexed.items():
        if rows.keys() != expected or len(reports[runtime]["records"]) != 400:
            raise ValueError("Missing or duplicated inference result")
    differences = [{"id": indexed["original"][key]["id"], "language": indexed["original"][key]["language"],
                    "original": indexed["original"][key]["text"], "optimized": indexed["optimized"][key]["text"]}
                   for key in sorted(expected) if indexed["original"][key]["text"] != indexed["optimized"][key]["text"]]
    receipts = [{"id": indexed["original"][key]["id"], "language": indexed["original"][key]["language"],
                 "audio_sha256": key,
                 "text_sha256": {runtime: hashlib.sha256(indexed[runtime][key]["text"].encode("utf-8")).hexdigest()
                                 for runtime in RUNTIMES},
                 "identical": indexed["original"][key]["text"] == indexed["optimized"][key]["text"]}
                for key in sorted(expected)]
    receipt_path = output / "per-clip.jsonl"
    receipt_path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in receipts))
    summary = {"clips": 400, "languages": dict(sorted(counts.items())), "runtime_commits": RUNTIMES,
               "model_profile": "int8sym-encoder-only", "transcript_mismatches": len(differences),
               "differences": differences, "manifest_sha256": sha(args.manifest),
               "dataset": manifest["dataset"], "dataset_revision": manifest["revision"],
               "scope": "Fixed-model runtime parity only. Corpus previously used for precision research; not a fresh accuracy holdout.",
               "batch_concurrency": 1,
               "encoder_requested_compute_units": "cpuAndNeuralEngine",
               "engine_source_sha256": sha(HERE / "source/OrukeetEngine.swift"),
               "model_cache_integrity_sha256": sha(HERE / "model-cache-integrity.json"),
               "result_sha256": {runtime: sha(private_output / f"{runtime}.json") for runtime in RUNTIMES},
               "binary_sha256": {runtime: reports[runtime]["binary_sha256"] for runtime in RUNTIMES},
               "per_clip_receipt_sha256": sha(receipt_path),
               "raw_result_storage": "Private scratch parity-results/. Exact transcript hashes are published per clip."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if differences:
        raise SystemExit("Runtime transcript parity failed")


if __name__ == "__main__":
    main()
