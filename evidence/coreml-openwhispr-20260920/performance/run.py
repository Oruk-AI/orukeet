#!/usr/bin/env python3
"""Build two local release harnesses and benchmark existing Core ML caches only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNTIMES = {
    "original": "19600a485baa4998812e4654b70d2bab8f2c9949",
    "optimized": "75377a8a824abcd84946d80f3c2279ed25b13dbc",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(args, **kw):
    return subprocess.check_output(args, text=True, **kw).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", choices=["build", "main", "concurrency", "gpu", "all"], default="all")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    scratch = Path(settings["scratch"])
    output = HERE / "results"
    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    engine = HERE / "source/OrukeetEngine.swift"
    if not engine.exists():
        engine.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "export/coreml/benchmark/Sources/OrukeetCoreML/OrukeetEngine.swift", engine)
    for name, expected in RUNTIMES.items():
        runtime = Path(settings["runtimes"][name])
        actual = command(["git", "-C", str(runtime), "rev-parse", "HEAD"])
        if actual != expected:
            raise ValueError(f"Wrong {name} revision: {actual}")
        if command(["git", "-C", str(runtime), "status", "--porcelain", "--untracked-files=no"]):
            raise ValueError(f"Dirty runtime checkout: {name}")
    fixtures = settings["fixtures"]
    for fixture in fixtures:
        actual = sha(fixture["path"])
        if actual != fixture["sha256"]:
            raise ValueError(f"Audio hash mismatch: {fixture['id']}")
    if args.phase in ("build", "all"):
        for name in RUNTIMES:
            package = scratch / name
            (package / "Sources/OrukeetCoreML").mkdir(parents=True, exist_ok=True)
            (package / "Sources/EngineBenchmark").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(engine, package / "Sources/OrukeetCoreML/OrukeetEngine.swift")
            shutil.copyfile(HERE / "EngineBenchmark.swift", package / "Sources/EngineBenchmark/main.swift")
            dependency = json.dumps(settings["runtimes"][name])
            (package / "Package.swift").write_text(f'''// swift-tools-version: 6.0
import PackageDescription
let package = Package(name: "OrukeetPerformance", platforms: [.macOS(.v14)],
    dependencies: [.package(name: "FluidAudio", path: {dependency})],
    targets: [
        .target(name: "OrukeetCoreML", dependencies: [.product(name: "FluidAudio", package: "FluidAudio")]),
        .executableTarget(name: "EngineBenchmark", dependencies: ["OrukeetCoreML"])
    ])
''')
            log = scratch / f"build-{name}.log"
            with log.open("w") as stream:
                result = subprocess.run(["swift", "build", "--package-path", str(package), "-c", "release", "--jobs", "4"], stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                print(log.read_text()[-12000:])
                raise SystemExit(result.returncode)
            runtime = settings["runtimes"][name]
            if command(["git", "-C", runtime, "rev-parse", "HEAD"]) != RUNTIMES[name] or command(
                ["git", "-C", runtime, "status", "--porcelain", "--untracked-files=no"]
            ):
                raise ValueError("Runtime source changed during build; use isolated frozen worktrees")
            print(f"Built {name}: {log}", flush=True)
        provenance = {
            "source_repo_head_at_snapshot": settings["engine_snapshot_head"],
            "engine_source_sha256": sha(engine), "harness_sha256": sha(HERE / "EngineBenchmark.swift"),
            "runtime_commits": RUNTIMES, "swift": command(["swift", "--version"]),
            "os": command(["sw_vers"]), "hardware": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "build_configuration": "release", "fixtures": [{k: v for k, v in x.items() if k != "path"} for x in fixtures],
            "model_source": settings["model_source"],
            "binaries": {name: sha(scratch / name / ".build/release/EngineBenchmark") for name in RUNTIMES},
            "no_model_downloads_or_tensor_copies": True,
        }
        (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if args.phase == "build":
        return
    phases = [args.phase] if args.phase != "all" else ["main", "concurrency"]
    for phase in phases:
        if phase == "main":
            variants = [(runtime, model, 1, "ane") for model in ["published", "int8"] for runtime in RUNTIMES]
        elif phase == "concurrency":
            variants = [("optimized", "int8", c, "ane") for c in [1, 2, 4]]
        else:
            variants = [("optimized", "int8", 1, units) for units in ["ane", "gpu"]]
        for round_id in [0, 1]:
            for runtime, model, concurrency, units in (variants if round_id == 0 else list(reversed(variants))):
                label = f"{phase}-{runtime}-{model}-c{concurrency}-{units}"
                result_path = output / f"{label}-r{round_id}.json"
                if result_path.exists():
                    raise ValueError(f"Refuse to overwrite existing result: {result_path}")
                selected = fixtures if phase != "concurrency" else [x for x in fixtures if x["id"] == "natural-long"]
                config = {
                    "label": label, "round": round_id, "runtime": runtime, "modelProfile": model,
                    "modelDirectory": settings["models"][model], "encoderUnits": units,
                    "concurrency": concurrency, "warmups": 3, "repetitions": 15,
                    "fixtures": selected if round_id == 0 else list(reversed(selected)),
                    "outputPath": str(result_path),
                }
                config_path = scratch / f"{label}-r{round_id}.json"
                config_path.write_text(json.dumps(config, indent=2) + "\n")
                binary = scratch / runtime / ".build/release/EngineBenchmark"
                if sha(binary) != json.loads((HERE / "provenance.json").read_text())["binaries"][runtime]:
                    raise ValueError("Benchmark binary changed since build")
                log = scratch / f"{label}-r{round_id}.log"
                with log.open("w") as stream:
                    completed = subprocess.run([str(binary), str(config_path)], stdout=stream, stderr=subprocess.STDOUT)
                if completed.returncode:
                    print(log.read_text()[-12000:])
                    raise SystemExit(completed.returncode)
                report = json.loads(result_path.read_text())
                timed = [x for x in report["records"] if not x["warmup"]]
                import statistics
                summary = {x["id"]: round(statistics.median(r["wall_ms"] for r in timed if r["fixture"] == x["id"]), 3) for x in selected}
                print(f"{label} round {round_id}: wall medians ms {summary}; peak RSS {report['peak_rss_bytes_final']}", flush=True)


if __name__ == "__main__":
    main()
