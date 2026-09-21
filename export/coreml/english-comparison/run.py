#!/usr/bin/env python3
"""Authenticate staged inputs, run the paired Mac diagnostic, and retain evidence."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile

import stage

HERE = Path(__file__).resolve().parent
FLUID_REVISION = "19600a485baa4998812e4654b70d2bab8f2c9949"


def verify_orukeet(archive_path, packages):
    spec = importlib.util.spec_from_file_location("verify_bundle", HERE.parent / "verify_bundle.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    receipt = verifier.verify(archive_path, "greedy")
    catalog = stage.read_json(HERE.parent / "huggingface/manifest.json")
    prefix = catalog["archives"]["greedy"]["archive_root"]
    files = []
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            relative = item.filename[len(prefix):]
            path = packages / relative
            if path.is_symlink() or not path.is_file() or path.stat().st_size != item.file_size:
                raise ValueError(f"Extracted Orukeet source is missing/different: {relative}")
            with archive.open(item) as source:
                expected = verifier.digest(source)
            with path.open("rb") as actual:
                if verifier.digest(actual) != expected:
                    raise ValueError(f"Extracted Orukeet hash mismatch: {relative}")
            files.append({"path": relative, "size_bytes": item.file_size, "sha256": expected})
    if any(p.is_symlink() for p in packages.rglob("*")):
        raise ValueError("Symlinks are not permitted in portable Orukeet inputs")
    present = {str(p.relative_to(packages)) for p in packages.rglob("*") if p.is_file()}
    if present != {f["path"] for f in files}:
        raise ValueError("Extra or missing Orukeet package files")
    return {**receipt, "source_files": files}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def check_runtime_pin():
    resolved = stage.read_json(HERE / "Package.resolved")
    pins = [pin for pin in resolved["pins"] if pin["identity"] == "fluidaudio"]
    if len(pins) != 1 or pins[0]["state"] != {"revision": FLUID_REVISION, "version": "0.15.5"}:
        raise ValueError("Resolved FluidAudio differs from the reviewed 0.15.5 revision")
    checkout = HERE / ".build/checkouts/FluidAudio"
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != FLUID_REVISION:
        raise ValueError("Actual Swift runtime checkout differs from the reviewed revision")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-models", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--orukeet-packages", type=Path, required=True)
    parser.add_argument("--orukeet-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--encoder-units", choices=("ane", "cpu"), default="ane")
    args = parser.parse_args()
    stage.require_ci()
    args.output.mkdir(parents=True, exist_ok=False)
    compiled_cache = tempfile.TemporaryDirectory(prefix=".english-compiled-", dir=args.orukeet_packages.parent)
    try:
        integrity = {"v2": stage.verify_v2(args.v2_models),
                     "audio": stage.verify_fixtures(args.audio_root),
                     "orukeet": verify_orukeet(args.orukeet_archive, args.orukeet_packages)}
        write_json(args.output / "integrity.json", integrity)
        with (args.output / "build.log").open("w") as log:
            subprocess.run(["swift", "build", "--package-path", str(HERE), "-c", "release", "--jobs", "4"],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        check_runtime_pin()
        bin_path = subprocess.check_output(
            ["swift", "build", "--package-path", str(HERE), "-c", "release", "--show-bin-path"], text=True).strip()
        command = [str(Path(bin_path) / "EnglishComparison"),
                   "--v2", str(args.v2_models.resolve()),
                   "--orukeet-packages", str(args.orukeet_packages.resolve()),
                   "--orukeet-compiled", str((Path(compiled_cache.name) / "orukeet").resolve()),
                   "--audio", str(args.audio_root.resolve()), "--fixtures", str(HERE / "fixtures.json"),
                   "--output", str((args.output / "raw.json").resolve()), "--encoder-units", args.encoder_units]
        with (args.output / "inference.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        raw_path = args.output / "raw.json"
        if not raw_path.exists():
            raise RuntimeError(f"Inference failed before a raw report (exit {result.returncode}); see inference.log")
        raw = stage.read_json(raw_path)
        raw["integrity"] = integrity
        raw["runner"] = {"machine": platform.machine(), "system": platform.platform(),
                         "sourceRevision": subprocess.check_output(
                             ["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip(),
                         "swiftVersion": subprocess.check_output(["swift", "--version"], text=True).strip(),
                         "inferenceExitCode": result.returncode}
        write_json(raw_path, raw)
        if result.returncode:
            raise RuntimeError(f"Incomplete inference (exit {result.returncode}); exact model/clip errors retained in raw.json")
        subprocess.run([sys.executable, str(HERE / "score.py"), "--raw", str(raw_path),
                        "--fixtures", str(HERE / "fixtures.json"), "--output", str(args.output / "scored.json")], check=True)
        print(f"Complete English16 paired diagnostic: {args.output / 'scored.json'}")
    except Exception as error:
        write_json(args.output / "failure.json", {"status": "failed", "error": str(error),
                   "qualification": "Incomplete diagnostics must not be scored as passing or promoted to release evidence."})
        raise
    finally:
        # Keep model payloads outside evidence/artifact folders, including failed runs.
        compiled_cache.cleanup()


if __name__ == "__main__":
    main()
