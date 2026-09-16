#!/usr/bin/env python3
"""Create a self-contained Core ML bundle with attribution and file checksums."""
import argparse
import json
import shutil
from pathlib import Path

from convert import SOURCE_SHA256, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--include-compiled", action="store_true", help="Include this Mac's compiled cache; redistribute mlpackages for other OS versions")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists; choose a new directory")
    args.output.mkdir(parents=True)
    for name in ["Preprocessor", "Encoder", "Decoder", "JointDecisionv3"]:
        shutil.copytree(args.models / f"{name}.mlpackage", args.output / f"{name}.mlpackage", symlinks=False)
        if args.include_compiled:
            shutil.copytree(args.models / f"{name}.mlmodelc", args.output / f"{name}.mlmodelc", symlinks=False)
    for name in ["parakeet_vocab.json", "conversion.json", "optimization.json", "precision.json"]:
        if (args.models / name).exists():
            shutil.copy2(args.models / name, args.output / name)
    root = Path(__file__).resolve().parents[2]
    for name in ["LICENSE-WEIGHTS", "NOTICE.md"]:
        shutil.copy2(root / name, args.output / name)
    (args.output / "COREML-NOTICE.txt").write_text(
        "Orukeet r3 Core ML conversion and optimizations by Oruk AI.\n"
        "Based on NVIDIA Parakeet TDT 0.6B v3 and Fluid Inference's Core ML graphs.\n"
        "Orukeet modified weights: CC BY-SA 4.0; original NVIDIA attribution retained.\n"
        "FluidAudio runtime: Apache-2.0. Conversion/integration code: MIT.\n"
        "Baseline graph metadata intentionally retains upstream values for byte identity.\n"
        "Use bundle.json and source_sha256 to identify these as Orukeet weights.\n"
    )
    files = {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(args.output.rglob("*")) if p.is_file()}
    manifest = {"model": "Orukeet", "source": "r3", "source_sha256": SOURCE_SHA256,
                "profile": args.profile, "sample_rate": 16000, "fluid_audio_version": "0.15.5",
                "compiled_cache_included": args.include_compiled, "files": files}
    (args.output / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
