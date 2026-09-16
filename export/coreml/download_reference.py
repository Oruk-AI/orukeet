#!/usr/bin/env python3
"""Download the exact model artifacts used for the TapTalk comparison."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "FluidInference/parakeet-tdt-0.6b-v3-coreml"
REVISION = "7dd20fe6b1797d35f5e3307e8b1732d9a178edfe"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    components = ["Encoder", "Decoder", "Preprocessor", "JointDecisionv3"]
    patterns = [f"{name}.mlmodelc/**" for name in components]
    patterns += [f"mlpackages/{name}.mlpackage/**" for name in components[:3]]
    patterns += ["JointDecisionv3.mlpackage/**", "parakeet_vocab.json", "config.json", "mlpackages/metadata.json"]
    snapshot_download(REPO, revision=REVISION, allow_patterns=patterns, local_dir=args.output, max_workers=4)
    (args.output / "reference.json").write_text(json.dumps({"repo": REPO, "revision": REVISION}, indent=2) + "\n")


if __name__ == "__main__":
    main()
