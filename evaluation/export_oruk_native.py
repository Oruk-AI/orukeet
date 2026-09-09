"""Convert the selected private checkpoint without losing its mel filterbank.

Usage: python scripts/export_oruk_native.py CHECKPOINT NEMO_SPEECH_CPP OUTPUT_DIR
Requires torch, numpy, librosa, gguf, PyYAML, sentencepiece, huggingface_hub.
The NVIDIA source checkout must be at the pinned v0.1.0 commit.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SOURCE_SHA = "313d615ca34c8ac3a183384e1f87e8274d748e5445af110013a335f02ae42e32"
CONVERTER_COMMIT = "4f9676226f667d14608487df744f375db87127f8"


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("converter", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import librosa  # Mandatory: upstream otherwise silently emits a lossy front end.
    from gguf import GGUFReader
    assert sha256(args.checkpoint) == SOURCE_SHA, "Wrong fine-tune checkpoint"
    revision = subprocess.check_output(["git", "-C", str(args.converter), "rev-parse", "HEAD"], text=True).strip()
    assert revision == CONVERTER_COMMIT, "Converter revision changed"
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = dict(source_sha256=SOURCE_SHA, converter_commit=revision,
                   librosa_version=librosa.__version__, artifacts=[])
    for precision in ("q8_0", "fp16"):
        output = args.output / f"oruk-parakeet-v3-20260905.{precision}.gguf"
        subprocess.run([sys.executable, str(args.converter / "convert_model.py"),
                        str(args.checkpoint), "--outfile", str(output), "--outtype", precision], check=True)
        reader = GGUFReader(str(output))
        filters = next(t for t in reader.tensors if t.name == "preprocessor.fb")
        assert filters.data.size == 128 * 257, "Missing or incompatible mel filterbank"
        receipt["artifacts"].append(dict(name=output.name, sha256=sha256(output),
                                         size_bytes=output.stat().st_size, precision=precision,
                                         mel_filterbank_sha256=hashlib.sha256(filters.data.tobytes()).hexdigest()))
    (args.output / "lineage.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
