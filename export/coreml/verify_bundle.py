#!/usr/bin/env python3
"""Verify a pinned portable preview ZIP without extracting model weights."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

SOURCE_SHA256 = "031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56"
REQUIRED = {"LICENSE-WEIGHTS", "NOTICE.md", "COREML-NOTICE.txt", "parakeet_vocab.json"}
for component in ("Preprocessor", "Encoder", "Decoder", "JointDecisionv3"):
    REQUIRED.update({f"{component}.mlpackage/Manifest.json",
                     f"{component}.mlpackage/Data/com.apple.CoreML/model.mlmodel",
                     f"{component}.mlpackage/Data/com.apple.CoreML/weights/weight.bin"})


def digest(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(chunk)
    return result.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or ".." in path.parts
            or str(path) != name or any(part in ("", ".") for part in path.parts)):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return path


def verify_contents(archive, root, profile):
    """Check every portable payload; caller authenticates the ZIP first."""
    safe_name(root.rstrip("/"))
    if not root.endswith("/"):
        raise ValueError("Archive root must end in /")
    members = {}
    for item in archive.infolist():
        name = item.filename.rstrip("/") if item.is_dir() else item.filename
        safe_name(name)
        if not item.filename.startswith(root):
            raise ValueError(f"Unexpected archive root: {item.filename}")
        if stat.S_ISLNK(item.external_attr >> 16) or item.flag_bits & 1:
            raise ValueError(f"Link or encrypted entry: {item.filename}")
        if item.is_dir():
            continue
        relative = item.filename[len(root):]
        if relative in members:
            raise ValueError(f"Duplicate archive entry: {relative}")
        members[relative] = item
    manifest = json.loads(archive.read(root + "bundle.json"))
    expected = {"model": "Orukeet", "source": "r3", "source_sha256": SOURCE_SHA256,
                "profile": profile, "sample_rate": 16000, "fluid_audio_version": "0.15.5",
                "compiled_cache_included": False}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Unexpected bundle identity: {key}")
    files = manifest["files"]
    if not REQUIRED <= files.keys():
        raise ValueError("Missing required portable components or attribution")
    if set(members) != set(files) | {"bundle.json"}:
        raise ValueError("Missing or unlisted archive payload")
    for name, metadata in files.items():
        safe_name(name)
        if ".mlmodelc" in name:
            raise ValueError("Portable archive must not include a compiled device cache")
        if members[name].file_size != metadata["bytes"]:
            raise ValueError(f"Size mismatch: {name}")
        with archive.open(members[name]) as stream:
            if digest(stream) != metadata["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {name}")
    vocabulary = json.loads(archive.read(root + "parakeet_vocab.json"))
    if (set(vocabulary) != {str(i) for i in range(8192)}
            or not all(isinstance(token, str) for token in vocabulary.values())):
        raise ValueError("Expected complete 8192-token v3 vocabulary")
    return len(files)


def verify(path, profile):
    catalog = json.loads((Path(__file__).parent / "huggingface/manifest.json").read_text())
    entry = catalog["archives"][profile]
    if path.stat().st_size != entry["bytes"]:
        raise ValueError("Archive size does not match the pinned release")
    with path.open("rb") as stream:
        actual_hash = digest(stream)
    if actual_hash != entry["sha256"]:
        raise ValueError("Archive SHA-256 does not match the pinned release")
    with zipfile.ZipFile(path) as archive:
        count = verify_contents(archive, entry["archive_root"], entry.get("manifest_profile", profile))
    return {"status": "verified", "profile": profile, "archive_sha256": actual_hash,
            "archive_bytes": path.stat().st_size, "verified_payload_files": count,
            "source_sha256": SOURCE_SHA256, "fluid_audio_version": "0.15.5",
            "hf_revision": entry["hf_revision"],
            "qualification": "Artifact integrity only; not an iOS device qualification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--profile", choices=("greedy", "baseline", "int8"), default="greedy")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(verify(args.archive, args.profile), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
