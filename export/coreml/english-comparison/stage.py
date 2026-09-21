#!/usr/bin/env python3
"""Explicit CI-only staging of pinned v2 inference assets and sealed English audio.

Importing this module, building Swift, and running unit tests never download assets.
No training checkpoints are needed. No remote shell or GPU access is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import struct
import tarfile
import tempfile
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
V2_REPO = "FluidInference/parakeet-tdt-0.6b-v2-coreml"
V2_REVISION = "ee09c569f73759e6d44c9bd16766f477b2b36d39"
FIXTURE_MANIFEST_SHA256 = "46d5db4c0a5b92557bf378bfeb9a7b150029714d449dc462d1a1b945475428d2"
FIXTURE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/en_us/audio/test.tar.gz"


def require_ci():
    if (os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
            or os.environ.get("RUNNER_OS") != "macOS"):
        raise RuntimeError("Asset staging/execution is restricted to an explicitly invoked GitHub-hosted macOS CI job")


def safe_relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
        raise ValueError(f"Unsafe relative path: {name!r}")
    return path


def read_json(path):
    return json.loads(path.read_text())


def verify_asset(path, entry):
    size = path.stat().st_size
    if size != entry["size_bytes"] or path.is_symlink():
        raise ValueError(f"Size/type mismatch: {entry['path']}")
    sha256 = hashlib.sha256()
    blob_sha1 = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(chunk)
            blob_sha1.update(chunk)
    actual = sha256.hexdigest()
    if entry.get("sha256"):
        if actual != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {entry['path']}")
    elif blob_sha1.hexdigest() != entry["git_blob_sha1"]:
        raise ValueError(f"Git blob SHA-1 mismatch: {entry['path']}")
    return {"path": entry["path"], "size_bytes": size, "sha256": actual}


def v2_manifest():
    manifest = read_json(HERE / "v2-assets.json")
    if (manifest["repository"] != V2_REPO or manifest["revision"] != V2_REVISION
            or len(manifest["files"]) != 21 or len({f["path"] for f in manifest["files"]}) != 21):
        raise ValueError("Unexpected v2 manifest identity")
    for entry in manifest["files"]:
        safe_relative(entry["path"])
    return manifest


def verify_v2(directory):
    manifest = v2_manifest()
    if directory.is_symlink() or any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("Symlinks are not permitted in the pinned v2 directory")
    actual = [verify_asset(directory / f["path"], f) for f in manifest["files"]]
    expected = {f["path"] for f in actual}
    found = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if found - expected - {"integrity.json"}:
        raise ValueError("Unexpected files in pinned v2 directory")
    return {"repository": V2_REPO, "revision": V2_REVISION, "status": "verified", "files": actual}


def download_file(url, destination, expected_size):
    request = urllib.request.Request(url, headers={"User-Agent": "Orukeet-English16-CI/1"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("xb") as stream:
        size = 0
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            size += len(chunk)
            if size > expected_size:
                raise ValueError("Download exceeds pinned size")
            stream.write(chunk)
    if size != expected_size:
        raise ValueError("Incomplete pinned download")


def stage_v2(destination):
    require_ci()
    if destination.exists():
        return verify_v2(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v2-stage-", dir=destination.parent) as temporary:
        directory = Path(temporary) / "models"
        directory.mkdir()
        for entry in v2_manifest()["files"]:
            path = directory / entry["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/{V2_REPO}/resolve/{V2_REVISION}/{urllib.parse.quote(entry['path'], safe='/')}"
            download_file(url, path, entry["size_bytes"])
            verify_asset(path, entry)
        receipt = verify_v2(directory)
        (directory / "integrity.json").write_text(json.dumps(receipt, indent=2) + "\n")
        directory.rename(destination)
    return receipt


def fixture_manifest():
    raw = (HERE / "fixtures.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXTURE_MANIFEST_SHA256:
        raise ValueError("Fixture manifest differs from the sealed English16 set")
    manifest = json.loads(raw)
    fixtures = manifest["fixtures"]
    if len(fixtures) != 16 or len({f["path"] for f in fixtures}) != 16:
        raise ValueError("Require exactly 16 unique sealed recordings")
    for fixture in fixtures:
        if (str(safe_relative(fixture["path"])) != Path(fixture["path"]).name
                or fixture["archive_url"] != FIXTURE_URL
                or not 0 < fixture["samples"] <= 240000):
            raise ValueError("Unexpected fixture identity")
        safe_relative(fixture["archive_member"])
    return manifest


def verify_wave(data, fixture):
    if hashlib.sha256(data).hexdigest() != fixture["sha256"]:
        raise ValueError(f"Audio SHA-256 mismatch: {fixture['path']}")
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Expected RIFF/WAVE")
    position, fmt, length = 12, None, None
    while position + 8 <= len(data):
        name, size = struct.unpack_from("<4sI", data, position)
        if position + 8 + size > len(data):
            raise ValueError("Truncated WAV chunk")
        if name == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", data, position + 8)
        if name == b"data":
            length = size
        position += 8 + size + size % 2
    if fmt is None or length is None:
        raise ValueError("Missing WAV format/data")
    tag, channels, rate, _, align, bits = fmt
    if (tag, channels, rate, bits, align) != (3, 1, 16000, 32, 4) or length != fixture["samples"] * 4:
        raise ValueError("Expected unchanged sealed Float32 mono 16 kHz samples")


def verify_fixtures(directory):
    fixtures = fixture_manifest()["fixtures"]
    for fixture in fixtures:
        path = directory / fixture["path"]
        if path.is_symlink():
            raise ValueError("Fixture symlink not permitted")
        verify_wave(path.read_bytes(), fixture)
    return {"status": "verified", "manifest_sha256": FIXTURE_MANIFEST_SHA256,
            "fixture_count": 16, "total_seconds": sum(f["samples"] for f in fixtures) / 16000,
            "files": [{"path": f["path"], "sha256": f["sha256"], "samples": f["samples"]} for f in fixtures]}


def extract_selected_audio(response, directory, fixtures):
    selected = {f["archive_member"]: f for f in fixtures}
    seen = set()
    with tarfile.open(fileobj=response, mode="r|gz") as archive:
        for member in archive:
            if member.name not in selected:
                continue
            if not member.isfile() or member.size > 4 * 1024 * 1024 or member.name in seen:
                raise ValueError("Invalid or duplicate selected archive member")
            fixture = selected[member.name]
            with archive.extractfile(member) as stream:
                data = stream.read()
            verify_wave(data, fixture)
            (directory / fixture["path"]).write_bytes(data)
            seen.add(member.name)
            if len(seen) == len(selected):
                break
    if seen != set(selected):
        raise ValueError("Pinned archive is missing sealed recordings")


def stage_fixtures(destination):
    require_ci()
    if destination.exists():
        return verify_fixtures(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".audio-stage-", dir=destination.parent) as temporary:
        directory = Path(temporary) / "audio"
        directory.mkdir()
        with urllib.request.urlopen(FIXTURE_URL, timeout=120) as response:
            extract_selected_audio(response, directory, fixture_manifest()["fixtures"])
        receipt = verify_fixtures(directory)
        (directory / "integrity.json").write_text(json.dumps(receipt, indent=2) + "\n")
        directory.rename(destination)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    require_ci()
    receipts = {"v2": stage_v2(args.destination / "v2"), "audio": stage_fixtures(args.destination / "audio")}
    (args.destination / "staging.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print("Verified 21 pinned v2 files and 16 unchanged public English recordings")


if __name__ == "__main__":
    main()
