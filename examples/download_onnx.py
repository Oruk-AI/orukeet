"""Download the pinned ONNX release and verify it against its Hub manifest."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "oruk/orukeet"
REVISION = "55a984d46f68323301837194ce647c702f55facc"
MANIFEST_PATH = "onnx/manifest.json"
MANIFEST_SHA256 = "7e80f93f0e9b923c392424b0f85d28a717feee0a4d2a6aa9bfa723693868e727"


def verify(path, expected):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch: {path.name}")


def download(cache, *, local_files_only=False):
    options = dict(revision=REVISION, local_dir=Path(cache) / "models",
                   local_files_only=local_files_only)
    manifest_path = Path(hf_hub_download(REPO_ID, MANIFEST_PATH, **options))
    verify(manifest_path, MANIFEST_SHA256)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = Path(hf_hub_download(REPO_ID, "onnx/" + manifest["archive"], **options))
    if archive.stat().st_size != manifest["archive_bytes"]:
        raise ValueError("ONNX archive size mismatch")
    verify(archive, manifest["archive_sha256"])
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/orukeet")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Use cached files without contacting Hugging Face")
    args = parser.parse_args()
    print(download(args.cache, local_files_only=args.local_files_only).resolve())


if __name__ == "__main__":
    main()
