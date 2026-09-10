#!/usr/bin/env python3
"""Build a deterministic, single-root tar.bz2 and record every payload hash."""
import argparse
import json
from pathlib import Path
import tarfile

from export_orukeet import SOURCE_SHA256, sha256, write_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model-dir", type=Path, required=True)
p.add_argument("--export-receipt", type=Path, required=True)
p.add_argument("--archive", type=Path, required=True)
p.add_argument("--manifest", type=Path, required=True)
a = p.parse_args()
receipt = json.loads(a.export_receipt.read_text())
assert receipt["status"] == "pass" and receipt["source_sha256"] == SOURCE_SHA256
expected = {g["file"]: g for g in receipt["graphs"] if ".int8.onnx" in g["file"]}
for name, info in expected.items():
    assert sha256(a.model_dir / name) == info["sha256"], name
    assert not info["external_files"], f"{name} must be a self-contained ONNX file"
assert sha256(a.model_dir / "tokens.txt") == receipt["tokens"]["sha256"]
names = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt", "bpe.vocab", "LICENSE-WEIGHTS", "NOTICE.md"]
files = [{"path": name, "bytes": (a.model_dir / name).stat().st_size, "sha256": sha256(a.model_dir / name)} for name in names]

def stable_metadata(info):
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o755 if info.isdir() else 0o644
    return info

a.archive.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(a.archive, "w:bz2", compresslevel=9, format=tarfile.USTAR_FORMAT) as tar:
    root = tarfile.TarInfo(a.model_dir.name)
    root.type = tarfile.DIRTYPE
    tar.addfile(stable_metadata(root))
    for name in names:
        tar.add(a.model_dir / name, arcname=f"{a.model_dir.name}/{name}", filter=stable_metadata)
manifest = {"source_sha256": SOURCE_SHA256, "archive": a.archive.name,
            "archive_bytes": a.archive.stat().st_size, "archive_sha256": sha256(a.archive),
            "extract_dir": a.model_dir.name, "files": files,
            "export_receipt_sha256": sha256(a.export_receipt),
            "packager_sha256": sha256(Path(__file__))}
write_json(a.manifest, manifest)
print(json.dumps(manifest, indent=2), flush=True)
