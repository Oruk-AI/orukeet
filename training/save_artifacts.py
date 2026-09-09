#!/usr/bin/env python3
"""Persist experiment artifacts in the already-private source model repository."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/home/nathanroll/parakeet-ft"))
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    api = HfApi()
    repo = "oruk/parakeet-tdt-0.6b-v3-cv-fleurs"
    info = api.model_info(repo)
    if not info.private:
        raise RuntimeError("Expected the existing private artifact repository")
    decision = json.loads((root / "eval/resume_20260905/selection_decision.json").read_text())
    prefix = "continuations/2026-09-05"
    model = root / "models/ft/stage3_anchored_accents_20260905.nemo"
    model_commit = api.upload_file(path_or_fileobj=str(model),
        path_in_repo=f"{prefix}/{model.name}", repo_id=repo,
        commit_message="Save evaluated September 5 encoder adaptation candidate")
    bundle_commit = api.upload_file(path_or_fileobj=str(args.bundle),
        path_in_repo=f"{prefix}/{args.bundle.name}", repo_id=repo,
        commit_message="Save cleaned manifests and continuation evidence")
    report_commit = api.upload_file(path_or_fileobj=str(args.report),
        path_in_repo=f"{prefix}/REPORT.md", repo_id=repo,
        commit_message="Record measured selection and confirmation outcome")
    files = {f.rfilename: f for f in api.model_info(repo, revision=report_commit.oid, files_metadata=True).siblings}
    def remote_sha(name):
        value = files[name].lfs
        return value.get("sha256") if isinstance(value, dict) else getattr(value, "sha256", None)
    if remote_sha(f"{prefix}/{model.name}") != decision["candidate_model_sha256"]:
        raise RuntimeError("Uploaded candidate hash mismatch")
    digest = hashlib.sha256()
    with args.bundle.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    if remote_sha(f"{prefix}/{args.bundle.name}") != digest.hexdigest():
        raise RuntimeError("Uploaded artifact bundle hash mismatch")
    receipt = {"repository": repo, "private": True, "prefix": prefix,
               "candidate_revision": model_commit.oid, "bundle_revision": bundle_commit.oid,
               "report_revision": report_commit.oid,
               "bundle_sha256": digest.hexdigest(), "remote_hashes_verified": True,
               "candidate_accepted": decision["candidate_accepted"],
               "candidate_model_sha256": decision["candidate_model_sha256"],
               "selected_model_sha256": decision["selected_model_sha256"],
               "existing_top_level_model_replaced": False}
    receipt_path = root / "eval/resume_20260905/upload_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt_commit = api.upload_file(path_or_fileobj=str(receipt_path),
        path_in_repo=f"{prefix}/upload_receipt.json", repo_id=repo,
        commit_message="Record verified artifact hashes and revisions")
    receipt["receipt_revision"] = receipt_commit.oid
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
