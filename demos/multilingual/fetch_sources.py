"""Fetch three preselected public FLEURS recordings, without Hub credentials.

Original WAV members are copied byte-for-byte from commit-pinned source
archives. The archive is streamed; unrelated recordings are not saved.
"""
from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
PROV = ROOT / "provenance"
CACHE = ROOT.parents[1] / ".tmp" / "multilingual-sources"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def get(url):
    # Deliberately no Hugging Face token or local dataset cache.
    return urllib.request.urlopen(url, timeout=60)


def fetch(config, selection):
    revision = selection["revision"]
    prefix = f"https://huggingface.co/datasets/google/fleurs/resolve/{revision}"
    tsv_url = f"{prefix}/data/{config}/test.tsv"
    with get(tsv_url) as response:
        tsv = response.read()
    saved = CACHE / f"{config}-test.tsv"
    if saved.exists() and saved.read_bytes() != tsv:
        raise ValueError(f"Pinned TSV changed: {config}")
    saved.write_bytes(tsv)
    fields = tsv.decode("utf-8").splitlines()[0].split("\t")
    selected_row = tsv.splitlines(keepends=True)[0]
    selected_path = PROV / f"{config}-test-row-0.tsv"
    selected_path.write_bytes(selected_row)
    if len(fields) != 7 or not fields[0].isdigit():
        raise ValueError(f"Unexpected source TSV schema: {config}")
    filename = fields[1]
    archive_url = f"{prefix}/data/{config}/audio/test.tar.gz"
    metadata_url = f"https://huggingface.co/api/datasets/google/fleurs/tree/{revision}/data/{config}/audio?expand=true"
    with get(metadata_url) as response:
        archive_metadata = next(item for item in json.load(response) if item["path"].endswith("/test.tar.gz"))
    audio = None
    archive_member = None
    members_scanned = 0
    print(f"Fetching fixed selection {config}: {filename}", flush=True)
    with get(archive_url) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                members_scanned += 1
                if member.isfile() and Path(member.name).name == filename:
                    if member.size > 20_000_000:
                        raise ValueError("Unexpectedly large selected WAV")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError("Selected member could not be opened")
                    audio = stream.read()
                    archive_member = member.name
                    break
    if audio is None:
        raise ValueError(f"Selected file missing from archive: {config}")
    path = ROOT / "audio" / f"{config}.source.wav"
    prior = PROV / f"{config}.json"
    if prior.exists():
        expected = json.loads(prior.read_text())["audio"]["sha256"]
        if digest(audio) != expected:
            raise ValueError(f"Selected WAV changed: {config}")
    path.write_bytes(audio)
    record = {
        "dataset": "google/fleurs", "provider": "Google / FLEURS authors",
        "revision": revision, "config": config, "split": "test", "original_tsv_row_zero_based": 0,
        "id": int(fields[0]), "source_filename": filename,
        "reference_raw": fields[2], "reference_normalized_by_provider": fields[3],
        "source_num_samples": int(fields[5]),
        "tsv": {"url": tsv_url, "source_file_sha256": digest(tsv),
                "selected_row_file": f"provenance/{selected_path.name}", "selected_row_sha256": digest(selected_row),
                "note": "Only the selected first row is distributed; the full TSV download is a private .tmp cache."},
        "archive": {"url": archive_url, "member": archive_member,
                    "provider_metadata": archive_metadata,
                    "verification_note": "Only the selected member was copied and hashed. The streamed archive was stopped after that member; its full advertised LFS hash was not locally verified.",
                    "members_scanned": members_scanned},
        "audio": {"local_file": f"audio/{path.name}", "sha256": digest(audio),
                  "bytes": len(audio), "changes": "None: original archive member copied byte-for-byte."},
        "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech (2022), Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera and Ankur Bapna; distributed by Google.",
        "card_url": f"https://huggingface.co/datasets/google/fleurs/blob/{revision}/README.md",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
    }
    prior.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {config}: {len(audio)} bytes, {digest(audio)}", flush=True)
    return record


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    selection = json.loads((PROV / "selection.json").read_text())
    revision = selection["revision"]
    with get(f"https://huggingface.co/datasets/google/fleurs/raw/{revision}/README.md") as response:
        card = response.read()
    if b"cc-by-4.0" not in card:
        raise ValueError("Expected explicit CC BY 4.0 card metadata")
    (CACHE / "google-fleurs-card.md").write_bytes(card)
    (PROV / "source-card.json").write_text(json.dumps({
        "url": f"https://huggingface.co/datasets/google/fleurs/raw/{revision}/README.md",
        "sha256": digest(card), "revision": revision, "license": "CC-BY-4.0",
        "verified_license_metadata": ["cc-by-4.0"],
        "note": "Full provider card is retained only in the private .tmp download cache; this record retains its revision, URL, hash and license metadata.",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, config, selection) for config in selection["configs"]]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
