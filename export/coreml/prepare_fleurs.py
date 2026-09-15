#!/usr/bin/env python3
"""Seal a small, deterministic multilingual regression corpus before inference.

Select the first N eligible files in each pinned FLEURS test archive, in archive
order, at 0 < duration <= 15 seconds. Stream only the prefix needed for selection.
This is a regression sample, not a replacement for the full published WER suite.
"""
import argparse
import csv
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
BASE = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/data"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-language", type=int, default=8)
    parser.add_argument("--skip-per-language", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--languages", nargs="+", default=["en_us", "fr_fr", "es_419", "lv_lv", "de_de", "it_it", "uk_ua", "ru_ru"])
    args = parser.parse_args()
    if args.per_language < 1 or args.skip_per_language < 0 or (args.output.exists() and not args.resume):
        raise ValueError("Choose a new output directory and a positive sample count")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    records = []
    for language in args.languages:
        metadata = urllib.request.urlopen(f"{BASE}/{language}/test.tsv", timeout=60).read()
        rows = {r[1]: r for r in csv.reader(io.StringIO(metadata.decode()), delimiter="\t")}
        archive_url = f"{BASE}/{language}/audio/test.tar.gz"
        count = 0
        skipped = 0
        with urllib.request.urlopen(archive_url, timeout=60) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                row = rows.get(Path(member.name).name)
                if not member.isfile() or row is None or not 0 < int(row[5]) <= 240000:
                    continue
                if skipped < args.skip_per_language:
                    skipped += 1
                    continue
                data = archive.extractfile(member).read()
                path = args.output / f"{language}-{Path(member.name).name}"
                if path.exists() and path.read_bytes() != data:
                    raise ValueError(f"Existing fixture differs from pinned source: {path}")
                path.write_bytes(data)
                records.append({"path": path.name, "language": language.split("_")[0], "config": language,
                                "sentence_id": row[0], "reference": row[2], "normalized_reference": row[3],
                                "samples": int(row[5]), "sha256": hashlib.sha256(data).hexdigest(),
                                "archive_url": archive_url, "archive_member": member.name})
                count += 1
                if count == args.per_language:
                    break
        if count != args.per_language:
            raise ValueError(f"Not enough eligible samples in {language}")
        print(language, count, flush=True)
    manifest = {"dataset": "google/fleurs", "revision": REVISION, "split": "test", "license": "CC-BY-4.0",
                "selection": "Eligible files in archive order, 0 < samples <= 240000; sealed before inference",
                "skip_per_language": args.skip_per_language, "per_language": args.per_language, "fixtures": records}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
