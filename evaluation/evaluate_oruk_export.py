"""Replay fixed evaluation manifests through the deployed native audio path.

This is an export regression check, not a new held-out model-selection study.
"""
import argparse
import array
from collections import defaultdict
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
CHAR_MAP = str.maketrans({"ς": "σ", "ё": "е", "Ё": "Е", "ѝ": "и", "ţ": "ț", "ş": "ș", "Ţ": "Ț", "Ş": "Ș"})
PUNCT = re.compile(r"[\"'“”„‘’«»()\[\]{}.,!?;:¡¿…\-–—/\\|*+=<>@#$%^&_~`]")


def normalize(text):
    return " ".join(PUNCT.sub(" ", unicodedata.normalize("NFC", text).translate(CHAR_MAP).lower()).split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-set", type=int, default=0)
    args = parser.parse_args()
    from rapidfuzz.distance import Levenshtein
    from orukeet.nvidia import NvidiaRecognizer
    from orukeet.audio import windows
    args.output.mkdir(parents=True, exist_ok=True)
    with args.model.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = dict(model_sha256=checksum, normalizer="legacy-compatible-nfc-v1", device=args.device,
                    runtime="NeMo-Speech.cpp v0.1.0", limit_per_set=args.limit_per_set, sets={})
    (args.output / "protocol.json").write_text(json.dumps(metadata, indent=2))
    engine = NvidiaRecognizer(args.runtime, str(args.model), args.device)
    try:
        for manifest in sorted(args.manifests.glob("*.jsonl")):
            rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
            if not rows:
                continue
            if args.limit_per_set:
                rows.sort(key=lambda row: hashlib.sha256(row["audio_filepath"].encode()).digest())
                rows = rows[:args.limit_per_set]
            output = args.output / (manifest.stem + "_hypotheses.jsonl")
            if output.exists():
                completed = [json.loads(line) for line in output.read_text().splitlines()]
                if len(completed) == len(rows):
                    continue
                raise RuntimeError(f"Incomplete previous output: {output}; use a new output directory")
            totals = defaultdict(int)
            started = time.perf_counter()
            with output.open("w") as stream:
                for row in rows:
                    texts = []
                    for offset, samples in windows(row["audio_filepath"]):
                        texts.append(engine.transcribe(array.array("f", samples), "auto")["text"])
                    hypothesis = " ".join(texts).strip()
                    ref, pred = normalize(row["text"]), normalize(hypothesis)
                    record = dict(row, pred_text=hypothesis, errors=Levenshtein.distance(ref.split(), pred.split()),
                                  words=len(ref.split()), char_errors=Levenshtein.distance(ref, pred), chars=len(ref))
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    for key in ("errors", "words", "char_errors", "chars"):
                        totals[key] += record[key]
                    totals["rows"] += 1
            totals["wer"] = 100 * totals["errors"] / totals["words"]
            metadata["sets"][manifest.stem] = dict(totals, seconds=time.perf_counter() - started,
                                                   manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
            (args.output / "results.json").write_text(json.dumps(metadata, indent=2))
            print(manifest.stem, json.dumps(dict(totals)), flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
