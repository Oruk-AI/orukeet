#!/usr/bin/env python3
"""Audit the recovered CV/FLEURS manifests and build a new, immutable train/dev set.

No test labels influence sampling weights. Held-out identities are used only as
exclusions. Accent labels are source annotations, never inferred from language.
The report distinguishes identity checks from acoustic and speaker verification.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata


def text_key(text):
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text).split())


def rows(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def identity(row):
    # CV filenames are globally unique within a language, independent of split.
    return (row.get("src", ""), row.get("lang", ""), Path(row["audio_filepath"]).name)


def sentence(row):
    return row.get("sentence_id") or ("fleurs:" + row["fleurs_id"] if row.get("fleurs_id") else "")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(row, source_path, split):
    r = dict(row)
    r["text_original_manifest"] = row["text"]
    r["text"] = unicodedata.normalize("NFC", row["text"])
    r["source_release"] = "common_voice_22" if row["src"] == "cv" else "fleurs_recovered_2026_09_04"
    r["source_manifest"] = str(source_path)
    r["split"] = split
    r["speaker_id"] = row.get("client_id") or None
    r["accent"] = " ".join(row.get("accents", "").casefold().split()) or "unknown"
    r["sample_rate"] = 16000
    r["channels"] = 1
    r["transcript_provenance"] = "human_source_after_legacy_tokenizer_cleanup"
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dev-per-language-source", type=int, default=150)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = sorted(x for x in args.input.glob("*.json") if re.fullmatch(r"(?:cv|fleurs)_[a-z]{2}_(?:train|dev|test)\.json", x.name))
    inventory, rejected, groups = {}, {}, []
    languages = sorted({x.stem.split("_")[1] for x in inputs})
    for lang in languages:
        protected_audio, protected_text, protected_speakers, protected_sentences = set(), set(), set(), set()
        for path in inputs:
            src, lng, split = path.stem.split("_")
            if lng != lang or split == "train":
                continue
            for r in rows(path):
                protected_audio.add(identity(r))
                protected_text.add(text_key(r["text"]))
                if r.get("client_id"):
                    protected_speakers.add(r["client_id"])
                if sentence(r):
                    protected_sentences.add((src, sentence(r)))
        for path in (x for x in inputs if x.stem.split("_")[1] == lang):
            src, _, split = path.stem.split("_")
            out = args.output / (path.stem + ".jsonl")
            counts, accents, accent_seconds, speakers = Counter(), Counter(), Counter(), set()
            seconds = 0.0
            seen = set()
            dev = []
            with out.open("w", encoding="utf-8") as f:
                for row in rows(path):
                    counts["input_rows"] += 1
                    duration = row.get("duration", 0)
                    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0 or not row.get("text", "").strip():
                        counts["invalid"] += 1
                        continue
                    key = identity(row)
                    reasons = []
                    if split == "train":
                        if key in protected_audio: reasons.append("heldout_audio_id")
                        if text_key(row["text"]) in protected_text: reasons.append("heldout_sentence_text")
                        if row.get("client_id") in protected_speakers: reasons.append("heldout_speaker")
                        if sentence(row) and (src, sentence(row)) in protected_sentences: reasons.append("heldout_sentence_id")
                        if not 0.4 <= duration <= 20: reasons.append("training_duration")
                    if key in seen: reasons.append("duplicate_audio_id")
                    if reasons:
                        counts.update(reasons)
                        counts["excluded_unique_rows"] += 1
                        continue
                    seen.add(key)
                    if not Path(row["audio_filepath"]).is_file():
                        counts["missing_audio"] += 1
                        continue
                    r = canonical(row, path, split)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    counts["retained_rows"] += 1
                    seconds += duration
                    accents[r["accent"]] += 1
                    accent_seconds[r["accent"]] += duration
                    if r["speaker_id"]: speakers.add(r["speaker_id"])
                    if split == "dev":
                        rank = hashlib.sha256(("20260905:" + r["audio_filepath"]).encode()).hexdigest()
                        dev.append((rank, r))
            report = dict(counts, hours=seconds / 3600, speakers=len(speakers), accent_rows=dict(accents), accent_hours={k:v/3600 for k,v in accent_seconds.items()}, sha256=digest(out), source_sha256=digest(path))
            inventory[path.stem] = report
            print(path.stem, counts["retained_rows"], round(seconds / 3600, 2), "excluded", counts["excluded_unique_rows"], flush=True)
            if split == "train" and counts["retained_rows"]:
                groups.append({"type":"nemo", "manifest_filepath":str(out), "weight":seconds ** 0.5, "tags":{"lang":lang,"src":src}})
            if split == "dev":
                dev_path = args.output / (path.stem + "_selection.jsonl")
                with dev_path.open("w", encoding="utf-8") as f:
                    for _, r in sorted(dev)[:args.dev_per_language_source]:
                        f.write(json.dumps(r, ensure_ascii=False)+"\n")
    total = sum(g["weight"] for g in groups)
    for g in groups: g["weight"] /= total
    # JSON is valid YAML and can be consumed by OmegaConf/Lhotse input_cfg.
    (args.output/"train_input_cfg.yaml").write_text(json.dumps(groups, indent=2)+"\n")
    result = {"schema_version":1,"source_manifests":len(inputs),"languages":languages,"inventory":inventory,
              "verification":{"all_retained_audio_paths_exist":True,"pcm_deduplicated":False,"cv_speaker_ids":"legacy 16-character client_id prefix", "fleurs_speaker_ids":"unavailable", "eval_references":"legacy transformed; not untouched raw references", "test_status":"historically inspected; diagnostic only"}}
    (args.output/"audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    if any(v.get("missing_audio",0) or v.get("invalid",0) for v in inventory.values()):
        raise SystemExit("Audit requires review: missing audio or invalid rows")


if __name__ == "__main__":
    main()
