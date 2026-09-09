#!/usr/bin/env python3
"""Common Voice 22.0 (fsicoli mirror) mp3 tars + TSVs -> 16 kHz FLAC + NeMo manifests.

Uses official train/dev/test splits (speaker-disjoint by construction).
Text target = `sentence` (cased, punctuated), NFC-normalized, quotes unified.
Output:
  data/cv/<iso>/<split>/<clip>.flac
  manifests/cv/<iso>_<split>.json
"""
import csv, glob, io, json, os, sys, tarfile, unicodedata
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import soundfile as sf
import soxr

ROOT = "/work/users/nathanroll/parakeet-ft"
RAW = f"{ROOT}/cv_raw"
LOCALES = {
    "bg": "bg", "cs": "cs", "da": "da", "nl": "nl", "en": "en", "et": "et", "fi": "fi",
    "fr": "fr", "de": "de", "el": "el", "hu": "hu", "it": "it", "lv": "lv", "lt": "lt",
    "mt": "mt", "pl": "pl", "pt": "pt", "ro": "ro", "ru": "ru", "sk": "sk", "sl": "sl",
    "es": "es", "sv-SE": "sv", "uk": "uk",
}
QUOTES = {"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
          "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
          "\u00ab": '"', "\u00bb": '"', "\u2039": "'", "\u203a": "'",
          "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " ", "\u2009": " ", "\u202f": " "}
MIN_DUR, MAX_DUR = 0.3, 20.0


def norm_text(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = "".join(QUOTES.get(c, c) for c in t)
    return " ".join(t.split())


def read_tsv(path):
    rows = {}
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for r in rd:
            p = os.path.basename(r.get("path", ""))
            if p:
                rows[p] = r
    return rows


def process(args):
    loc, iso, split, tar_path = args
    tsv = read_tsv(f"{RAW}/transcript/{loc}/{split}.tsv")
    out_dir = f"{ROOT}/data/cv/{iso}/{split}"
    os.makedirs(out_dir, exist_ok=True)
    shard = os.path.basename(tar_path)[:-4]
    man_path = f"{ROOT}/manifests/cv/parts/{iso}_{split}_{shard}.json"
    os.makedirs(os.path.dirname(man_path), exist_ok=True)
    if os.path.exists(man_path + ".done"):
        return loc, split, shard, -1, 0.0, 0, 0
    n = 0; dur_total = 0.0; missing = 0; bad = 0
    with tarfile.open(tar_path, "r") as tar, open(man_path, "w", encoding="utf-8") as man:
        for m in tar:
            if not m.isfile() or not m.name.endswith(".mp3"):
                continue
            fname = os.path.basename(m.name)
            r = tsv.get(fname)
            if r is None:
                missing += 1
                continue
            text = norm_text(r.get("sentence") or "")
            if not text:
                continue
            try:
                data, sr = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="float32")
            except Exception:
                bad += 1
                continue
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                data = soxr.resample(data, sr, 16000, quality="HQ")
            dur = len(data) / 16000
            if dur < MIN_DUR or dur > MAX_DUR:
                continue
            out = f"{out_dir}/{fname[:-4]}.flac"
            if not os.path.exists(out):
                sf.write(out, (np.clip(data, -1, 1) * 32767).astype("int16"), 16000, format="FLAC")
            man.write(json.dumps({
                "audio_filepath": out, "duration": round(dur, 3), "text": text,
                "lang": iso, "src": "cv", "client_id": r.get("client_id", "")[:16],
                "sentence_id": r.get("sentence_id", ""), "gender": r.get("gender", ""),
                "accents": r.get("accents", ""),
            }, ensure_ascii=False) + "\n")
            n += 1; dur_total += dur
    open(man_path + ".done", "w").close()
    return loc, split, shard, n, round(dur_total / 3600, 2), missing, bad


if __name__ == "__main__":
    jobs = []
    for loc, iso in LOCALES.items():
        for split in ("train", "dev", "test"):
            for tp in sorted(glob.glob(f"{RAW}/audio/{loc}/{split}/*.tar")):
                jobs.append((loc, iso, split, tp))
    # big shards first so the pool stays busy
    jobs.sort(key=lambda j: -os.path.getsize(j[3]))
    print(f"{len(jobs)} tar jobs", flush=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 36
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for loc, split, shard, n, h, miss, bad in ex.map(process, jobs):
            print(f"{loc:6} {split:5} {shard:16} utts={n:7d} hours={h:7.2f} missing_tsv={miss} bad={bad}", flush=True)
    # merge parts into per-language/split manifests
    for loc, iso in LOCALES.items():
        for split in ("train", "dev", "test"):
            parts = sorted(glob.glob(f"{ROOT}/manifests/cv/parts/{iso}_{split}_*.json"))
            with open(f"{ROOT}/manifests/cv/{iso}_{split}.json", "w", encoding="utf-8") as out:
                for p in parts:
                    with open(p, encoding="utf-8") as f:
                        out.write(f.read())
    print("CV_PREP_COMPLETE", flush=True)
