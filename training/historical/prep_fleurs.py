#!/usr/bin/env python3
"""FLEURS tar.gz + TSV -> 16 kHz FLAC + NeMo manifests (train/dev/test per language).

Text target = raw_transcription (cased, punctuated), NFC-normalized with unified quotes.
Output layout:
  data/fleurs/<iso>/<split>/<id>.flac
  manifests/fleurs/<iso>_<split>.json   (audio_filepath, duration, text, lang, src="fleurs")
"""
import csv, io, json, os, sys, tarfile, unicodedata
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import soundfile as sf

ROOT = "/work/users/nathanroll/parakeet-ft"
RAW = f"{ROOT}/fleurs_raw/data"
LANGS = {
    "bg_bg": "bg", "hr_hr": "hr", "cs_cz": "cs", "da_dk": "da", "nl_nl": "nl",
    "en_us": "en", "et_ee": "et", "fi_fi": "fi", "fr_fr": "fr", "de_de": "de",
    "el_gr": "el", "hu_hu": "hu", "it_it": "it", "lv_lv": "lv", "lt_lt": "lt",
    "mt_mt": "mt", "pl_pl": "pl", "pt_br": "pt", "ro_ro": "ro", "ru_ru": "ru",
    "sk_sk": "sk", "sl_si": "sl", "es_419": "es", "sv_se": "sv", "uk_ua": "uk",
}
QUOTES = {"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
          "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
          "\u00ab": '"', "\u00bb": '"', "\u2039": "'", "\u203a": "'",
          "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " ", "\u2009": " ", "\u202f": " "}


def norm_text(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = "".join(QUOTES.get(c, c) for c in t)
    return " ".join(t.split())


def read_tsv(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            _id, fname, raw, _norm, _chars, _nsamp, gender = p[:7]
            rows[fname] = (_id, raw, gender)
    return rows


def process(args):
    code, iso, split = args
    tsv = read_tsv(f"{RAW}/{code}/{split}.tsv")
    out_dir = f"{ROOT}/data/fleurs/{iso}/{split}"
    os.makedirs(out_dir, exist_ok=True)
    man_path = f"{ROOT}/manifests/fleurs/{iso}_{split}.json"
    os.makedirs(os.path.dirname(man_path), exist_ok=True)
    n = 0; dur_total = 0.0; missing = 0
    with tarfile.open(f"{RAW}/{code}/audio/{split}.tar.gz", "r:gz") as tar, \
            open(man_path, "w", encoding="utf-8") as man:
        for m in tar:
            if not m.isfile() or not m.name.endswith(".wav"):
                continue
            fname = os.path.basename(m.name)
            meta = tsv.get(fname)
            if meta is None:
                missing += 1
                continue
            _id, raw, gender = meta
            text = norm_text(raw)
            if not text:
                continue
            # FLEURS wavs are 32-bit float PCM; read as float and scale to int16
            data, sr = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                raise RuntimeError(f"unexpected sr {sr} in {code}/{split}/{fname}")
            dur = len(data) / sr
            if float(np.abs(data).max()) == 0.0:
                missing += 1  # silent file; skip
                continue
            out = f"{out_dir}/{fname[:-4]}.flac"
            if not os.path.exists(out):
                sf.write(out, (np.clip(data, -1, 1) * 32767).astype("int16"), sr, format="FLAC")
            man.write(json.dumps({
                "audio_filepath": out, "duration": round(dur, 3), "text": text,
                "lang": iso, "src": "fleurs", "fleurs_id": _id, "gender": gender,
            }, ensure_ascii=False) + "\n")
            n += 1; dur_total += dur
    return code, split, n, round(dur_total / 3600, 2), missing


if __name__ == "__main__":
    jobs = [(c, i, s) for c, i in LANGS.items() for s in ("train", "dev", "test")]
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 12) as ex:
        for code, split, n, h, miss in ex.map(process, jobs):
            print(f"{code:6} {split:5} utts={n:6d} hours={h:6.2f} missing_tsv={miss}", flush=True)
    print("FLEURS_PREP_COMPLETE", flush=True)
