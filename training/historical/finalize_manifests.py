#!/usr/bin/env python3
"""Apply v3-tokenizer-compatible text cleanup, drop residual-<unk> utterances, dedup
train against test sentences, and write final manifests + a stats table.

  manifests/final/{cv,fleurs}_<lang>_<split>.json
  manifests/final/stats.json
"""
import collections, glob, json, os, re, sys, unicodedata
import sentencepiece as spm

ROOT = "/work/users/nathanroll/parakeet-ft"
SP = f"{ROOT}/models/parakeet-tdt-0.6b-v3/nemo_extracted/902810505b78491d95378fa3000d1bc6_tokenizer.model"
sp = spm.SentencePieceProcessor(model_file=SP)
UNK = sp.unk_id()

CHAR_MAP = {
    "\u03c2": "\u03c3",   # Greek final sigma -> sigma (v3 vocab has no ς)
    "\u0451": "\u0435",   # ё -> е
    "\u0401": "\u0415",   # Ё -> Е
    "\u045d": "\u0438",   # ѝ -> и
    "\u040d": "\u0418",
    "\u0163": "\u021b", "\u015f": "\u0219",   # Romanian cedilla -> comma-below (ţ->ț, ş->ș)
    "\u0162": "\u021a", "\u015e": "\u0218",
    ";": ",",
    "\"": "", "(": "", ")": "", "[": "", "]": "", "{": "", "}": "",
    "\u00b0": "", "\u00d7": "x", "&": " and ", "+": " plus ",
}
# "&"/"+" replacements are language-agnostic approximations; they are rare (<10 per language)


def clean(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = "".join(CHAR_MAP.get(c, c) for c in t)
    # drop stray combining marks that NFC could not compose
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"\s+([,.!?:])", r"\1", t)       # no space before punctuation
    t = re.sub(r"([,.!?:])\1+", r"\1", t)       # collapse repeated punctuation
    t = " ".join(t.split())
    return t.strip(" -,")


def norm_key(t: str) -> str:
    return re.sub(r"[^\w\s]", "", t.lower())


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    out_dir = f"{ROOT}/manifests/final"
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    for src in ("fleurs", "cv"):
        for test_path in sorted(glob.glob(f"{ROOT}/manifests/{src}/*_test.json")):
            lang = os.path.basename(test_path).split("_")[0]
            test_keys = {norm_key(r["text"]) for r in load(test_path)}
            for split in ("train", "dev", "test"):
                p = f"{ROOT}/manifests/{src}/{lang}_{split}.json"
                if not os.path.exists(p):
                    continue
                rows = load(p)
                kept, dropped_unk, dropped_dup, dropped_empty, hours = [], 0, 0, 0, 0.0
                unk_chars = collections.Counter()
                for r in rows:
                    t = clean(r["text"])
                    if not t:
                        dropped_empty += 1
                        continue
                    if split == "train" and norm_key(t) in test_keys:
                        dropped_dup += 1
                        continue
                    ids = sp.encode(t, out_type=int)
                    if UNK in ids:
                        dropped_unk += 1
                        for piece, i in zip(sp.encode(t, out_type=str), ids):
                            if i == UNK:
                                unk_chars[piece] += 1
                        if split != "test":       # keep test set intact for scoring
                            continue
                    r = dict(r); r["text"] = t
                    kept.append(r); hours += r["duration"]
                with open(f"{out_dir}/{src}_{lang}_{split}.json", "w", encoding="utf-8") as f:
                    for r in kept:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                stats[f"{src}_{lang}_{split}"] = {
                    "utts": len(kept), "hours": round(hours / 3600, 2),
                    "dropped_unk": dropped_unk, "dropped_dup_vs_test": dropped_dup,
                    "dropped_empty": dropped_empty, "unk_chars": dict(unk_chars.most_common(5)),
                }
                print(f"{src:6} {lang:3} {split:5} utts={len(kept):7d} h={hours/3600:8.2f} "
                      f"unk_drop={dropped_unk} dup_drop={dropped_dup} {dict(unk_chars.most_common(3))}", flush=True)
    json.dump(stats, open(f"{out_dir}/stats.json", "w"), indent=1, ensure_ascii=False)
    print("FINALIZE_COMPLETE")


if __name__ == "__main__":
    main()
