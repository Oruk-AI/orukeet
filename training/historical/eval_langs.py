#!/usr/bin/env python3
"""Load a Parakeet checkpoint once and score many manifests (per language) with WER/CER.

Usage: eval_langs.py MODEL.nemo OUT.json MANIFEST [MANIFEST ...]
Normalization for scoring: NFC, lowercase, strip punctuation, collapse spaces.
Also applies the v3-tokenizer character map to *both* ref and hyp (ς->σ, ё->е, cedilla->comma)
so base-vs-finetune comparisons are fair.
"""
import json, os, re, sys, time, unicodedata
import torch
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.metrics.wer import word_error_rate

PUNCT = re.compile(r"[\"'“”„‘’«»()\[\]{}.,!?;:¡¿…\-–—/\\|*+=<>@#$%^&_~`]")
CHAR_MAP = {"\u03c2": "\u03c3", "\u0451": "\u0435", "\u0401": "\u0415", "\u045d": "\u0438",
            "\u0163": "\u021b", "\u015f": "\u0219", "\u0162": "\u021a", "\u015e": "\u0218"}


def norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = "".join(CHAR_MAP.get(c, c) for c in t).lower()
    t = PUNCT.sub(" ", t)
    return " ".join(t.split())


def main():
    model_path, out_path, manifests = sys.argv[1], sys.argv[2], sys.argv[3:]
    if model_path.endswith(".ckpt"):
        # Lightning checkpoint (EMA or raw): load its weights into the base architecture
        base = os.environ.get("BASE", "/home/nathanroll/parakeet-ft/models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo")
        model = ASRModel.restore_from(base, map_location="cuda")
        sd = torch.load(model_path, map_location="cuda", weights_only=False)["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded {model_path}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    else:
        model = ASRModel.restore_from(model_path, map_location="cuda")
    model.eval()
    results = {}
    for mp in manifests:
        name = os.path.basename(mp).replace(".json", "")
        rows = [json.loads(l) for l in open(mp, encoding="utf-8")]
        if not rows:
            continue
        paths = [r["audio_filepath"] for r in rows]
        t0 = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hyps = model.transcribe(paths, batch_size=32, num_workers=4, verbose=False)
        hyps = [h.text if hasattr(h, "text") else h for h in hyps]
        refs_n = [norm(r["text"]) for r in rows]
        hyps_n = [norm(h) for h in hyps]
        wer = word_error_rate(hyps_n, refs_n) * 100
        cer = word_error_rate(hyps_n, refs_n, use_cer=True) * 100
        hours = sum(r["duration"] for r in rows) / 3600
        results[name] = {"wer": round(wer, 2), "cer": round(cer, 2), "utts": len(rows),
                         "hours": round(hours, 2), "sec": round(time.time() - t0, 1)}
        print(f"{name:28} WER {wer:6.2f}  CER {cer:6.2f}  n={len(rows):5d}  {hours:5.2f}h  {time.time()-t0:5.1f}s", flush=True)
        with open(out_path.replace(".json", f"_{name}_hyps.jsonl"), "w", encoding="utf-8") as f:
            for r, h in zip(rows, hyps):
                f.write(json.dumps({"audio_filepath": r["audio_filepath"], "text": r["text"], "pred_text": h}, ensure_ascii=False) + "\n")
        json.dump(results, open(out_path, "w"), indent=1)
    wers = [v["wer"] for k, v in results.items() if "fleurs" in k]
    if wers:
        print(f"FLEURS macro-avg WER over {len(wers)} langs: {sum(wers)/len(wers):.2f}")
    print("EVAL_COMPLETE")


if __name__ == "__main__":
    main()
