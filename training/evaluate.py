#!/usr/bin/env python3
"""Matched greedy TDT evaluation with per-language and annotated-accent metrics."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata


CHAR_MAP = str.maketrans({"ς":"σ", "ё":"е", "Ё":"Е", "ѝ":"и", "ţ":"ț", "ş":"ș", "Ţ":"Ț", "Ş":"Ș"})
PUNCT = re.compile(r"[\"'“”„‘’«»()\[\]{}.,!?;:¡¿…\-–—/\\|*+=<>@#$%^&_~`]")


def normalize(text):
    return " ".join(PUNCT.sub(" ", unicodedata.normalize("NFC", text).translate(CHAR_MAP).lower()).split())


def distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("manifests", nargs="+", type=Path)
    args = p.parse_args()
    import torch
    from nemo.collections.asr.models import ASRModel
    from omegaconf import OmegaConf
    args.output.mkdir(parents=True, exist_ok=False)
    model = ASRModel.restore_from(str(args.model), map_location="cuda")
    model.eval()
    decode = OmegaConf.create(OmegaConf.to_container(model.cfg.decoding, resolve=True))
    decode.strategy = "greedy_batch"
    model.change_decoding_strategy(decode)
    results = {"model_path":str(args.model), "model_sha256":sha(args.model), "decoding":OmegaConf.to_container(decode,resolve=True), "normalizer":"legacy-compatible-nfc-v1", "sets":{}}
    for path in args.manifests:
        data = [json.loads(l) for l in path.open() if l.strip()]
        if not data: continue
        start = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hypotheses = model.transcribe([r["audio_filepath"] for r in data], batch_size=args.batch_size, num_workers=4, verbose=False)
        assert len(hypotheses) == len(data)
        slices = defaultdict(lambda: {"errors":0,"words":0,"char_errors":0,"chars":0,"rows":0})
        with (args.output/(path.stem+"_hypotheses.jsonl")).open("w",encoding="utf-8") as f:
            for row, hyp in zip(data, hypotheses):
                hyp = hyp.text if hasattr(hyp,"text") else hyp
                ref, prediction = normalize(row["text"]), normalize(hyp)
                words, chars = len(ref.split()), len(ref)
                errors, char_errors = distance(ref.split(),prediction.split()), distance(ref,prediction)
                record = dict(row, pred_text=hyp, errors=errors, words=words, char_errors=char_errors, chars=chars)
                f.write(json.dumps(record,ensure_ascii=False)+"\n")
                accent = row.get("accent") or row.get("accents") or "unknown"
                for key in ["all", "accent:"+accent]:
                    s=slices[key]
                    for k,v in {"errors":errors,"words":words,"char_errors":char_errors,"chars":chars,"rows":1}.items(): s[k]+=v
        for v in slices.values():
            v["wer"]=100*v["errors"]/v["words"] if v["words"] else None
            v["cer"]=100*v["char_errors"]/v["chars"] if v["chars"] else None
        results["sets"][path.stem]={"manifest_sha256":sha(path),"seconds":time.monotonic()-start,"slices":dict(slices)}
        (args.output/"results.json").write_text(json.dumps(results,indent=2,ensure_ascii=False)+"\n")
        print(path.stem, json.dumps(slices["all"]), flush=True)
    print("EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__": main()
