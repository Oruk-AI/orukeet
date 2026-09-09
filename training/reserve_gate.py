#!/usr/bin/env python3
"""Reserve previously unscored Common Voice dev speakers for confirmation.

Excludes all original training records, all historically scored tests, old
validation subsets, and the new selection subsets, using speaker/clip/sentence
identities. Coverage gaps remain explicit. No model predictions are read.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from audit_prepare import rows, text_key, identity, sentence, digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--audited",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-per-language",type=int,default=500)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report={}
    global_exposed_speakers=set()
    for path in list(args.original.glob("cv_*_train.json"))+list(args.original.glob("cv_*_test.json"))+list(args.original.glob("val_cv_dev_sub.json"))+list(args.audited.glob("cv_*_dev_selection.jsonl")):
        global_exposed_speakers.update(r["client_id"] for r in rows(path) if r.get("client_id"))
    for dev in sorted(args.audited.glob("cv_*_dev.jsonl")):
        lang=dev.stem.split("_")[1]
        speakers,texts,clips,ids=set(global_exposed_speakers),set(),set(),set()
        exclude=list(args.original.glob(f"*_{lang}_train.json"))+list(args.original.glob(f"*_{lang}_test.json"))+list(args.original.glob("val_*_dev_sub.json"))+list(args.audited.glob(f"*_{lang}_dev_selection.jsonl"))
        for path in exclude:
            for r in rows(path):
                if r.get("lang")!=lang:continue
                if r.get("client_id"):speakers.add(r["client_id"])
                texts.add(text_key(r["text"]))
                clips.add(identity(r))
                if sentence(r):ids.add((r.get("src"),sentence(r)))
        counts=Counter();pool=[]
        for r in rows(dev):
            counts["input_rows"]+=1
            if not r.get("speaker_id"):counts["missing_speaker"]+=1;continue
            if r["speaker_id"] in speakers:counts["previously_exposed_speaker"]+=1;continue
            if text_key(r["text"]) in texts or identity(r) in clips or (r.get("src"),sentence(r)) in ids:
                counts["previously_exposed_utterance"]+=1;continue
            rank=hashlib.sha256(("gate-20260905:"+r["audio_filepath"]).encode()).hexdigest()
            pool.append((rank,r))
        dest=args.output/f"cv_{lang}_confirmation.jsonl"
        selected=[r for _,r in sorted(pool)[:args.max_per_language]]
        with dest.open("w") as f:
            for r in selected:f.write(json.dumps(r,ensure_ascii=False)+"\n")
        report[lang]=dict(counts,eligible_rows=len(pool),selected_rows=len(selected),speakers=len({r['speaker_id'] for r in selected}),sha256=digest(dest))
        print(lang,json.dumps(report[lang]),flush=True)
    (args.output/"gate_registry.json").write_text(json.dumps({"status":"sealed_before_candidate_training", "limitations":["Legacy speaker IDs are truncated hashes","PCM near-duplicate audit still needed","This audits exposure in our recovered run, not upstream pretraining"],"languages":report},indent=2)+"\n")


if __name__=="__main__":main()
