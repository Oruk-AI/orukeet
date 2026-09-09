#!/usr/bin/env python3
"""Build language/source/accent sampling from audited training data only.

Language weights use sqrt(hours), source weights use sqrt(hours), and accent
upsampling is capped at 3x its within-source empirical frequency. Tiny accent
labels (<200 clips) share a pool; unknown labels remain explicitly unknown.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from audit_prepare import rows,digest


def main():
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    inventory=json.loads((args.input/"audit.json").read_text())["inventory"]
    # The same multilingual CV speaker may record in more than one language.
    protected_speakers=set()
    for path in sorted(args.input.glob("cv_*.jsonl")):
        if path.stem.endswith(("_dev","_test")):
            protected_speakers.update(r["client_id"] for r in rows(path) if r.get("client_id"))
    lang_groups=defaultdict(list);details={}
    for src_path in sorted(args.input.glob("*_train.jsonl")):
        src,lang,_=src_path.stem.split("_")
        counts=inventory[src_path.stem]["accent_rows"]
        seconds=Counter();files={};paths={};excluded_speakers=0
        try:
            for row in rows(src_path):
                if row.get("client_id") in protected_speakers:
                    excluded_speakers+=1
                    continue
                accent=row.get("accent","unknown")
                bucket=accent if counts.get(accent,0)>=200 or accent=="unknown" else "rare_annotated"
                name=hashlib.sha256(bucket.encode()).hexdigest()[:16]
                if bucket not in files:
                    paths[bucket]=args.output/f"{src}_{lang}_accent_{name}.jsonl"
                    files[bucket]=paths[bucket].open("w",encoding="utf-8")
                files[bucket].write(json.dumps(row,ensure_ascii=False)+"\n")
                seconds[bucket]+=row["duration"]
        finally:
            for f in files.values():f.close()
        if not seconds:continue
        total=sum(seconds.values());weights={k:v**0.5 for k,v in seconds.items()};z=sum(weights.values())
        weights={k:min(weights[k]/z,3*v/total) for k,v in seconds.items()};z=sum(weights.values())
        # Renormalization can otherwise exceed the cap: allocate the remaining
        # mass to uncapped buckets until probabilities sum to one.
        for _ in range(len(weights)+2):
            remaining=1-sum(weights.values())
            if remaining<1e-12:break
            free={k for k,v in seconds.items() if weights[k]<3*v/total-1e-12}
            if not free:raise RuntimeError("No sampling capacity")
            denom=sum(seconds[k]**0.5 for k in free)
            for k in free:weights[k]=min(3*seconds[k]/total,weights[k]+remaining*seconds[k]**0.5/denom)
        inputs=[{"type":"nemo","manifest_filepath":str(paths[k]),"weight":weights[k],"tags":{"lang":lang,"src":src,"accent_pool":k}} for k in sorted(paths)]
        lang_groups[lang].append({"type":"group","weight":total**0.5,"tags":{"src":src},"input_cfg":inputs,"seconds":total})
        details[src_path.stem]={"hours":total/3600,"excluded_global_heldout_speaker_rows":excluded_speakers,"accents":{k:{"hours":seconds[k]/3600,"probability":weights[k],"multiplier":weights[k]/(seconds[k]/total),"manifest_sha256":digest(paths[k])} for k in seconds}}
    outer=[]
    for lang,groups in sorted(lang_groups.items()):
        seconds=sum(g.pop("seconds") for g in groups);z=sum(g["weight"] for g in groups)
        for g in groups:g["weight"]/=z
        outer.append({"type":"group","weight":seconds**0.5,"tags":{"lang":lang},"input_cfg":groups})
    z=sum(g["weight"] for g in outer)
    for g in outer:g["weight"]/=z
    (args.output/"input_cfg.yaml").write_text(json.dumps(outer,indent=2,ensure_ascii=False)+"\n")
    (args.output/"sampling_report.json").write_text(json.dumps(details,indent=2,ensure_ascii=False)+"\n")
    with (args.output/"selection_dev.jsonl").open("w") as f:
        for path in sorted(args.input.glob("*_dev_selection.jsonl")):f.write(path.read_text())
    print("SAMPLING_COMPLETE",flush=True)


if __name__=="__main__":main()
