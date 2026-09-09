#!/usr/bin/env python3
"""Validate/normalize the recovered accent audio, preserving all source splits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unicodedata

import numpy as np
import sentencepiece as spm
import soundfile as sf
import soxr

from audit_prepare import rows,text_key


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);args=p.parse_args();root=args.root
    src=root/"data/accent_extension_20260905";out=root/"manifests/accent_extension_20260905";out.mkdir(parents=True,exist_ok=False)
    audio=src/"normalized";audio.mkdir(exist_ok=False)
    base=root/"models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo"
    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(base) as tar:
            tokenizers=[m for m in tar.getmembers() if m.isfile() and m.name.endswith('tokenizer.model')]
            if len(tokenizers)!=1:raise RuntimeError('Tokenizer ambiguity')
            path=Path(d)/'tokenizer.model';path.write_bytes(tar.extractfile(tokenizers[0]).read())
        sp=spm.SentencePieceProcessor(model_file=str(path))
    protected_text=set();protected_groups=set();protected_audio=set()
    for path in sorted((root/"manifests/audited_20260905").glob("*_en_*.jsonl")):
        if path.stem.endswith(("_dev","_test")):
            protected_text.update(text_key(r["text"]) for r in rows(path))
    for split in ["dev","test"]:
        for r in rows(src/f"{split}.jsonl"):
            protected_text.add(text_key(r["text"]))
            protected_groups.add(r.get("split_group"))
            protected_audio.add(r["audio_sha256"])
    handles={};stats={};audio_hash_splits={}
    try:
        for split in ["test","dev","train"]:
            for r in rows(src/f"{split}.jsonl"):
                source=r["source_id"];key=f"{source}_{split}";s=stats.setdefault(key,Counter());s['input_rows']+=1
                text=unicodedata.normalize('NFC',r['text']).replace('’',"'").strip()
                if split=='train':
                    if r.get('split_group') in protected_groups or r['audio_sha256'] in protected_audio or text_key(text) in protected_text:
                        s['excluded_heldout_identity_or_text']+=1;continue
                    if not 0.4<=r['duration']<=20:s['excluded_duration']+=1;continue
                    if sp.unk_id() in sp.encode(text,out_type=int):s['excluded_unknown_tokens']+=1;continue
                data,sr=sf.read(r['audio_filepath'],dtype='float32',always_2d=True)
                if not np.isfinite(data).all():s['invalid_audio']+=1;continue
                data=data.mean(axis=1)
                if sr!=16000:data=soxr.resample(data,sr,16000)
                if not len(data) or not np.any(data):s['silent_or_empty']+=1;continue
                pcm=(np.clip(data,-1,1)*32767).astype('<i2');h=hashlib.sha256(pcm.tobytes()).hexdigest()
                if h in audio_hash_splits:
                    s['duplicate_pcm']+=1
                    if audio_hash_splits[h]!=split:s['duplicate_pcm_cross_split']+=1
                    continue
                audio_hash_splits[h]=split
                dest=audio/(h+'.flac');sf.write(dest,pcm,16000,subtype='PCM_16')
                row=dict(r,audio_filepath=str(dest),text_original_manifest=r['text'],text=text,duration=len(pcm)/16000,sample_rate=16000,channels=1,pcm_sha256=h)
                if key not in handles:handles[key]=(out/(key+'.jsonl')).open('w')
                handles[key].write(json.dumps(row,ensure_ascii=False)+'\n');s['rows']+=1;s['hours']+=row['duration']/3600
    finally:
        for f in handles.values():f.close()
    balanced=root/'manifests/balanced_global_20260905';cfg=json.loads((balanced/'input_cfg.yaml').read_text())
    for g in cfg:g['weight']*=0.92
    train=[]
    for key,s in stats.items():
        if key.endswith('_train') and s['rows']:
            train.append({'type':'nemo','manifest_filepath':str(out/(key+'.jsonl')),'weight':s['hours']**0.5,'tags':{'lang':'en','src':key[:-6]}})
    if not train:raise RuntimeError('No additional training data')
    z=sum(x['weight'] for x in train)
    for x in train:x['weight']/=z
    cfg.append({'type':'group','weight':0.08,'tags':{'lang':'en','cohort':'accent_extension'},'input_cfg':train})
    (balanced/'input_cfg_with_accents.yaml').write_text(json.dumps(cfg,indent=2)+'\n')
    (out/'preparation_report.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2),flush=True)
    print('ACCENT_EXTENSION_COMPLETE',flush=True)


if __name__=='__main__':main()
