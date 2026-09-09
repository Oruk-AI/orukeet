#!/usr/bin/env python3
"""Deterministic text-offset probes; never repair references or alter scoring."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from rapidfuzz.distance import Levenshtein
from metrics import normalize
from run import atomic_json, read_rows, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args()
    seal=json.loads((a.experiment/'metadata/seal.json').read_text());result={}
    for bucket,info in sorted(seal['sets'].items()):
        if not bucket.startswith('eurospeech_'):continue
        rows=read_rows(info['path']);byuid={r['uid']:r for r in rows}
        preds={r['uid']:r['pred_text'] for r in read_rows(a.experiment/'results/parakeet'/(bucket+'.jsonl'))}
        tokens={r['uid']:normalize(r['text']).split() for r in rows}
        # Membership is selected using reference length and source ID only.
        eligible=sorted((r for r in rows if len(tokens[r['uid']])>=15),key=lambda r:r['uid'])
        selected=[eligible[i] for i in np.linspace(0,len(eligible)-1,min(16,len(eligible)),dtype=int)]
        probes=[]
        for row in selected:
            words=normalize(preds[row['uid']]).split();ref=tokens[row['uid']]
            assigned=Levenshtein.distance(words,ref)/max(len(words),len(ref),1)
            candidates=[(Levenshtein.distance(words,ref)/max(len(words),len(ref),1),uid) for uid,ref in tokens.items() if byuid[uid]['cluster']==row['cluster']]
            distance,uid=min(candidates);offset=byuid[uid]['start_seconds']-row['start_seconds']
            evidence=len(words)>=8 and assigned>.85 and distance<.6 and abs(offset)>20
            probes.append({'record_sha256':hashlib.sha256(row['uid'].encode()).hexdigest(),'assigned_distance':assigned,
                           'nearest_distance':distance,'nearest_start_offset_seconds':offset,'strong_offset_signal':evidence})
        result[bucket]={'eligible_rows':len(eligible),'probes':probes,'strong_offset_signals':sum(r['strong_offset_signal'] for r in probes)}
        print(bucket,result[bucket]['strong_offset_signals'],'/',len(probes),flush=True)
    atomic_json(a.experiment/'reference-offset-probes.json',{
        'purpose':'Post-inference diagnostic, not a scoring variant or exhaustive reference-quality audit.',
        'selection':'16 evenly spaced IDs per language among references with at least 15 normalized words.',
        'signal_definition':'At least eight predicted words; assigned normalized word distance >0.85, closest same-session reference distance <0.6, and timestamp offset >20 seconds. This heuristic flags evidence for inspection; it does not establish a population error rate.',
        'script_sha256':sha(__file__),'sets':result})


if __name__=='__main__':main()
