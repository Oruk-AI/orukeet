#!/usr/bin/env python3
"""Paired comparisons from evaluate.py outputs; bootstrap speakers within sets."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random


def load_rows(directory,name):
    return {r['audio_filepath']:r for r in (json.loads(l) for l in (directory/(name+'_hypotheses.jsonl')).open())}


def compare(a_dir,b_dir,boots=2000,minimum_groups=1):
    a=json.loads((a_dir/'results.json').read_text());b=json.loads((b_dir/'results.json').read_text())
    if a['normalizer']!=b['normalizer'] or a['decoding']!=b['decoding']:raise ValueError('Unmatched evaluation configuration')
    if a['sets'].keys()!=b['sets'].keys():raise ValueError('Different manifest sets')
    result={};rng=random.Random(20260905);distributions=[]
    for name in a['sets']:
        if a['sets'][name]['manifest_sha256']!=b['sets'][name]['manifest_sha256']:raise ValueError(name+' manifest changed')
        ar=load_rows(a_dir,name);br=load_rows(b_dir,name)
        if ar.keys()!=br.keys():raise ValueError(name+' utterance mismatch')
        groups=defaultdict(lambda:[0,0,0])
        for key,ra in ar.items():
            rb=br[key]
            if ra['text']!=rb['text'] or ra['words']!=rb['words']:raise ValueError('Reference mismatch')
            group=ra.get('speaker_id') or ra.get('client_id') or ra.get('fleurs_id') or key
            g=groups[group];g[0]+=ra['errors'];g[1]+=rb['errors'];g[2]+=ra['words']
        gs=list(groups.values());wa=sum(g[2] for g in gs)
        if len(gs)<minimum_groups:continue
        if not wa:continue
        av=100*sum(g[0] for g in gs)/wa;bv=100*sum(g[1] for g in gs)/wa
        deltas=[]
        for _ in range(boots):
            selected=rng.choices(gs,k=len(gs));words=sum(g[2] for g in selected)
            if words:deltas.append(100*sum(g[1]-g[0] for g in selected)/words)
        ordered=sorted(deltas)
        lo=ordered[int(.025*len(ordered))];hi=ordered[min(len(ordered)-1,int(.975*len(ordered)))]
        result[name]={'base_wer':av,'candidate_wer':bv,'delta_pp':bv-av,'reference_words':wa,'rows':len(ar),'resampling_groups':len(gs),'paired_delta_95ci':[lo,hi], 'population_inference_supported':len(gs)>=5}
        distributions.append(deltas)
    macro_a=sum(x['base_wer'] for x in result.values())/len(result);macro_b=sum(x['candidate_wer'] for x in result.values())/len(result)
    # Equal weight to each included source-language set, with speakers resampled
    # within each set. Scope is the fixed included languages, not all languages.
    n=min(map(len,distributions));macro_boot=sorted(sum(ds[i] for ds in distributions)/len(distributions) for i in range(n))
    return {'reference_model':a['model_sha256'],'candidate_model':b['model_sha256'],'sets':result,'macro':{'base_wer':macro_a,'candidate_wer':macro_b,'delta_pp':macro_b-macro_a,'paired_delta_95ci':[macro_boot[int(.025*n)],macro_boot[min(n-1,int(.975*n))]]},'bootstrap':{'replicates':boots,'seed':20260905,'unit':'speaker when available; FLEURS sentence ID otherwise; utterance fallback','scope':'fixed included source-language sets'},'limitations':['Source speaker IDs may be unavailable or truncated','Sets with fewer than 5 groups cannot support population-level slice inference','No claim of upstream pretraining decontamination']}


def main():
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--minimum-groups',type=int,default=1);args=p.parse_args()
    r=compare(args.reference,args.candidate,args.bootstrap,args.minimum_groups);r['bootstrap']['minimum_groups']=args.minimum_groups;args.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['macro'],indent=2))


if __name__=='__main__':main()
