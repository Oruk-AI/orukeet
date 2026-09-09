"""Strict paired development comparison; never treats incomplete output as pass."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

def compare(reference,candidate,registry):
    read=lambda p:json.loads(p.read_text())
    a,b=read(reference/'results.json'),read(candidate/'results.json')
    for k in ['decoding','normalizer']:
        if a[k]!=b[k]:raise ValueError('Unmatched '+k)
    expected={r['name']:r for r in registry['records']}
    if set(a['sets'])!=set(expected) or set(b['sets'])!=set(expected):raise ValueError('Incomplete evaluation')
    values={}
    for name,record in expected.items():
        for results in [a,b]:
            if results['sets'][name]['manifest_sha256']!=record['sha256']:raise ValueError('Manifest mismatch '+name)
        def rows(path):
            records=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            keyed={r['audio_filepath']:r for r in records}
            if len(keyed)!=len(records) or len(records)!=record['rows']:raise ValueError('Duplicate/missing rows')
            return keyed
        ar=rows(reference/(name+'_hypotheses.jsonl'));br=rows(candidate/(name+'_hypotheses.jsonl'))
        if ar.keys()!=br.keys():raise ValueError('Unmatched audio')
        for key in ar:
            if ar[key]['text']!=br[key]['text'] or ar[key]['words']!=br[key]['words']:raise ValueError('Unmatched references')
        words=sum(r['words'] for r in ar.values())
        if not words:raise ValueError('No reference words')
        av=100*sum(r['errors'] for r in ar.values())/words
        bv=100*sum(r['errors'] for r in br.values())/words
        values[name]={'original_wer':av,'candidate_wer':bv,'delta_pp':bv-av,'rows':len(ar),'words':words}
    def macro(names):
        return {k:mean(values[n][k] for n in names) for k in ['original_wer','candidate_wer','delta_pp']}
    cv=macro([n for n in values if n.startswith('cv_')]);fl=macro([n for n in values if n.startswith('fleurs_')])
    primary={k:(cv[k]+fl[k])/2 for k in cv}
    failures=[]
    if primary['delta_pp']>1e-12:failures.append('Primary WER exceeds original')
    for n,v in values.items():
        cap=.5 if n.startswith(('cv_','fleurs_')) else 1.
        if v['delta_pp']>cap:failures.append(f'{n}: regression {v["delta_pp"]:.6f} pp exceeds {cap}')
    return {'status':'pass' if not failures else 'fail','limitations':'Previously exposed development subset; point estimates, no unseen-data claim',
        'original_sha256':a['model_sha256'],'candidate_sha256':b['model_sha256'],
        'primary':primary,'cv_macro':cv,'fleurs_macro':fl,'sets':values,'failures':failures}

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--registry',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=compare(a.reference,a.candidate,json.loads(a.registry.read_text()))
    result['registry_sha256']=hashlib.sha256(a.registry.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sets'}))
if __name__=='__main__':main()
