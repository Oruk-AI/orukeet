#!/usr/bin/env python3
"""Select a newly produced checkpoint against stock, never on confirmation."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def decide(base_parts, candidates, protocol):
    baseline = base_parts[0]
    sets = {}
    for part in base_parts:
        for field in ['model_sha256', 'normalizer', 'decoding']:
            if baseline[field] != part[field]:
                raise ValueError('Unmatched baseline ' + field)
        if sets.keys() & part['sets'].keys():
            raise ValueError('Duplicate baseline sets')
        sets.update(part['sets'])
    assert len(sets) == 52
    d = protocol['development']
    comparisons = []
    for fraction, candidate in candidates:
        for field in ['normalizer', 'decoding']:
            if baseline[field] != candidate[field]:
                raise ValueError('Unmatched candidate ' + field)
        if sets.keys() != candidate['sets'].keys():
            raise ValueError('Incomplete candidate development coverage')
        scores, failures = {}, []
        for name, a in sets.items():
            b = candidate['sets'][name]
            if a['manifest_sha256'] != b['manifest_sha256']:
                raise ValueError('Changed manifest ' + name)
            av,bv = a['slices']['all'], b['slices']['all']
            if (av['words'],av['rows']) != (bv['words'],bv['rows']):
                raise ValueError('Changed evaluation rows ' + name)
            if not all(math.isfinite(v['wer']) for v in [av,bv]):
                raise ValueError('Invalid score')
            delta = bv['wer'] - av['wer']
            scores[name] = {'base_wer':av['wer'],'candidate_wer':bv['wer'],'delta_pp':delta}
            is_multi = name.startswith(('cv_', 'fleurs_'))
            limit = d['maximum_source_language_regression_pp'] if is_multi else d['maximum_each_english_guardrail_regression_pp']
            if delta > limit:
                failures.append({'set':name,'delta_pp':delta,'maximum_pp':limit})
        families = {}
        for source, n in [('cv',24),('fleurs',25)]:
            members = [v for k,v in scores.items() if k.startswith(source+'_')]
            assert len(members) == n
            families[source] = {key:sum(r[key] for r in members)/n for key in ['base_wer','candidate_wer','delta_pp']}
        primary = {key:sum(r[key] for r in families.values())/2 for key in ['base_wer','candidate_wer','delta_pp']}
        primary['relative_improvement'] = 1-primary['candidate_wer']/primary['base_wer']
        if primary['relative_improvement'] < d['minimum_relative_primary_improvement_over_stock']:
            failures.append({'criterion':'minimum_primary_relative_improvement','actual':primary['relative_improvement']})
        comparisons.append({'fraction':fraction,'model_path':candidate['model_path'],
            'model_sha256':candidate['model_sha256'],'eligible':not failures,'failures':failures,
            'primary':primary,'families':families,'sets':scores})
    eligible = sorted((r for r in comparisons if r['eligible']),key=lambda r:(r['primary']['candidate_wer'],r['fraction']))
    return {'status':'selected' if eligible else 'continue_development','selected':eligible[0] if eligible else None,
        'candidates':comparisons,'confirmation_used':False,'baseline_sha256':baseline['model_sha256']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    args = p.parse_args()
    root = args.root
    read = lambda path:json.loads(path.read_text())
    protocol_path = root/'goal_v2/protocol.json'
    old = root/'eval/resume_20260905'
    new = root/'eval/goal_v2'
    result = decide([read(old/'base_dev/results.json'),read(old/'base_accents/results.json'),read(new/'base_core_dev/results.json')],
        [(0.5,read(new/'a050_dev/results.json')),(0.75,read(new/'a075_dev/results.json'))],read(protocol_path))
    result['protocol_sha256'] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    with (new/'development_decision.json').open('x') as f:
        json.dump(result,f,indent=2)
        f.write('\n')
    print(json.dumps({'status':result['status'],'candidates':[{k:r[k] for k in ['fraction','eligible','failures','primary']} for r in result['candidates']]},indent=2),flush=True)


if __name__ == '__main__':
    main()
