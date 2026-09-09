#!/usr/bin/env python3
"""Paired global cluster bootstrap; no independent resampling of shared speakers."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from seal_confirmation import cluster


def interval(values):
    finite = values[np.isfinite(values)]
    if len(finite) < 0.8*len(values):
        return {'paired_delta_95ci':None,'valid_bootstrap_replicates':len(finite)}
    return {'paired_delta_95ci':np.quantile(finite,[.025,.975]).tolist(),
            'valid_bootstrap_replicates':len(finite)}


def compare(reference,candidate,registry,protocol,replicates=5000):
    read = lambda p:json.loads(p.read_text())
    a,b = read(reference/'results.json'),read(candidate/'results.json')
    for field in ['normalizer','decoding']:
        if a[field] != b[field]:
            raise ValueError('Unmatched '+field)
    expected = set(registry['sets'])
    if set(a['sets']) != expected or set(b['sets']) != expected:
        raise ValueError('Incomplete matched confirmation')
    sets = {}
    all_groups = set()
    for name in sorted(expected):
        for evaluation in [a,b]:
            if evaluation['sets'][name]['manifest_sha256'] != registry['sets'][name]['manifest_sha256']:
                raise ValueError('Manifest changed: '+name)
        def load(directory):
            with (directory/(name+'_hypotheses.jsonl')).open() as f:
                return {r['audio_filepath']:r for r in (json.loads(line) for line in f)}
        ar,br = load(reference),load(candidate)
        if ar.keys() != br.keys() or len(ar) != registry['sets'][name]['rows']:
            raise ValueError('Unmatched recordings: '+name)
        data = []
        for key,ra in ar.items():
            rb = br[key]
            if ra['text'] != rb['text'] or ra['words'] != rb['words'] or cluster(ra) != cluster(rb):
                raise ValueError('Unmatched reference or grouping')
            group = cluster(ra)
            all_groups.add(group)
            data.append((group,ra['errors'],rb['errors'],ra['words'],ra))
        sets[name] = data
    metrics = {'set:'+name:data for name,data in sets.items()}
    for lang,entry in registry['primary_languages'].items():
        metrics['language:'+lang] = [r for name in entry['sets'] for r in sets[name]]
    for corpus,names in registry['english_corpora'].items():
        metrics['english:'+corpus] = [r for name in names for r in sets[name]]
    # Disclose named accent slices and the novel-prompt portions of the two
    # source-speaker holdouts. These do not influence selection or gate rules.
    accents = defaultdict(list)
    for name,data in sets.items():
        if registry['sets'][name]['language'] != 'en':
            continue
        for row in data:
            label = row[4].get('accent') or row[4].get('accents') or 'unknown'
            if label.lower() not in ['unknown','none','na','']:
                accents[registry['sets'][name]['source']+':'+label].append(row)
    metrics.update({'accent:'+label:data for label,data in accents.items()})
    for corpus in ['learner','dialect']:
        data = metrics['english:'+corpus]
        novel = [r for r in data if not (r[4].get('transcript_seen_in_prior_campaign') or r[4].get('transcript_seen_in_prior_source_partitions'))]
        if novel:
            metrics['novel_prompt:'+corpus] = novel
    group_order = {g:i for i,g in enumerate(sorted(all_groups))}
    names = list(metrics)
    matrix = np.zeros((len(group_order),len(names),3),dtype=np.float64)
    for j,name in enumerate(names):
        for group,ae,be,words,_ in metrics[name]:
            matrix[group_order[group],j] += [ae,be,words]
    totals = matrix.sum(axis=0)
    rng = np.random.default_rng(20260905)
    differences = np.full((replicates,len(names)),np.nan)
    flat = matrix.reshape(len(group_order),-1)
    probabilities = np.full(len(group_order),1/len(group_order))
    for start in range(0,replicates,100):
        n = min(100,replicates-start)
        weights = rng.multinomial(len(group_order),probabilities,size=n)
        draws = (weights @ flat).reshape(n,len(names),3)
        with np.errstate(divide='ignore',invalid='ignore'):
            differences[start:start+n] = 100*(draws[:,:,1]-draws[:,:,0])/draws[:,:,2]
    results = {}
    for j,name in enumerate(names):
        total = totals[j]
        av,bv = 100*total[:2]/total[2]
        groups = len({r[0] for r in metrics[name]})
        results[name] = {'base_wer':float(av),'candidate_wer':float(bv),'delta_pp':float(bv-av),
            'relative_improvement':float(1-bv/av) if av else None,'rows':len(metrics[name]),
            'groups':groups,'reference_words':int(total[2]),'population_inference_supported':groups>=5,
            **interval(differences[:,j])}
    index = {name:i for i,name in enumerate(names)}
    def macro(members):
        av = float(np.mean([results[n]['base_wer'] for n in members]))
        bv = float(np.mean([results[n]['candidate_wer'] for n in members]))
        return {'base_wer':av,'candidate_wer':bv,'delta_pp':bv-av,'relative_improvement':1-bv/av,
                'members':members,**interval(differences[:,[index[n] for n in members]].mean(axis=1))}
    primary = macro(['language:'+lang for lang in registry['primary_languages']])
    families = {source:macro(['set:'+name for name in members]) for source,members in registry['families'].items()}
    english = macro(['english:'+name for name in registry['english_corpora']])
    failures = []
    if primary['relative_improvement'] < protocol['confirmation']['minimum_relative_primary_improvement_over_stock']:
        failures.append('Primary relative improvement is below 10%')
    if primary['paired_delta_95ci'] is None or primary['paired_delta_95ci'][1] >= 0:
        failures.append('Primary paired 95% interval is not entirely below zero')
    for family,value in families.items():
        if value['delta_pp'] >= 0:
            failures.append('Multilingual family did not improve: '+family)
    for corpus in registry['english_corpora']:
        value = results['english:'+corpus]
        if value['delta_pp'] > .3:
            failures.append('English corpus regression exceeds 0.3 pp: '+corpus)
    for label,value in [('English aggregate',english)]+[(name,results['english:'+name]) for name in ['core','learner','dialect']]:
        if value['paired_delta_95ci'] is None or value['paired_delta_95ci'][1] >= .3:
            failures.append('English noninferiority is unconfirmed at +0.3 pp: '+label)
    return {'status':'passed' if not failures else 'failed','failures':failures,'primary':primary,
        'families':families,'english_aggregate':english,'metrics':results,
        'base_model_sha256':a['model_sha256'],'candidate_model_sha256':b['model_sha256'],
        'bootstrap':{'replicates':replicates,'seed':20260905,'distinct_global_clusters':len(group_order),
            'method':'Paired multinomial bootstrap of all source-qualified global clusters; identical draws for both models and all metrics',
            'unit':'speaker; FLEURS parallel sentence identity shared across languages',
            'scope':'Fixed declared languages and corpora, not a random sample of all languages',
            'empty_draw_handling':'Only undefined metric replicates are omitted; report count and suppress interval if fewer than 80% remain'},
        'limitations':['Public pretraining overlap is unknown','FLEURS sentence clusters are not speaker clusters',
            'CV uses inherited truncated speaker hashes','RixVox protocols are not always verbatim',
            'Repeated English Dialects prompts are present; novel-prompt results are diagnostic',
            'Per-accent intervals are descriptive, without simultaneous multiple-comparison guarantees']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    args = p.parse_args()
    root = args.root
    registry_path = root/'manifests/goal_v2_confirmation/sealed_metric_registry.json'
    protocol_path = root/'goal_v2/protocol.json'
    registry = json.loads(registry_path.read_text())
    if registry['protocol_sha256'] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise ValueError('Protocol changed since sealing')
    result = compare(root/'eval/goal_v2/base_confirmation',root/'eval/goal_v2/selected_confirmation',
        registry,json.loads(protocol_path.read_text()))
    decision = json.loads((root/'eval/goal_v2/development_decision.json').read_text())
    protocol = json.loads(protocol_path.read_text())
    if result['base_model_sha256'] != protocol['baseline_sha256']:
        raise ValueError('Incorrect stock model')
    if decision['status'] != 'selected' or result['candidate_model_sha256'] != decision['selected']['model_sha256']:
        raise ValueError('Confirmation candidate differs from development selection')
    result['registry_sha256'] = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    result['protocol_sha256'] = registry['protocol_sha256']
    output = root/'eval/goal_v2/confirmation_comparison.json'
    with output.open('x') as f:
        json.dump(result,f,indent=2)
        f.write('\n')
    print(json.dumps({k:result[k] for k in ['status','failures','primary','families','english_aggregate']},indent=2),flush=True)


if __name__ == '__main__':
    main()
