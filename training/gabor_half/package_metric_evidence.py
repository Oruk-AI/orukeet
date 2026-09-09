"""Package a surgery comparison for CPU reproduction without transcript text."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'evaluation'))
from compare_oruk_export import cluster, compare as full_compare
from compare import compare as dev_compare
from identity import sha
from metric_qualification import qualify

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--label', required=True)
p.add_argument('--execution', choices=['source', 'native'], default='source')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
exp = a.root / 'gabor_half_20260906'
a.output.mkdir(parents=True, exist_ok=False)

def sanitize(inputs, registry, is_full, destination):
    names = list(registry['sets']) if is_full else [r['name'] for r in registry['records']]
    groups = set()
    if is_full:
        for name in names:
            for line in (inputs['reference'] / (name + '_hypotheses.jsonl')).open():
                groups.add(cluster(json.loads(line)))
    ranks = {key:f'{i:08d}' for i,key in enumerate(sorted(groups))}
    for model, source in inputs.items():
        target = destination / model
        target.mkdir(parents=True)
        meta = json.loads((source/'results.json').read_text())
        clean_meta = {k:meta[k] for k in ('normalizer','decoding','model_sha256')}
        clean_meta['sets'] = {n:{'manifest_sha256':meta['sets'][n]['manifest_sha256']} for n in names}
        (target/'results.json').write_text(json.dumps(clean_meta,indent=2)+'\n')
        for name in names:
            with (target/(name+'_hypotheses.jsonl')).open('w') as stream:
                for line in (source/(name+'_hypotheses.jsonl')).open():
                    row = json.loads(line)
                    clean = {k:row[k] for k in ('words','errors')}
                    clean['audio_filepath'] = 'record:'+hashlib.sha256(row['audio_filepath'].encode()).hexdigest()
                    clean['text'] = 'sha256:'+hashlib.sha256(row['text'].encode()).hexdigest()
                    if is_full:
                        clean['src'] = row['src']
                        clean['fleurs_id' if row['src']=='fleurs' else 'speaker_id'] = ranks[cluster(row)]
                    stream.write(json.dumps(clean)+'\n')
    if is_full:
        clean_registry = {k:v for k,v in registry.items() if k!='sets'}
        clean_registry['sets'] = {n:{k:v for k,v in r.items() if k!='path'} for n,r in registry['sets'].items()}
    else:
        clean_registry = {'records':[{k:r[k] for k in ('name','sha256','rows')} for r in registry['records']]}
    (destination/'registry.json').write_text(json.dumps(clean_registry,indent=2)+'\n')
    return clean_registry

checks = {}
for suite in ('development','larger'):
    full = suite == 'larger'
    rp = a.root/'manifests/goal_v2_confirmation/sealed_metric_registry.json' if full else exp/'development/registry.json'
    registry = json.loads(rp.read_text())
    inputs = {'reference': a.root/'eval/goal_v2/selected_confirmation' if full else exp/'eval-original',
              'candidate': exp/('full-'+a.label)/'candidate' if full else exp/('eval-'+a.label)}
    if a.execution == 'native':
        native = exp / ('native-' + a.label) / suite
        inputs = {'reference': native / 'original', 'candidate': native / 'candidate'}
    target = a.output/suite
    clean_registry = sanitize(inputs,registry,full,target)
    compare = full_compare if full else dev_compare
    actual = compare(target/'reference',target/'candidate',clean_registry)
    if full:
        dev = json.loads((a.output/'development/expected.json').read_text())
        actual = qualify(actual, dev['status'], dev['status'] != 'pass')
        if a.execution == 'native':
            actual['native_accuracy_qualified'] = actual.pop('release_qualified')
    if a.execution == 'native':
        # Native evaluators replace these prose fields; they are not statistics.
        actual.pop('interpretation' if full else 'limitations')
    expected_path = exp/('full-'+a.label)/'comparison.json' if full else exp/('comparison-'+a.label+'.json')
    if a.execution == 'native':
        expected_path = native / 'comparison.json'
    expected = json.loads(expected_path.read_text())
    for key,value in actual.items():
        assert value == expected[key], (suite,key)
    (target/'expected.json').write_text(json.dumps(actual,indent=2)+'\n')
    checks[suite] = {'comparison_fields_exact':True,'original_registry_sha256':sha(rp),
                     'raw_result_sha256':sha(expected_path),'sanitized_registry_sha256':sha(target/'registry.json')}
    if a.execution == 'native':
        checks[suite]['omitted_narrative_field'] = 'interpretation' if full else 'limitations'
    if full:
        names = {n for spec in registry['primary_languages'].values() for n in spec['sets']}
        checks[suite].update(total_rows=registry['total_rows'],primary_rows=sum(registry['sets'][n]['rows'] for n in names),
                             primary_slices=len(names),primary_languages=len(registry['primary_languages']))
receipt = {'status':'pass','label':a.label,'checks':checks,
           'audio_included':False,'transcripts_included':False,'speaker_names_included':False,
           'identity_note':'Reference/path hashes and order-preserving pseudonymous cluster ranks are retained for exact matching. This is not a formal anonymization guarantee.',
           'scope':'Reproduces stored-count statistics and development/larger qualification gates, not acoustic recognition. Raw runtime provenance fields are not regenerated.'}
if a.execution == 'native':
    receipt['execution'] = 'native'
(a.output/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
(a.output/'README.txt').write_text('Orukeet surgery metric evidence. Copyright 2026 Oruk AI. CC BY 4.0.\nStored error/word counts, reference fingerprints and pseudonymous cluster ranks.\nNo audio, transcript text or speaker names. Reproduces statistics, not ASR.\nRun training/gabor_half/reproduce_metric_evidence.py against this directory.\n')
archive = a.output.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for path in sorted(a.output.rglob('*')):
        if path.is_file():
            data=path.read_bytes();member=tarfile.TarInfo(a.output.name+'/'+path.relative_to(a.output).as_posix())
            member.size=len(data);member.mode=0o644;member.mtime=0
            tar.addfile(member,io.BytesIO(data))
print(json.dumps({'archive':str(archive),'sha256':sha(archive),'verification':receipt}),flush=True)
