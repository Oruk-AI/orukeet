"""Prepare shareable metric evidence without audio, transcripts or speaker names.

Counts retain their original normalization and manifest identities. Pseudonymous
cluster ranks preserve the original sort order so seeded bootstrap draws remain
exactly reproducible. This cannot replace audio-based recognition evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path

from compare_oruk_export import cluster, compare


def prepare(reference, source, candidate, registry_path, output):
    registry=json.loads(registry_path.read_text())
    inputs={'stock':reference,'source':source,'q8':candidate}
    groups=set()
    for name in registry['sets']:
        for line in (reference/(name+'_hypotheses.jsonl')).read_text().splitlines():
            groups.add(cluster(json.loads(line)))
    ranks={key:f'{i:08d}' for i,key in enumerate(sorted(groups))}
    output.mkdir(parents=True,exist_ok=False)
    for model,root in inputs.items():
        dest=output/model;dest.mkdir()
        original=json.loads((root/'results.json').read_text())
        metadata={key:original[key] for key in ('normalizer','model_sha256')}
        metadata['reference_representation']='sha256 of original reference; use stored error/word counts'
        metadata['sets']={name:{'manifest_sha256':original['sets'][name]['manifest_sha256']} for name in registry['sets']}
        (dest/'results.json').write_text(json.dumps(metadata,indent=2)+'\n')
        for name in registry['sets']:
            with (dest/(name+'_hypotheses.jsonl')).open('w') as stream:
                for line in (root/(name+'_hypotheses.jsonl')).read_text().splitlines():
                    row=json.loads(line)
                    clean={key:row[key] for key in ('src','words','errors')}
                    clean['audio_filepath']='record:'+hashlib.sha256(row['audio_filepath'].encode()).hexdigest()
                    clean['text']='sha256:'+hashlib.sha256(row['text'].encode()).hexdigest()
                    clean['fleurs_id' if row['src']=='fleurs' else 'speaker_id']=ranks[cluster(row)]
                    stream.write(json.dumps(clean)+'\n')
    sanitized={key:value for key,value in registry.items() if key!='sets'}
    sanitized['sets']={name:{key:value for key,value in spec.items() if key!='path'} for name,spec in registry['sets'].items()}
    (output/'registry.json').write_text(json.dumps(sanitized,indent=2)+'\n')
    expected=compare(reference,candidate,registry)
    actual=compare(output/'stock',output/'q8',sanitized)
    assert actual==expected, 'Sanitization changed a reported statistic'
    assert compare(source,candidate,registry)==compare(output/'source',output/'q8',sanitized)
    (output/'verification.json').write_text(json.dumps({'status':'passed','comparison':'All comparison fields exactly match raw stock/Q8 evidence.','source_comparison':'All comparison fields exactly match raw source/Q8 evidence.','original_registry_sha256':hashlib.sha256(registry_path.read_bytes()).hexdigest(),'records_per_model':registry['total_rows'],'raw_audio':False,'transcript_text':False,'speaker_names':False,'identity_note':'Pseudonymous cluster ranks preserve grouping; not a formal anonymization guarantee.'},indent=2)+'\n')
    (output/'README.txt').write_text('Orukeet metric evidence. Copyright 2026 Oruk AI.\nCC BY 4.0: https://creativecommons.org/licenses/by/4.0/\n\nPer-recording errors and word counts, computed with the original references.\nText fields contain reference SHA-256 fingerprints, not transcript text.\nCluster ranks preserve shared speaker/sentence grouping and bootstrap order.\nNo audio or speaker names are included. This reproduces statistics, not ASR.\nSource corpora retain their own licenses; see docs/data-and-licenses.md.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','source','candidate','registry','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    prepare(args.reference,args.source,args.candidate,args.registry,args.output)
