"""Re-score the private hypotheses and compare every record with the targeted-pass counts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import ast
from collections import defaultdict
from difflib import SequenceMatcher
import importlib.util
import re
import num2words
from kaldialign import batch_error_rate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'evaluation/standard_asr'))
from scoring import counts


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--private-records',type=Path,required=True)
    p.add_argument('--evidence',type=Path,default=ROOT/'evidence/targeted-ft-20260908')
    p.add_argument('--upstream-root',type=Path)
    a=p.parse_args()
    result=json.loads((a.evidence/'comparison.json').read_text())
    manifest=a.private_records/'manifest.jsonl'
    assert sha(manifest)==result['manifest_sha256']
    rows=[json.loads(line) for line in manifest.open()]
    numeric={}
    with gzip.open(a.evidence/'numeric-evidence.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            assert row['record_sha256'] not in numeric
            numeric[row['record_sha256']]=row
    assert len(numeric)==len(rows)==result['rows']
    empty={}
    paired=defaultdict(lambda: [[], []])
    for model in result['models']:
        path=a.private_records/(model+'.jsonl')
        assert sha(path)==result['prediction_sha256'][model]
        records=[json.loads(line) for line in path.open()]
        predictions={r['uid']:r for r in records}
        assert len(predictions)==len(records)==len(rows)
        empty[model]=0
        for row in rows:
            prediction=predictions[row['uid']]
            assert prediction['model_sha256']==result['models'][model]
            assert prediction['manifest_sha256']==result['manifest_sha256']
            values=counts(row['text'],prediction['prediction'],row['language'])
            expected=numeric[hashlib.sha256(row['uid'].encode()).hexdigest()]
            assert expected['split']==row['split']
            for metric in ['errors','words','char_errors','chars','utterance_error']:
                assert values[metric]==expected['counts'][model][metric],(row['uid'],model,metric)
            empty[model]+=not bool(prediction['prediction'].strip())
            paired[(row['split'], model)][0].append(row['text'])
            paired[(row['split'], model)][1].append(prediction['prediction'])
    audit=dict(status='passed',publication_authorized=False,records=len(rows),paired_predictions=len(result['models'])*len(rows),
               models=result['models'],empty_hypotheses=empty,
               checks=['Every private hypothesis re-scored locally','All reference and prediction hashes match',
                       'Every WER/CER numerator and denominator matches saved counts'],
               inputs_sha256={name:sha(a.evidence/name) for name in ['comparison.json','numeric-evidence.jsonl.gz']},
               manifest_sha256=result['manifest_sha256'],prediction_sha256=result['prediction_sha256'],
               script_sha256=sha(Path(__file__)))
    if a.upstream_root:
        provenance=json.loads((ROOT/'evaluation/standard_asr/vendor/provenance.json').read_text())
        for name,digest in provenance['upstream_sha256'].items():
            assert sha(a.upstream_root/name)==digest,name
        # Load upstream normalizers independently of the release's vendored package.
        init=a.upstream_root/'normalizer/__init__.py'
        spec=importlib.util.spec_from_file_location('upstream_text',init,
                         submodule_search_locations=[str(init.parent)])
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        namespace=dict(BasicMultilingualTextNormalizer=module.BasicMultilingualTextNormalizer,
                       re=re,num2words=num2words,FILLER_WORDS={},SequenceMatcher=SequenceMatcher)
        for filename,name in [('data_utils.py','MultilingualNormalizer'),
                              ('eval_utils.py','normalize_compound_pairs')]:
            tree=ast.parse((init.parent/filename).read_text())
            definition=next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
            exec(compile(ast.Module(body=[definition],type_ignores=[]),filename,'exec'),namespace)
        english=module.EnglishTextNormalizer()
        multilingual=namespace['MultilingualNormalizer'](remove_diacritics=False)
        for (split,model),(refs,hyps) in paired.items():
            lang=result['sets'][split]['language']
            normalize=english if lang=='en' else lambda text:multilingual(text,lang=lang)
            refs,hyps=list(map(normalize,refs)),list(map(normalize,hyps))
            if lang!='en':refs,hyps=namespace['normalize_compound_pairs'](refs,hyps)
            batch=batch_error_rate([tuple(r.split()) for r in refs],[tuple(h.split()) for h in hyps],merge_compounds=True)
            target=result['sets'][split]['models'][model]
            assert batch['total']==target['errors'] and batch['ref_len']==target['words']
            assert abs(100*batch['err_rate']-target['wer'])<1e-12
        audit['upstream_scoring_verified']=dict(revision=provenance['revision'],model_partition_pairs=len(paired),
            check='Unmodified upstream normalizer definitions and full-partition batch scoring reproduce all paired WERs.')
        audit['checks'].append('All model/partition WERs match independent upstream batch scoring')
    (a.evidence/'hypotheses-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':main()
