"""Recompute both surgery regression statistics from a pseudonymous count bundle."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'evaluation'))
from compare_oruk_export import compare as full_compare
from compare import compare as dev_compare
from identity import sha
from metric_qualification import qualify

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
execution=json.loads((a.directory/'verification.json').read_text()).get('execution','source')
if execution not in ('source','native'):raise ValueError('Unknown metric bundle execution')
receipt={'status':'pass','suites':{},'scope':'Stored-count statistics, not acoustic ASR inference.'}
for suite,compare in [('development',dev_compare),('larger',full_compare)]:
    root=a.directory/suite
    registry=json.loads((root/'registry.json').read_text())
    actual=compare(root/'reference',root/'candidate',registry)
    if suite == 'larger':
        actual = qualify(actual, development_status, development_status != 'pass')
        if execution == 'native':
            actual['native_accuracy_qualified'] = actual.pop('release_qualified')
    else:
        development_status = actual['status']
    if execution == 'native':
        actual.pop('interpretation' if suite == 'larger' else 'limitations')
    expected=json.loads((root/'expected.json').read_text())
    assert actual==expected, suite+' did not exactly reproduce'
    receipt['suites'][suite]={'all_fields_exact':True,'reference_sha256':actual.get('reference_model_sha256',actual.get('original_sha256')),
                            'candidate_sha256':actual.get('candidate_model_sha256',actual.get('candidate_sha256')),
                            'registry_sha256':sha(root/'registry.json'),'primary':actual['primary'],
                            'status':actual['status'],'failures':actual['failures']}
    if suite == 'larger':
        key = 'native_accuracy_qualified' if execution == 'native' else 'release_qualified'
        receipt['suites'][suite][key] = actual[key]
if execution == 'native':receipt['execution'] = 'native'
a.output.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
