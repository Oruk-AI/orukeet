import json
from pathlib import Path
import pytest
from compare import compare

def fixture(tmp_path):
    registry={'records':[]};a={'normalizer':'same','decoding':{},'model_sha256':'a','sets':{}}
    for name in ['cv_en_dev','fleurs_en_dev','english_dialects_dev']:
        registry['records'].append({'name':name,'sha256':name,'rows':1})
        a['sets'][name]={'manifest_sha256':name}
    for label in ['reference','candidate']:
        d=tmp_path/label;d.mkdir()
        (d/'results.json').write_text(json.dumps(a))
        for name in a['sets']:(d/(name+'_hypotheses.jsonl')).write_text(json.dumps({'audio_filepath':name,'text':'a b','words':2,'errors':0})+'\n')
    return registry,tmp_path/'reference',tmp_path/'candidate'

def test_equal_pass_and_missing_fails(tmp_path):
    registry,a,b=fixture(tmp_path)
    assert compare(a,b,registry)['status']=='pass'
    r=json.loads((b/'results.json').read_text());del r['sets']['cv_en_dev'];(b/'results.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Incomplete'):compare(a,b,registry)

def test_regression_fails(tmp_path):
    registry,a,b=fixture(tmp_path)
    p=b/'cv_en_dev_hypotheses.jsonl';r=json.loads(p.read_text());r['errors']=1;p.write_text(json.dumps(r)+'\n')
    result=compare(a,b,registry)
    assert result['status']=='fail' and len(result['failures'])==2
