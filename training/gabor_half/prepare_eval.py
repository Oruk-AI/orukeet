"""Freeze deterministic development membership before candidate inference."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
paths=sorted((a.root/'manifests/audited_20260905').glob('*_dev_selection.jsonl'))
paths+=sorted((a.root/'manifests/accent_extension_20260905').glob('*_dev*.jsonl'))
if not paths:raise RuntimeError('No development inputs')
records=[]
for path in paths:
    data=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # Fixed hash ordering is independent of any current model prediction.
    data.sort(key=lambda r:hashlib.sha256(('gabor-half-20260906:'+r['audio_filepath']).encode()).hexdigest())
    chosen=data[:100]
    dest=a.output/(path.stem+'.jsonl')
    dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen))
    records.append({'name':path.stem,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(chosen),
        'seconds':sum(r['duration'] for r in chosen),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
(a.output/'registry.json').write_text(json.dumps({'purpose':'Previously exposed development regression; not a new holdout',
    'selection':'first 100 records per source/language by SHA256(gabor-half-20260906:audio_filepath)',
    'primary':'equal CV/FLEURS corpus macro WER; languages equal within corpus',
    'acceptance':'candidate primary <= original; each language/corpus regression <= 0.5 percentage points; each accent source <= original + 1 point',
    'records':records},indent=2)+'\n')
print(json.dumps({'sets':len(records),'rows':sum(x['rows'] for x in records),'hours':sum(x['seconds'] for x in records)/3600,'keys':list(data[0])}))
