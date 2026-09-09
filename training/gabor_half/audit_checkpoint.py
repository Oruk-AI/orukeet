"""Independent on-disk audit of all tensor changes and immutable Gabor taps."""
import argparse
import io
import json
from pathlib import Path
import tarfile
import numpy as np
import torch
from fit import sha,SOURCE_SHA,reconstruct

def load(path):
    with tarfile.open(path) as tar:
        member=next(m for m in tar.getmembers() if m.name.endswith('model_weights.ckpt'))
        state=torch.load(io.BytesIO(tar.extractfile(member).read()),map_location='cpu',weights_only=True)
        assets=sorted(sha_bytes(tar.extractfile(m).read()) for m in tar.getmembers() if m.isfile() and any(m.name.endswith(s) for s in ['.vocab','.model','vocab.txt']))
    return state,assets

def sha_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()

p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--fits',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert sha(a.original)==SOURCE_SHA
original,oa=load(a.original);candidate,ca=load(a.candidate)
assert oa==ca and original.keys()==candidate.keys()
fit=json.loads(a.fits.read_text());assert fit['source_sha256']==SOURCE_SHA
count=0;max_error=0
for name in sorted({r['name'] for r in fit['records']}):
    rows=sorted((r for r in fit['records'] if r['name']==name and r['selected']),key=lambda r:r['channel'])
    if not rows:continue
    indices=[r['channel'] for r in rows]
    expected=torch.from_numpy(reconstruct([r['params'] for r in rows]).astype('float32')).unsqueeze(1)
    actual=candidate[name][indices]
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    count+=len(indices)
assert count==12288
changed=[];finite=True
for name,value in candidate.items():
    if value.is_floating_point() and not torch.isfinite(value).all():raise ValueError('Nonfinite '+name)
    if not torch.equal(value,original[name]):changed.append(name)
result={'status':'pass','original_sha256':SOURCE_SHA,'candidate_sha256':sha(a.candidate),
 'tokenizer_assets_equal':True,'tensor_keys_equal':True,'all_tensors_finite':True,'frozen_gabor_rows_exact':count,
 'changed_tensor_count':len(changed),'changed_tensors':changed,
 'changed_groups':{group:sum(n.startswith(group) for n in changed) for group in ['encoder.layers.0.','encoder.layers.23.','encoder.pre_encode.','decoder.','joint.']}}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='changed_tensors'}))
