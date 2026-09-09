"""Diagnostic: retain learned parameters and restore original BN state buffers."""
import argparse
import io
import json
from pathlib import Path
import tarfile
import torch
from identity import sha,SOURCE_SHA

p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert sha(a.original)==SOURCE_SHA and not a.output.exists()

def state(path):
    with tarfile.open(path) as tar:
        member=next(m for m in tar.getmembers() if m.name.endswith('model_weights.ckpt'))
        return torch.load(io.BytesIO(tar.extractfile(member).read()),map_location='cpu',weights_only=True)
original=state(a.original);candidate=state(a.candidate);changed=[];statistics=[]
for name in candidate:
    if name.endswith(('running_mean','running_var','num_batches_tracked')):
        if name.endswith('running_var'):
            ratios=candidate[name]/original[name].clamp_min(1e-12)
            statistics.append({'tensor':name,'variance_ratio_median':float(ratios.median()),'variance_ratio_min':float(ratios.min()),'variance_ratio_max':float(ratios.max())})
        candidate[name]=original[name].clone();changed.append(name)
assert len(changed)==72
buffer=io.BytesIO();torch.save(candidate,buffer);data=buffer.getvalue()
with tarfile.open(a.candidate) as source,tarfile.open(a.output,'w') as dest:
    for member in source.getmembers():
        if member.name.endswith('model_weights.ckpt'):
            member.size=len(data);dest.addfile(member,io.BytesIO(data))
        else:dest.addfile(member,source.extractfile(member) if member.isfile() else None)
receipt={'source_sha256':SOURCE_SHA,'trained_checkpoint_sha256':sha(a.candidate),'output_sha256':sha(a.output),
    'modified_buffers':changed,'learned_parameters_retained':True,'gabor_taps_retained':True,
    'diagnostic':'Restore BatchNorm running state only; no trainable weights are reverted. Not a retraining run.',
    'trained_vs_original_variance':statistics}
a.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ['modified_buffers','trained_vs_original_variance']}),flush=True)
