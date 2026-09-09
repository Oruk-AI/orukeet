import argparse
import json
from pathlib import Path
from fit import sha, SOURCE_SHA
from frozen import install, verify, materialize

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--fits',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert sha(a.source)==SOURCE_SHA
assert not a.output.exists()
from nemo.collections.asr.models import ASRModel
model=ASRModel.restore_from(str(a.source),map_location='cpu')
original_params=sum(p.numel() for p in model.parameters())
receipts=install(model,a.fits);verify(model,receipts)
trainable=sum(p.numel() for p in model.parameters())
assert original_params-trainable==12288*9
materialize(model);model.save_to(str(a.output))
(a.output.with_suffix('.json')).write_text(json.dumps({'source_sha256':SOURCE_SHA,'candidate_sha256':sha(a.output),
    'fits_sha256':sha(a.fits),'frozen_rows':receipts,'original_parameters':original_params,'trainable_parameters':trainable},indent=2)+'\n')
print('EXPORT_COMPLETE',a.output,flush=True)
