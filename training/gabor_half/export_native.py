"""Candidate-only native export. Never changes canonical release artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from identity import sha
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
p.add_argument('--fits',type=Path,required=True)
p.add_argument('--parent-audit',type=Path,action='append',default=[],
               help='Freeze audits from the direct parent back to the original fit audit, in order')
p.add_argument('--converter',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--outtype',choices=['q8_0','f16','q8_0_t16'],default='q8_0')
p.add_argument('--baseline-q8', type=Path)
a=p.parse_args();audit=json.loads(a.audit.read_text());source=sha(a.candidate)
assert audit['candidate_sha256']==source and audit['status']=='pass' and audit['frozen_gabor_rows_exact']==12288
revision=subprocess.check_output(['git','-c','safe.directory='+str(a.converter),'-C',str(a.converter),'rev-parse','HEAD'],text=True).strip()
assert revision=='4f9676226f667d14608487df744f375db87127f8'
import librosa
import numpy as np
from gguf import GGUFReader
from fit import reconstruct
a.output.mkdir(parents=True,exist_ok=False);dest=a.output/('orukeet-gabor-half.'+a.outtype+'.gguf')
mixed_audit = None
if a.outtype == 'q8_0_t16':
    if not a.baseline_q8:
        raise ValueError('Mixed export requires its audited Q8 baseline')
    baseline_lineage = json.loads((a.baseline_q8.parent / 'lineage.json').read_text())
    assert baseline_lineage['candidate_sha256'] == source
    assert baseline_lineage['export_precision'] == 'q8_0'
    assert baseline_lineage['gguf_sha256'] == sha(a.baseline_q8)
    from transducer_f16 import convert, audit as audit_mixed
    module = convert(a.candidate, dest, a.converter)
    mixed_audit = audit_mixed(a.candidate, a.baseline_q8, dest, module)
else:
    subprocess.run([sys.executable,str(a.converter/'convert_model.py'),str(a.candidate),'--outfile',str(dest),'--outtype',a.outtype],check=True)
reader=GGUFReader(str(dest));fb=next(t for t in reader.tensors if t.name=='preprocessor.fb')
assert fb.data.size==128*257
fits=json.loads(a.fits.read_text());selected=[r for r in fits['records'] if r['selected']]
assert len(selected)==12288
if 'original_sha256' in audit:
    assert fits['source_sha256']==audit['original_sha256']
else:
    assert a.parent_audit, 'A continued checkpoint requires its parent freeze audit chain'
    parent_audit=audit
    for path in a.parent_audit:
        assert 'original_sha256' not in parent_audit, 'Unexpected audit after the original fit audit'
        assert parent_audit['gabor_values_match_parent_and_original_functions'] is True
        assert parent_audit['frozen_coefficients_exact']==110592
        previous=json.loads(path.read_text())
        assert previous['status']=='pass' and previous['frozen_gabor_rows_exact']==12288
        assert previous['candidate_sha256']==parent_audit['parent_sha256']
        parent_audit=previous
    assert fits['source_sha256']==parent_audit['original_sha256']
    assert audit['gabor_values_match_parent_and_original_functions'] is True
    assert audit['frozen_coefficients_exact']==110592
tensors={t.name:t for t in reader.tensors}
checked=0
for name in sorted({r['name'] for r in selected}):
    records=sorted((r for r in selected if r['name']==name),key=lambda r:r['channel'])
    tensor=tensors[name]
    assert tensor.data.dtype==np.float16 and tuple(tensor.shape)==(9,1,1024)
    actual=tensor.data.reshape(1024,1,9)[[r['channel'] for r in records],0,:]
    # The pinned deployment converter stores depthwise taps as F16. Compare
    # the exact same FP64 -> training FP32 -> deployment F16 rounding path.
    expected=reconstruct([r['params'] for r in records],9).astype(np.float32).astype(np.float16)
    assert np.array_equal(actual,expected),name
    checked+=len(records)
receipt={'candidate_sha256':source,'gguf_sha256':sha(dest),'gguf_bytes':dest.stat().st_size,'converter_commit':revision,
 'librosa_version':librosa.__version__,'mel_filterbank_sha256':hashlib.sha256(fb.data.tobytes()).hexdigest(),
 'fits_sha256':sha(a.fits),'fitted_rows_exact_after_f16_rounding':checked,'depthwise_storage_dtype':'float16',
 'export_precision':a.outtype,
 'source_audit_sha256':sha(a.audit),
 'scope':'Exact-source native export; format-specific ASR measurements are recorded separately'}
if a.parent_audit:
    receipt['parent_audit_sha256']=sha(a.parent_audit[0])
    receipt['ancestor_audits_sha256']=[sha(path) for path in a.parent_audit]
if mixed_audit is not None:
    receipt['transducer_precision_audit'] = mixed_audit
    receipt['mixed_export_code_sha256'] = sha(Path(__file__).with_name('transducer_f16.py'))
(a.output/'lineage.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
