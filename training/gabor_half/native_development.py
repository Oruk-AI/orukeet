"""Compare original and candidate Q8 exports through the same deployed audio path."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from compare import compare
from identity import sha
from reuse_native_reference import reuse_reference

p=argparse.ArgumentParser()
for key in ('original','candidate','lineage','evaluator','manifests','output'):
    p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--runtime',required=True)
p.add_argument('--device',choices=['cpu','metal','cuda','vulkan'],required=True)
p.add_argument('--reference-cache', type=Path)
a=p.parse_args()
lineage=json.loads(a.lineage.read_text())
if sha(a.candidate)!=lineage['gguf_sha256']:raise ValueError('Candidate lineage mismatch')
if sha(a.original)!='8967e04bd73fd8e88bccbfe97d0ec952beac73b1b3b3cba4b5c03316fcd8baa0':
    raise ValueError('Original Q8 identity mismatch')
if lineage.get('fitted_rows_exact_after_f16_rounding')!=12288:
    raise ValueError('Native Gabor row audit is required')
a.output.mkdir(parents=True,exist_ok=False)
import orukeet.audio
import orukeet.nvidia
protocol={'converter_commit':lineage['converter_commit'],'device':a.device,
          'runtime_library_sha256':{f.name:sha(f) for f in sorted((Path(a.runtime)/'lib').iterdir())
                                    if f.is_file() and ('.so' in f.name or f.suffix=='.dylib')},
          'decoder':'native TDT greedy defaults; automatic language; word offsets and punctuation enabled',
          'evaluation_script_sha256':sha(a.evaluator),
          'binding_sha256':sha(orukeet.nvidia.__file__), 'audio_path_sha256':sha(orukeet.audio.__file__)}
registry=json.loads((a.manifests/'registry.json').read_text())
for label in ('original','candidate'):
    dest=a.output/label
    if label == 'original' and a.reference_cache:
        reuse_reference(a.reference_cache, dest, protocol, registry, full=False)
    else:
        with (a.output/(label+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(a.evaluator),'--model',str(getattr(a,label)),
                '--runtime',a.runtime,'--device',a.device,'--manifests',str(a.manifests),
                '--output',str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True)
    metadata=json.loads((dest/'results.json').read_text())
    metadata['decoding']=protocol
    (dest/'results.json').write_text(json.dumps(metadata,indent=2)+'\n')
result=compare(a.output/'original',a.output/'candidate',registry)
result['registry_sha256']=sha(a.manifests/'registry.json')
result['protocol']=protocol
result['limitations']='Paired native deployment regression on the same exposed 5,100 development recordings; not new model-selection evidence.'
(a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='sets'}),flush=True)
