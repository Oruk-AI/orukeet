"""Paired warm native ASR timings on identical, predecoded fixture audio."""
import argparse
import array
import hashlib
import json
import platform
import os
import subprocess
import statistics
import time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--runtime',required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('audio',nargs='+',type=Path);a=p.parse_args()
from orukeet.nvidia import NvidiaRecognizer
from orukeet.audio import windows

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
fixtures=[]
for path in a.audio:
    chunks=list(windows(path))
    if len(chunks)!=1:raise ValueError('Benchmark fixtures must fit a single window')
    samples=array.array('f',chunks[0][1]);fixtures.append((path,samples))
results={};loads={}
# Reverse model order in the second block; preserve every timing sample.
for block,order in enumerate([['original','candidate'],['candidate','original']]):
    for label in order:
        model=getattr(a,label);started=time.perf_counter();engine=NvidiaRecognizer(a.runtime,str(model),a.device)
        loads.setdefault(label,[]).append(time.perf_counter()-started)
        try:
            for path,samples in fixtures:
                for _ in range(2):engine.transcribe(samples,'auto')
                key=label+':'+path.name
                record=results.setdefault(key,{'model':label,'audio':path.name,'audio_sha256':sha(path),
                    'duration_seconds':len(samples)/16000,'seconds':[],'transcripts':[]})
                for _ in range(10):
                    started=time.perf_counter();text=engine.transcribe(samples,'auto')['text'];elapsed=time.perf_counter()-started
                    record['seconds'].append(elapsed);record['transcripts'].append(text)
        finally:engine.close()
for r in results.values():
    r['median_seconds']=statistics.median(r['seconds']);r['rtfx']=r['duration_seconds']/r['median_seconds']
    r['deterministic']=len(set(r.pop('transcripts')))==1
report={'scope':'Two warmup calls then 10 timed calls per fixture/model/block, two blocks with reversed model order. Audio decode and model loading excluded from warm timing.',
    'platform':platform.platform(),'device':a.device,'runtime':a.runtime,'original_sha256':sha(a.original),
    'candidate_sha256':sha(a.candidate),'load_seconds':loads,'results':list(results.values()),
    'ratios_candidate_over_original':{path.name:results['candidate:'+path.name]['median_seconds']/results['original:'+path.name]['median_seconds'] for path,_ in fixtures}}
report['hardware']={'machine':platform.machine(),'logical_cpu_count':os.cpu_count()}
if platform.system()=='Darwin':
    report['hardware']['cpu']=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
elif Path('/proc/cpuinfo').exists():
    report['hardware']['cpu']=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor())
if a.device=='cuda':
    report['hardware']['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
runtime=Path(a.runtime)/'lib'
report['runtime_library_sha256']={p.name:sha(p) for p in sorted(runtime.iterdir()) if p.is_file() and ('.so' in p.name or p.suffix=='.dylib')}
report['deterministic']=all(r['deterministic'] for r in results.values())
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['ratios_candidate_over_original']))
