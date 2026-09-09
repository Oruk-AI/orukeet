#!/usr/bin/env python3
"""Reuse the frozen decoder for a separately sealed all-language coverage pass."""
import argparse,io,json,shutil,sys
from collections import defaultdict
from pathlib import Path
import sqlite3
import pyarrow.parquet as pq
import soundfile as sf
import run as core


def materialize(bucket,rows,sources,scratch,out,history):
    receipt=out/'audio-receipts'/(bucket+'.jsonl');receipt.parent.mkdir(exist_ok=True)
    folder=scratch/'audio'/bucket;folder.mkdir(parents=True,exist_ok=True)
    if receipt.exists():
        prior=core.read_rows(receipt)
        if all(Path(r['audio_filepath']).is_file() and core.sha(r['audio_filepath'])==r['audio_file_sha256'] for r in prior):return prior
    downloads=scratch/'downloads'/bucket;downloads.mkdir(parents=True,exist_ok=True)
    source=sources[rows[0]['source']];assert source['format']=='parquet'
    db=sqlite3.connect('file:'+str(history)+'?mode=ro',uri=True);byfile=defaultdict(dict);result={}
    for row in rows:byfile[row['source_file']][row['source_index']]=row
    for filename,indices in sorted(byfile.items()):
        info=next(f for f in source['files'] if f['path']==filename);path=core.fetch(source,info,downloads);index=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16,columns=['audio']):
            for raw in batch.to_pylist():
                if index in indices:
                    row=indices[index];data,sr=sf.read(io.BytesIO(raw['audio']['bytes']),dtype='float32',always_2d=True)
                    declared=row['duration'];prepared=dict(row,duration=declared if declared is not None else len(data)/sr)
                    value=core.write_audio(prepared,data,sr,folder,db);value['source_declared_duration']=declared;value['duration_provenance']='decoded_source_audio' if declared is None else 'published_metadata'
                    result[row['uid']]=value
                index+=1
        path.unlink();print('COVERAGE_AUDIO_SHARD',bucket,filename,len(result),flush=True)
    db.close();assert set(result)=={r['uid'] for r in rows}
    result=[result[r['uid']] for r in rows];temp=Path(str(receipt)+'.tmp');temp.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in result));temp.replace(receipt)
    shutil.rmtree(downloads);print('COVERAGE_AUDIO_READY',bucket,len(result),flush=True);return result


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--experiment',type=Path,required=True);a,_=p.parse_known_args();seal=json.loads((a.experiment/'metadata/seal.json').read_text());assert seal['suite'] in ['language_coverage_extension','reference_quality_followup']
    if '--sets' not in sys.argv:sys.argv.extend(['--sets',*sorted(seal['sets'])])
    core.materialize=materialize
    core.main()
    path=a.experiment/'results/runtime.json';runtime=json.loads(path.read_text());runtime['coverage_wrapper_sha256']=core.sha(__file__);core.atomic_json(path,runtime)

if __name__=='__main__':main()
