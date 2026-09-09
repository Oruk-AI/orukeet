"""Rebuild the selected full partitions as a finite, verified training cache."""
import argparse,hashlib,json,os,shutil,sys
from pathlib import Path
import numpy as np
import soundfile as sf
import soxr


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.root/'unseen_20260907/code'))
    import run as core
    import run_coverage as coverage
    plan=json.loads(a.plan.read_text());a.output.mkdir(parents=True,exist_ok=True);a.scratch.mkdir(parents=True,exist_ok=True)
    receipts={};original=core.materialize;all_manifests=[]
    def write_audio(row,data,sr,out,db):
        data=np.asarray(data,dtype=np.float32)
        if data.ndim==2:data=data.mean(axis=1)
        assert data.ndim==1 and len(data) and np.isfinite(data).all()
        if sr!=16000:data=soxr.resample(data,sr,16000,quality='HQ')
        data=np.ascontiguousarray(data,dtype='<f4');digest=hashlib.sha256(data.tobytes()).hexdigest()
        expected=receipts[row['uid']];assert digest==expected['float32_pcm_sha256'],row['uid']
        path=out/(hashlib.sha256(row['uid'].encode()).hexdigest()+'.flac')
        sf.write(path,data,16000,subtype='PCM_16')
        decoded,rate=sf.read(path,dtype='float32');assert rate==16000 and len(decoded)==len(data)
        return dict(row,audio_filepath=str(path),actual_duration=len(data)/16000,
                    benchmark_pcm_sha256=digest,training_pcm_sha256=hashlib.sha256(decoded.astype('<f4').tobytes()).hexdigest(),
                    audio_file_sha256=core.sha(path),known_pcm_overlap=expected['known_pcm_overlap'],
                    declared_duration_mismatch=expected.get('declared_duration_mismatch',False),
                    clipped_samples=int(np.count_nonzero((data>32767/32768)|(data< -1))),
                    duration_difference_seconds=len(data)/16000-row['duration'])
    core.write_audio=write_audio
    # Largest source first bounds the scratch peak before the rest of the cache accumulates.
    ordered=sorted(plan['sets'],key=lambda s:-s['hours'])
    for selection in ordered:
        key=selection['split'];phase=a.root/'unseen_20260907'/selection['experiment_subdirectory']
        seal=json.loads((phase/'metadata/seal.json').read_text())
        assert core.sha(phase/'metadata/seal.json')==plan['benchmark_inputs'][selection['phase']]['protocol_sha256']
        source=Path(seal['sets'][key]['path']);assert core.sha(source)==seal['sets'][key]['sha256']
        rows=core.read_rows(source);assert len(rows)==selection['rows']
        receipts={r['uid']:r for r in core.read_rows(phase/'audio-receipts'/(key+'.jsonl'))}
        sources=json.loads((phase/'metadata/sources.json').read_text())
        materialize=original if selection['phase']=='primary' else coverage.materialize
        ready=materialize(key,rows,sources,a.scratch,a.output,phase/'history/history.sqlite')
        assert len(ready)==len(rows) and {r['uid'] for r in ready}=={r['uid'] for r in rows}
        for row in ready:
            row['duration']=row['actual_duration'];row['split']=key
            row['alignment_status']=selection['alignment_status']
            row['previous_role']='evaluation';row['current_role']='training'
        manifest=a.output/(key+'.jsonl')
        manifest.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in ready))
        all_manifests.append({'split':key,'path':str(manifest),'sha256':core.sha(manifest),'rows':len(ready),
                              'hours':sum(r['duration'] for r in ready)/3600,
                              'flac_bytes':sum(Path(r['audio_filepath']).stat().st_size for r in ready),
                              'clipped_samples':sum(r['clipped_samples'] for r in ready),
                              'alignment_status':selection['alignment_status']})
        core.atomic_json(a.output/'preparation-progress.json',{'complete':False,'manifests':all_manifests})
        print('TRAINING_AUDIO_READY',key,len(ready),all_manifests[-1]['flac_bytes'],flush=True)
    result={'complete':True,'plan_sha256':core.sha(a.plan),'script_sha256':core.sha(__file__),
            'rows':sum(x['rows'] for x in all_manifests),'hours':sum(x['hours'] for x in all_manifests),
            'manifests':all_manifests,'audio_format':'FLAC PCM16, mono 16000 Hz',
            'original_float_pcm_verified_for_every_row':True,'publication_authorized':False}
    assert result['rows']==plan['selected_rows']
    core.atomic_json(a.output/'prepared.json',result);print('ALL_TRAINING_AUDIO_READY',result['rows'],flush=True)


if __name__=='__main__':main()
