"""Fit single real Gabors to temporal depthwise kernels, then select globally.

No waveform, label, or evaluation result is used for selection. All fits use
float64 and deterministic multistart bounded least squares (a local solver).
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import time
import numpy as np
from scipy.optimize import least_squares

PATTERN = re.compile(r"encoder\.layers\.(\d+)\.conv\.depthwise_conv\.weight$")
SOURCE_SHA = "313d615ca34c8ac3a183384e1f87e8274d748e5445af110013a335f02ae42e32"

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def basis(p, size=9):
    mu, log_sigma, frequency = p
    x = np.arange(size, dtype=np.float64) - (size-1)/2
    t = x-mu
    sigma = np.exp(log_sigma)
    envelope = np.exp(-0.5*(t/sigma)**2)
    angle = 2*np.pi*frequency*t
    d = envelope[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    derivatives = np.stack([
        t[:, None]/sigma**2*d + 2*np.pi*frequency*np.stack([d[:,1], -d[:,0]], axis=1),
        (t/sigma)[:,None]**2*d,
        2*np.pi*t[:,None]*np.stack([-d[:,1], d[:,0]], axis=1),
    ], axis=0)
    return d, derivatives

def projected(p, y, jac=False):
    d, deriv = basis(p, len(y))
    inv = np.linalg.pinv(d, rcond=1e-12)
    coeff = inv @ y
    residual = d @ coeff-y
    if not jac: return residual
    columns=[]
    for dd in deriv:
        z = dd @ coeff
        columns.append(z-d @ (inv @ z)-inv.T @ (dd.T @ residual))
    return np.stack(columns, axis=1)

def generate_bank(size=9):
    radius=(size-1)/2
    grid=np.array([(m,np.log(s),f) for m in np.linspace(-radius,radius,9)
                   for s in np.geomspace(.3,4*size,9)
                   for f in np.linspace(.0001,.4999,41)])
    ds=np.stack([basis(p,size)[0] for p in grid])
    # SVD, rather than an unpivoted QR, handles nearly rank-one sinusoids.
    u,s,_=np.linalg.svd(ds,full_matrices=False)
    u *= (s > s[:,:1]*1e-12)[:,None,:]
    return grid,u

def fit_one(task):
    y, starts = task
    norm=float(np.linalg.norm(y))
    if norm==0: return [0.,0.,0.,1.,0.], 0., 0
    yn=y/norm
    size=len(y); radius=(size-1)/2
    bounds=([-radius,np.log(.25),.000001],[radius,np.log(4*size),.499999])
    best=None; total_nfev=0
    for start in starts:
        initial=projected(start,yn)
        choices=[(float(initial@initial),start)]
        result=least_squares(projected,start,jac=lambda p,y:projected(p,y,True),
            args=(yn,),bounds=bounds,xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=160)
        total_nfev+=result.nfev
        choices.append((float(result.fun@result.fun),result.x))
        choice=min(choices,key=lambda q:q[0])
        if best is None or choice[0]<best[0]: best=choice
    error,p=best
    d,_=basis(p,size)
    a,b=np.linalg.lstsq(d,y,rcond=1e-12)[0]
    # a*cos(theta)+b*sin(theta) = A*cos(theta+phase).
    params=[float(np.hypot(a,b)),float(p[0]),float(np.exp(p[1])),float(p[2]),float(np.arctan2(-b,a))]
    return params,error,total_nfev

def reconstruct(params,size=9):
    p=np.asarray(params,dtype=np.float64)
    a,mu,sigma,f,phase=p.T
    x=np.arange(size,dtype=np.float64)-(size-1)/2
    t=x[None,:]-mu[:,None]
    return a[:,None]*np.exp(-.5*(t/sigma[:,None])**2)*np.cos(2*np.pi*f[:,None]*t+phase[:,None])

def choose_starts(y,grid,u):
    scores=np.einsum('gki,k->gi',u,y)**2
    scores=scores.sum(axis=1)
    indices=list(np.argsort(-scores,kind='stable')[:4])
    for lo,hi in zip([0,.125,.25,.375],[.125,.25,.375,.5]):
        subset=np.flatnonzero((grid[:,2]>=lo)&(grid[:,2]<hi))
        indices.append(int(subset[np.argmax(scores[subset])]))
    return grid[list(dict.fromkeys(indices))]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--limit',type=int,default=0,help='Debug only; cannot produce a full selection.')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    assert sha(args.source)==SOURCE_SHA,'Unexpected initialization checkpoint'
    import torch
    with tarfile.open(args.source) as tar:
        member=next(m for m in tar.getmembers() if m.name.endswith('model_weights.ckpt'))
        state=torch.load(io.BytesIO(tar.extractfile(member).read()),map_location='cpu',weights_only=True)
    inventory=[]; arrays=[]
    for name in sorted(state,key=lambda k:int(PATTERN.match(k)[1]) if PATTERN.match(k) else 100000):
        if not PATTERN.match(name):continue
        w=state[name].numpy().astype(np.float64)
        if w.shape!=(1024,1,9): raise ValueError((name,w.shape))
        arrays.append(w[:,0,:]); inventory.extend((name,i) for i in range(len(w)))
    if len(arrays)!=24: raise ValueError('Expected all 24 temporal convolution modules')
    del state
    kernels=np.concatenate(arrays)
    if args.limit: kernels=kernels[:args.limit];inventory=inventory[:args.limit]
    grid,u=generate_bank()
    started=time.monotonic()
    tasks=((y,choose_starts(y,grid,u)) for y in kernels)
    fitted=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,result in enumerate(pool.map(fit_one,tasks,chunksize=16)):
            fitted.append(result)
            if (i+1)%256==0: print('FIT',i+1,len(kernels),round(time.monotonic()-started,1),flush=True)
    params=np.array([x[0] for x in fitted]); values=reconstruct(params)
    norm2=(kernels*kernels).sum(1)
    errors=((values-kernels)**2).sum(1)/np.maximum(norm2,1e-30)
    order=np.argsort(errors,kind='stable'); selected=np.zeros(len(kernels),dtype=bool)
    if not args.limit: selected[order[:len(order)//2]]=True
    np.savez_compressed(args.output/'fits.npz', original=kernels.astype('float32'), fitted=values.astype('float32'),
                        params=params, normalized_sse=errors, selected=selected)
    records=[dict(name=n,channel=c,params=params[i].tolist(),normalized_sse=float(errors[i]),selected=bool(selected[i]))
             for i,(n,c) in enumerate(inventory)]
    report={'schema':'orukeet.gabor-half/1','source_sha256':SOURCE_SHA,'formula':'A exp(-0.5 ((t-mu)/sigma)^2) cos(2 pi f (t-mu)+phase)',
      'parameter_order':['amplitude','center','sigma','frequency_cycles_per_tap','phase'],
      'grid_taps':list(range(-4,5)),'eligible':'24 Conformer temporal depthwise Conv1d weights; global channel ranking',
      'fit_method':'float64 bounded multistart variable projection; best 4 grid starts plus best in each frequency quartile; no global-optimum guarantee',
      'bounds':{'center':[-4,4],'sigma':[.25,36],'frequency':[.000001,.499999]},
      'rank_metric':'sum((w-g)^2)/sum(w^2); stable layer/channel tie break','debug_limit':args.limit,
      'kernels':len(kernels),'selected':int(selected.sum()),'elapsed_seconds':time.monotonic()-started,
      'fit_archive_sha256':sha(args.output/'fits.npz'),'records':records}
    (args.output/'fits.json').write_text(json.dumps(report,indent=2)+'\n')
    summary={k:v for k,v in report.items() if k!='records'}
    if selected.any(): summary['selected_error_quantiles']=dict(zip(['min','median','p95','max'],np.quantile(errors[selected],[0,.5,.95,1]).tolist()))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
