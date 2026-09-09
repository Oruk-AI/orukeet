"""Plot actual fitted taps and global-selection membership, without ASR claims."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--fits',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
z=np.load(a.fits);order=np.argsort(z['normalized_sse'],kind='stable');chosen=order[:12288]
assert len(z['original'])==24576 and int(z['selected'].sum())==12288
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
    'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
fig=plt.figure(figsize=(7,4.8),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[1,1.05])
for column,(label,idx) in enumerate(zip(['Best selected fit','Median selected fit','Cutoff selected fit'],[chosen[0],chosen[6143],chosen[-1]])):
    ax=fig.add_subplot(grid[0,column]);w=z['original'][idx];norm=np.linalg.norm(w)
    amp,mu,sigma,f,phase=z['params'][idx];x=np.linspace(-4,4,300)
    y=amp*np.exp(-.5*((x-mu)/sigma)**2)*np.cos(2*np.pi*f*(x-mu)+phase)
    ax.plot(np.arange(-4,5),w/norm,'o:',color='#333333',markersize=3,linewidth=.8,label='Original taps')
    ax.plot(x,y/norm,color='#168071',linewidth=1.4,label='Fitted Gabor')
    ax.set(title=label+'\n'+f'{100*np.sqrt(z["normalized_sse"][idx]):.2f}% relative RMS error',xlabel='Temporal tap',ylim=(-1.1,1.1),xticks=[-4,0,4])
    if column==0:ax.set_ylabel('Weight / original L2 norm')
    if column==2:ax.legend(loc='upper right',fontsize=7,frameon=False)
    ax.axhline(0,color='#bbbbbb',linewidth=.5,zorder=0)
ax=fig.add_subplot(grid[1,:]);counts=z['selected'].reshape(24,1024).sum(1)
ax.bar(np.arange(1,25),counts,color='#168071',width=.72)
ax.axhline(512,color='#555555',linewidth=.8,linestyle='--')
ax.set(xlabel='Conformer layer',ylabel='Replaced kernels / 1,024',xticks=[1,4,8,12,16,20,24],yticks=[0,256,512,768,1024],ylim=(0,1024),xlim=(.2,24.8),title='12,288 of 24,576 temporal kernels selected globally')
ax.text(12.5,915,'The dashed line marks 50% per layer; selection uses global fit rank.',ha='center',fontsize=8,color='#555555')
a.output.mkdir(parents=True,exist_ok=True)
for ext in ['pdf','png','svg']:fig.savefig(a.output/('gabor-fits.'+ext),dpi=300)
(a.output/'gabor-fits-caption.md').write_text('Original Orukeet temporal convolution taps and fitted real Gabor functions. Examples are selected deterministically at ranks 1, 6,144, and 12,288. Each example is normalized by its original kernel L2 norm. Relative RMS error equals the square root of normalized squared fitting error. The bottom panel shows the globally selected 12,288 kernels across 24 layers; a per-layer 50% quota was not imposed. These are weight-fit measurements, not transcription accuracy results.\n')
