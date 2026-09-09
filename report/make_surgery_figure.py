"""Two measured panels at the paper's final 5.5-inch insertion width."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
path = root / 'training/gabor_half/fits/fits.npz'
z = np.load(path)
assert len(z['original']) == 24576 and z['selected'].sum() == 12288
order = np.argsort(z['normalized_sse'], kind='stable')[:12288]
idx = order[6143]
w = z['original'][idx]
norm = np.linalg.norm(w)
amp, mu, sigma, f, phase = z['params'][idx]
x = np.linspace(-4, 4, 300)
y = amp * np.exp(-.5 * ((x - mu) / sigma)**2) * np.cos(2 * np.pi * f * (x - mu) + phase)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
    'axes.titlesize':8.5, 'axes.labelsize':8, 'pdf.fonttype':42, 'svg.fonttype':'none',
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.6})
fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.52), layout='constrained',
                         gridspec_kw={'width_ratios':[1,1.25]})
ax = axes[0]
ax.plot(np.arange(-4,5),w/norm,'o:',color='#333333',markersize=2.5,linewidth=.7,label='Original taps')
ax.plot(x,y/norm,color='#168071',linewidth=1.25,label='Fitted Gabor')
ax.set(title='A  Median selected fit (6.32% RMS error)', xlabel='Temporal tap',
       ylabel='Weight / original norm', xticks=[-4,0,4], yticks=[0,.5,1], ylim=(-.3,1.08))
ax.legend(frameon=False,fontsize=6.8,loc='upper right',handlelength=1.1)
ax = axes[1]
ax.bar(np.arange(1,25),z['selected'].reshape(24,1024).sum(1),color='#168071',width=.7)
ax.axhline(512,color='#555555',linestyle='--',linewidth=.7)
ax.set(title='B  Global selection varies by layer', xlabel='Conformer layer',
       ylabel='Selected / 1,024',xticks=[1,6,12,18,24],yticks=[0,512,1024],ylim=(0,1024),xlim=(.2,24.8))
out=root/'report/assets'
for ext in ('pdf','svg','png'): fig.savefig(out/('surgery.'+ext),dpi=300)
(out/'surgery-source.json').write_text(json.dumps({'source':str(path.relative_to(root)),
    'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'median_selected_rank':6144,
    'median_original_flat_index':int(idx),'selected_count':12288,
    'caption':'Weight-fit measurements. The left example is rank 6,144 of the globally selected rows, normalized by original L2 norm. The right shows all layer counts; dashed line is 512, not a quota. No recognition accuracy is inferred.'},indent=2)+'\n')
