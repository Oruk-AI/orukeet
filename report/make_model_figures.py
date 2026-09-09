"""Orukeet figures from exact fitted taps and paired recognition counts.

Final manuscript width: 5.5 inches. Each exported figure keeps that size.
No smoothing, resampling, omitted outliers, or inferred training trajectories.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report/assets'
BLUE, GRAY, GREEN = '#0F4D92', '#767676', '#42949E'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8.5,
    'axes.labelsize':9, 'axes.titlesize':10, 'xtick.labelsize':8, 'ytick.labelsize':8,
    'legend.fontsize':8, 'pdf.fonttype':42, 'svg.fonttype':'none',
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.7,
    'lines.linewidth':1.4, 'savefig.facecolor':'white'})
fit_path = ROOT / 'training/gabor_half/fits/fits.npz'
z = np.load(fit_path)
assert z['original'].shape == (24576,9) and z['selected'].sum() == 12288
order = np.argsort(z['normalized_sse'], kind='stable')
assert np.array_equal(np.sort(order[:12288]), np.flatnonzero(z['selected']))
selected = order[:12288]
error = 100 * np.sqrt(z['normalized_sse'])
result_path = ROOT / 'training/gabor_half/results/full-r15-0100.json'
r = json.loads(result_path.read_text())
assert r['candidate_model_sha256'] == '4295a6d820a40b99786331d1c7a6b6c328916c8329b23d39415b0649a5d42811'
provenance = {'model_sha256':r['candidate_model_sha256'],
    'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [fit_path,result_path,Path(__file__).resolve()]},
    'style':'academic-figures: final-size typography, restrained blue/gray/teal, vector export',
    'design_precedent':{'repository':'https://github.com/Nathan-Roll1/gabormer',
        'manuscripts':['interspeech/paper/gabormer_interspeech/gabormer_interspeech.tex',
                       'colm/paper/gabormer_unified/gabormer_unified.tex'],
        'adapted_ideas':['original/fitted kernel small multiples','fit distribution','layer-depth profile'],
        'data':'All plotted measurements are Orukeet data. No Gabormer values or artwork copied.'},
    'figures':{}}

def save(fig,name,caption,details):
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT / (name+'.'+ext), dpi=300)
    # Manuscript-size screen preview, useful alongside the vector PDF.
    fig.savefig(OUT/(name+'-preview.png'),dpi=150)
    (OUT/(name+'-caption.md')).write_text(caption+'\n')
    provenance['figures'][name] = {'size_inches':list(fig.get_size_inches()),
        'caption':caption, **details,
        'sha256':{ext:hashlib.sha256((OUT/(name+'.'+ext)).read_bytes()).hexdigest()
                  for ext in ['pdf','svg','png']}}
    plt.close(fig)

fig, axs = plt.subplots(1,4,figsize=(5.5,1.9),sharex=True,sharey=True)
fig.subplots_adjust(left=.095,right=.985,bottom=.25,top=.73,wspace=.15)
examples=[]
for ax,q,letter in zip(axs,[.125,.375,.625,.875],'ABCD'):
    rank=round(12287*q); i=int(selected[rank]); norm=np.linalg.norm(z['original'][i])
    ax.plot(range(-4,5),z['original'][i]/norm,'o-',color=GRAY,markersize=3,
            linewidth=1,label='Original taps',markerfacecolor='white')
    ax.plot(range(-4,5),z['fitted'][i]/norm,'x--',color=BLUE,markersize=3,
            linewidth=1.2,label='Frozen Gabor taps')
    ax.axhline(0,color='#dddddd',lw=.6,zorder=0)
    ax.set(xticks=[-4,0,4],yticks=[-1,0,1],ylim=(-1.08,1.08),xlabel='Tap index')
    ax.set_title(f'{letter}  Rank {rank+1:,}\n{error[i]:.2f}% RMS',loc='left',fontsize=9)
    examples.append({'rank':rank+1,'layer_zero_based':i//1024,'channel':i%1024,
                     'relative_rms_percent':float(error[i])})
axs[0].set_ylabel('Weight / original norm')
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=2,
           frameon=False,bbox_to_anchor=(.55,1.025),handlelength=2)
save(fig,'kernel-fits','Four exact kernel replacements. Examples are ranks 1,537, 4,609, 7,680 and 10,752 of the selected 12,288, nearest the 12.5th, 37.5th, 62.5th and 87.5th fit-error percentiles. Each pair is divided by the original kernel L2 norm. Markers are the nine stored taps; connecting lines guide the eye. RMS percentages are relative to the original norm.',{'examples':examples,'unit':'one temporal depthwise kernel','uncertainty':'Exact weights; no sampling interval.'})

fig, axs = plt.subplots(1,2,figsize=(5.5,2.15))
fig.subplots_adjust(left=.1,right=.98,bottom=.24,top=.82,wspace=.38)
ax=axs[0]; y=100*np.arange(1,len(order)+1)/len(order)
ax.plot(error[order],y,color=GRAY,lw=1.5)
ax.plot(error[selected],y[:12288],color=BLUE,lw=2)
cut=float(error[selected[-1]])
ax.scatter([cut],[50],color=BLUE,s=20,zorder=3)
ax.axhline(50,color='#bbbbbb',lw=.7,ls=':')
ax.annotate(f'50% selected\ncutoff {cut:.2f}%',xy=(cut,50),xytext=(29,19),
            fontsize=8,color=BLUE,arrowprops={'arrowstyle':'-','color':BLUE,'lw':.7})
ax.set(xlabel='Relative RMS fit error (%)',ylabel='Kernels at or below (%)',
       xlim=(0,max(100,float(error.max())*1.02)),ylim=(0,100),yticks=[0,50,100],xticks=[0,25,50,75,100])
ax.set_title('A  Global fit ranking',loc='left')
ax=axs[1]; counts=z['selected'].reshape(24,1024).sum(1)
ax.bar(np.arange(1,25),counts,color=BLUE,width=.72)
ax.axhline(512,color=GRAY,lw=.8,ls='--')
ax.set(xlabel='Encoder layer',ylabel='Frozen kernels / 1,024',xlim=(.2,24.8),
       ylim=(0,1024),xticks=[1,6,12,18,24],yticks=[0,512,1024])
ax.set_title('B  Selection by layer',loc='left')
save(fig,'selection-profile','Global Gabor selection. (A) Empirical cumulative distribution of relative RMS fit error for all 24,576 original kernels; blue marks the selected half. Every kernel is included. (B) The resulting allocation across all 24 encoder layers. The dashed 512 line denotes half a layer; selection uses a single global ranking. Counts range from 175 to 748 per layer.',{'selected_count':12288,'total_count':24576,'cutoff_percent':cut,'layer_counts':counts.tolist(),'uncertainty':'Complete kernel population; no sampling interval.'})

fig,ax=plt.subplots(figsize=(5.5,1.75))
fig.subplots_adjust(left=.26,right=.98,bottom=.34,top=.84)
for y,key,label in [(1,'primary','20 languages'),(0,'english','English, 7 corpora')]:
    d=r[key]; lo,hi=d['paired_delta_95ci']; x=d['delta_pp']
    ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt='o',color=BLUE,capsize=3,markersize=5,lw=1.6)
ax.axvline(0,color=GRAY,lw=.8,ls='--')
ax.set(yticks=[0,1],yticklabels=['English, 7 corpora','20 languages'],ylim=(-.55,1.55),
       xlim=(-.16,.115),xticks=[-.15,-.1,-.05,0,.05,.1],
       xlabel='WER change from adaptation baseline (percentage points)')
ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
ax.set_title('Recognition after recovery',loc='left')
save(fig,'recognition-deltas','Orukeet minus the pre-Gabor adaptation baseline under matched NeMo decoding. Points are macro WER differences; bars are 95% paired global-cluster bootstrap intervals (5,000 replicates, seed 20260905; 4,266 speaker or parallel-sentence clusters). Negative values mean lower WER. The primary endpoint uses 13,246 recordings and 20 fixed languages; English averages seven corpus WERs. Both systems were selected during the same adaptive campaign; intervals condition on those selections.',{'data':{k:r[k] for k in ['primary','english']},'independent_unit':r['bootstrap']['unit'],'replicates':5000})

# Complete per-language view: common axes retain the wide Lithuanian interval.
langs=[(k.split(':')[1],v) for k,v in r['metrics'].items() if k.startswith('language:')]
fig,ax=plt.subplots(figsize=(5.5,5.4));fig.subplots_adjust(left=.17,right=.97,bottom=.13,top=.92)
for y,(name,d) in enumerate(reversed(langs)):
    x=d['delta_pp'];lo,hi=d['paired_delta_95ci']
    ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt='o',color=BLUE,capsize=2,markersize=3.5,lw=1)
ax.set(yticks=range(20),yticklabels=[s.upper() for s,_ in reversed(langs)],
       xlabel='WER change from adaptation baseline (pp)',ylim=(-.7,19.7),xlim=(-2.4,.95))
ax.axvline(0,color=GRAY,lw=.8,ls='--');ax.spines['left'].set_visible(False)
ax.tick_params(axis='y',length=0);ax.set_title('All 20 languages in the primary endpoint',loc='left')
save(fig,'language-deltas','Per-language Orukeet WER change under matched NeMo decoding, with the same paired bootstrap as the macro comparison. All 20 primary languages and complete 95% intervals are shown, alphabetically. Languages share a scale. These marginal intervals are not adjusted for simultaneous inference.',{'data':dict(langs),'independent_unit':r['bootstrap']['unit']})

# A full layer atlas uses each layer's median selected row, not hand-picked waves.
fig,axs=plt.subplots(6,4,figsize=(7,8),sharex=True,sharey=True)
fig.subplots_adjust(left=.1,right=.98,bottom=.08,top=.93,wspace=.25,hspace=.52)
atlas=[]
for layer,ax in enumerate(axs.flat):
    candidates=selected[selected//1024==layer];i=int(candidates[(len(candidates)-1)//2]);norm=np.linalg.norm(z['original'][i])
    ax.plot(range(-4,5),z['original'][i]/norm,'o-',color=GRAY,ms=2.6,lw=1,mfc='white')
    ax.plot(range(-4,5),z['fitted'][i]/norm,'x--',color=BLUE,ms=2.6,lw=1)
    ax.axhline(0,color='#dddddd',lw=.5)
    ax.set_title(f'Layer {layer+1} · ch {i%1024}',loc='left',fontsize=9)
    ax.set(xticks=[-4,0,4],yticks=[-1,0,1],ylim=(-1.08,1.08))
    atlas.append({'layer':layer+1,'channel':i%1024,'error_percent':float(error[i])})
fig.suptitle('One median selected kernel from every encoder layer',fontsize=12,x=.1,ha='left')
fig.supxlabel('Tap index',fontsize=10);fig.supylabel('Weight / original norm',fontsize=10)
save(fig,'kernel-atlas','Layer atlas: the median selected fit-error row within each of the 24 layers. Original (gray circles) and frozen Gabor (blue crosses) taps share the original L2 normalization and common axes. All plotted samples come from the recorded fits; lines join discrete samples.',{'examples':atlas,'unit':'one selected kernel per layer','selection_rule':'lower median by global fit-error order within layer'})
(OUT/'model-figures-source.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps({'figures':list(provenance['figures']),'selected_count':12288,'cutoff_percent':cut}))
