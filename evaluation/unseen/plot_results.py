#!/usr/bin/env python3
"""Publication-size paired WER comparisons from the completed experiment."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,MaxNLocator
import numpy as np

TEAL='#168071'
EURO={'bg':'Bulgarian','de':'German','el':'Greek','en':'English','et':'Estonian','fi':'Finnish','fr':'French','hr':'Croatian','it':'Italian','lt':'Lithuanian','lv':'Latvian','mt':'Maltese','pt':'Portuguese','sk':'Slovak','sl':'Slovenian','uk':'Ukrainian'}
ACCENTS={'chn':'Chinese accent','ind':'Indian accent','jpn':'Japanese accent','phl':'Filipino accent','sct':'Scottish accent','sgp':'Singaporean accent'}
DOMAINS={'agr':'Agriculture','ait':'Artificial intelligence','art':'Art','bio':'Biology','ecm':'Economics','eng':'Engineering','ent':'Entertainment','fin':'Finance','hum':'Humanities','law':'Law','med':'Medicine','mil':'Military'}


def limits(keys,data):
    values=[0.]
    for k in keys:
        r=data[k];values.append(r['wer_delta_pp']);values.extend(r['uncertainty'].get('wer_delta_pp_95ci',[]))
    lo=min(values);hi=max(values);span=max(hi-lo,.2)
    return lo-.08*span,hi+.12*span


def plot(names,data,path,title,scoring,xlim):
    n=len(names);height=.27*n+1.50
    fig,ax=plt.subplots(figsize=(7,height))
    fig.subplots_adjust(left=.24,right=.69,bottom=.65/height,top=1-.75/height)
    ys=np.arange(n);ax.set_ylim(n-.5,-.5);ax.set_yticks(ys,list(names.values()))
    ax.axvline(0,color='#777777',linewidth=.7,zorder=0)
    ax.set_xlim(*xlim);ax.xaxis.set_major_locator(MaxNLocator(5));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:'0' if abs(x)<1e-9 else f'{x:+.1f}'))
    ax.set_xlabel('WER difference (Orukeet − Parakeet), pp',labelpad=7)
    ax.tick_params(axis='y',length=0,pad=8);ax.grid(axis='x',color='#e9e9e9',linewidth=.5)
    for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
    for y,(key,label) in enumerate(names.items()):
        r=data[key];ci=r['uncertainty'].get('wer_delta_pp_95ci');point=r['wer_delta_pp']
        if ci:ax.plot(ci,[y,y],color=TEAL,linewidth=1.15,zorder=2)
        ax.plot(point,y,'o',color=TEAL,markersize=4.5,markerfacecolor=TEAL if ci else 'white',markeredgewidth=1,zorder=3)
        groups='—' if r['clusters'] is None else f"{r['clusters']:,}"
        for x,value,color in [(1.09,f"{r['parakeet']['wer']:.2f}",'#555555'),(1.32,f"{r['orukeet']['wer']:.2f}",TEAL),(1.58,groups,'#555555')]:
            ax.text(x,y,value,transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=8.5,color=color,clip_on=False)
    for x,label in [(1.09,'Parakeet'),(1.32,'Orukeet'),(1.58,'Groups')]:
        ax.text(x,1.025,label,transform=ax.transAxes,ha='center',va='bottom',fontsize=8,color=TEAL if label=='Orukeet' else '#444444',clip_on=False)
    fig.text(.015,1-.08/height,title,ha='left',va='top',fontsize=11,fontweight='bold')
    fig.text(.015,1-.30/height,scoring+' · WER (%) · 95% paired cluster-bootstrap intervals',ha='left',va='top',fontsize=8,color='#555555')
    fig.text(.015,.06/height,'Negative differences indicate lower Orukeet WER. Open point: interval unavailable.',fontsize=7.5,color='#555555',va='bottom')
    for ext in ['pdf','png','svg']:fig.savefig(path.with_suffix('.'+ext),dpi=300)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--comparison',type=Path,required=True);p.add_argument('--coverage',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.comparison.read_text());coverage=json.loads(a.coverage.read_text()) if a.coverage else None
    if d['status']!='complete':raise SystemExit('Publication figures require all sealed splits to be complete')
    if coverage is not None and coverage['status']!='complete':raise SystemExit('Coverage figures require all sealed splits to be complete')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','xtick.labelsize':8})
    a.output.mkdir(parents=True,exist_ok=True)
    euro={'eurospeech_'+k:v for k,v in EURO.items()};accent={'gigaspeechbench_'+k+'_en':v for k,v in ACCENTS.items()};accent['monsoon_en_in']='Monsoon · Indian English';domain={'gigaspeechbench_'+k+'_en':v for k,v in DOMAINS.items()}
    english=d['english_standard_sets'];english_limits=limits(list(accent)+list(domain),english)
    plot(euro,d['sets'],a.output/'unseen-eurospeech','EuroSpeech published-reference scores','Legacy scoring · reference offsets in Greek and Italian',limits(euro,d['sets']))
    plot(accent,english,a.output/'unseen-english-accents','English accents and spontaneous conversation','GigaSpeechBench + Monsoon · standard English scoring',english_limits)
    plot(domain,english,a.output/'unseen-english-domains','English speech across twelve specialist domains','GigaSpeechBench · standard English scoring',english_limits)
    names={'voxpopuli_cs':'Czech · VoxPopuli','nst_da_da':'Danish · NST','voxpopuli_nl':'Dutch · VoxPopuli','voxpopuli_hu':'Hungarian · VoxPopuli','voxpopuli_pl':'Polish · VoxPopuli','voxpopuli_ro':'Romanian · VoxPopuli','golos_crowd_ru':'Russian · Golos crowd','golos_farfield_ru':'Russian · Golos far field','voxpopuli_es':'Spanish · VoxPopuli','nst_sv_sv':'Swedish · NST mirror'}
    if coverage is not None:plot(names,coverage['sets'],a.output/'unseen-language-coverage','Supplementary coverage of nine languages','Supplementary test/validation · legacy scoring',limits(names,coverage['sets']))
    caption='Orukeet R15-0100 minus stock NVIDIA Parakeet TDT 0.6B v3 in word-error-rate percentage points; negative values favor Orukeet. Points use every segment of the sealed official test splits. Horizontal intervals use 10,000 paired cluster-bootstrap replicates, resampling source recording sessions for EuroSpeech, source recordings for GigaSpeechBench, and speakers for Monsoon. The groups column gives the number of resampling units; it does not assert that all groups contain different speakers. The French partition has one recording session and is plotted without an interval. English plots use whisper-normalizer 0.1.12; EuroSpeech uses the predeclared legacy-compatible normalizer. The report provides both normalizations and the exact-history-disjoint sensitivity analysis. Per-split intervals are unadjusted for multiple comparisons.\n'
    if coverage is not None:caption+='The supplementary coverage plot uses its own pre-inference registry and legacy normalization. Missing or incomplete speaker/session metadata yield an open point and no interval. Danish NST has 56 source speakers and Dutch VoxPopuli 47. Reference-alignment defects were observed in the Greek and Italian EuroSpeech material; the locked published-reference scores are retained, without treating them as headline accuracy evidence.\n'
    (a.output/'unseen-benchmark-captions.md').write_text(caption)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt={'comparison_sha256':digest(a.comparison),'coverage_comparison_sha256':digest(a.coverage) if a.coverage else None,'script_sha256':digest(Path(__file__)),'matplotlib':matplotlib.__version__,'files':{p.name:digest(p) for p in a.output.glob('unseen-*') if p.name!='unseen-figure-receipt.json'},'visual_review':'pending'}
    (a.output/'unseen-figure-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
