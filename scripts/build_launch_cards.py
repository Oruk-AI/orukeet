"""Render social cards from committed measurements; no external assets or fonts."""
from pathlib import Path
import json
from PIL import Image, ImageDraw
from matplotlib import font_manager
from PIL import ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'launch/assets'
INK='#172A34'; BLUE='#12617A'; PAPER='#F5F2E9'; MUTED='#42565E'

def font(size,bold=False):
    path=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='bold' if bold else 'normal'))
    return ImageFont.truetype(path,size)

def put(d,xy,text,size,color=INK,bold=False):
    d.text(xy,text,font=font(size,bold),fill=color,stroke_width=0)

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    data=json.loads((ROOT/'evidence/r3-promotion-20260908/scores.json').read_text())['summaries']['fleurs']['models']
    macro=f"{data['parakeet']['wer']:.2f}%  →  {data['orukeet']['wer']:.2f}%"
    for name,w,h in [('landscape',1600,900),('square',1200,1200)]:
        im=Image.new('RGB',(w,h),PAPER); d=ImageDraw.Draw(im)
        pad=80
        d.line((pad,54,w-pad,54),fill=INK,width=3)
        put(d,(pad,83),'Orukeet',112 if w==1600 else 104,bold=True)
        put(d,(pad,221),'Better ASR with fitted, frozen Gabor kernels',40 if w==1600 else 34)
        put(d,(pad,282),'ORUKEET r3 · 25 LANGUAGES',24,BLUE,True)
        y=333 if w==1600 else 379
        put(d,(pad,y),'FLEURS POOLED WER · LOWER IS BETTER',27,BLUE,True)
        put(d,(pad,y+62),macro,112 if w==1600 else 94,bold=True)
        put(d,(pad,y+203),'NVIDIA Parakeet → Orukeet r3',30 if w==1600 else 29)
        yy=y+290
        lines=['10.6% lower WER · 20,146 clips · 25 FLEURS languages.',
               'English accents/domains: 8.84% vs Parakeet 9.51%.',
               '5,120 English clips · matched NeMo decoding.',
               'All split scores and methods included.']
        for i,line in enumerate(lines): put(d,(pad,yy+i*36),line,27 if w==1600 else 29,MUTED)
        foot=h-89
        d.line((pad,foot-24,w-pad,foot-24),fill=INK,width=2)
        put(d,(pad,foot),'714 MB Q8 · code MIT · weights CC BY-SA 4.0',25 if w==1600 else 28)
        if w==1600: put(d,(1110,foot),'Oruk AI / 0.1.0rc1',25,BLUE)
        im.save(OUT/f'orukeet-card-{name}.png',optimize=True)
    meta={'status':'private draft; future publication needs approval','source':'evidence/r3-promotion-20260908/scores.json', 'model':'Orukeet r3',
          'alt_text':'Orukeet r3: 12,288 fitted, frozen Gabor kernels. Pooled WER falls from NVIDIA Parakeet’s 11.01% to Orukeet’s 9.85% across the 25-language, 20,146-clip FLEURS comparison, a 10.6% relative reduction. On 5,120 English accent/domain clips, WER is 8.84% for Orukeet versus 9.51% for Parakeet. Matched NeMo decoding; full split scores and methods included. Native Q8 is 714 MB.',
          'files':['orukeet-card-landscape.png','orukeet-card-square.png']}
    (OUT/'orukeet-cards.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__': build()
