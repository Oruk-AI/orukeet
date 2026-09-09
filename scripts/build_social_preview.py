"""Build the 1280×640 repository preview from the release's existing card style."""
import hashlib
import json
from PIL import Image, ImageDraw
from build_launch_cards import OUT, INK, PAPER, BLUE, MUTED, put

def main():
    im=Image.new('RGB',(1280,640),PAPER);d=ImageDraw.Draw(im)
    d.line((70,54,1210,54),fill=INK,width=3)
    put(d,(70,90),'Orukeet',108,bold=True)
    put(d,(72,236),'Better speech recognition. Same 627M parameters.',35)
    put(d,(72,296),'FLEURS · 25 LANGUAGES · POOLED WER · 10.6% LOWER',24,BLUE,True)
    put(d,(72,330),'Orukeet 9.85%  vs  Parakeet 11.01%',38,bold=True)
    put(d,(72,393),'ENGLISH ACCENTS/DOMAINS · WER · 7.0% LOWER',24,BLUE,True)
    put(d,(72,427),'Orukeet 8.84%  vs  Parakeet 9.51%',38,bold=True)
    put(d,(72,483),'20,146 FLEURS clips · 5,120 accent/domain clips · r3',22,MUTED)
    d.line((70,526,1210,526),fill=INK,width=2)
    put(d,(72,553),'Code MIT · weights CC BY-SA 4.0',25)
    put(d,(915,553),'0.1.0rc1',25,BLUE)
    file=OUT/'orukeet-social-preview.png';im.save(file,optimize=True)
    meta={'status':'private-prepared','file':str(file.relative_to(OUT.parents[1])),
          'dimensions':[1280,640],'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
          'model':'Orukeet r3',
          'alt_text':'Orukeet r3 and Parakeet have 627 million parameters. Across the 25-language, 20,146-clip FLEURS comparison, pooled WER is 9.85% for Orukeet versus 11.01% for Parakeet (10.6% relative reduction). On 5,120 English clips, WER is 8.84% for Orukeet versus 9.51% for Parakeet (7.0% relative reduction). Code MIT; weights CC BY-SA 4.0.',
          'source':'https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview',
          'benchmark_source':'evidence/r3-promotion-20260908/scores.json',
          'benchmark_source_sha256':hashlib.sha256((OUT.parents[1]/'evidence/r3-promotion-20260908/scores.json').read_bytes()).hexdigest(),
          'interpretation':'Pooled WER under matched NeMo decoding; relative improvements computed from integer edit counts.'}
    assert meta['bytes']<1_000_000
    (OUT/'orukeet-social-preview.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
