"""Deterministic reference/prediction normalization and edit counts."""
import re,unicodedata
from rapidfuzz.distance import Levenshtein
from whisper_normalizer.english import EnglishTextNormalizer
CHAR_MAP=str.maketrans({'ς':'σ','ё':'е','Ё':'Е','ѝ':'и','ţ':'ț','ş':'ș','Ţ':'Ț','Ş':'Ș'})
PUNCT=re.compile(r'''["'“”„‘’«»()\[\]{}.,!?;:¡¿…\-–—/\\|*+=<>@#$%^&_~`]''')
ENGLISH=EnglishTextNormalizer()


def normalize(text):
    return ' '.join(PUNCT.sub(' ',unicodedata.normalize('NFC',text).translate(CHAR_MAP).lower()).split())


def counts(reference,prediction,normalizer=normalize):
    ref=normalizer(reference);hyp=normalizer(prediction);rw=ref.split();hw=hyp.split()
    edits=Levenshtein.editops(rw,hw);ops={'replace':0,'delete':0,'insert':0}
    for tag,_,_ in edits:ops[tag]+=1
    assert sum(ops.values())==Levenshtein.distance(rw,hw)
    return {'words':len(rw),'errors':len(edits),'substitutions':ops['replace'],'deletions':ops['delete'],'insertions':ops['insert'],'chars':len(ref),'char_errors':Levenshtein.distance(ref,hyp),'utterance_error':int(bool(edits))}
