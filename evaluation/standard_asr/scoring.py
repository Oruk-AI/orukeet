"""Pinned English/multilingual normalization and compound-aware WER."""
from kaldialign import batch_error_rate
from rapidfuzz.distance import Levenshtein
from vendor.normalizer import EnglishTextNormalizer
from vendor.multilingual import MultilingualNormalizer, normalize_compound_pairs

ENGLISH = EnglishTextNormalizer()
MULTILINGUAL = MultilingualNormalizer(remove_diacritics=False)


def normalized_pair(reference, prediction, language):
    normalizer = ENGLISH if language == 'en' else lambda text: MULTILINGUAL(text, lang=language)
    return normalizer(reference), normalizer(prediction)


def counts(reference, prediction, language):
    ref, hyp = normalized_pair(reference, prediction, language)
    # CER uses the normalized strings before pair-dependent word-boundary changes.
    chars, char_errors = len(ref), Levenshtein.distance(ref, hyp)
    if language != 'en':
        refs, hyps = normalize_compound_pairs([ref], [hyp])
        ref, hyp = refs[0], hyps[0]
    result = batch_error_rate([tuple(ref.split())], [tuple(hyp.split())], merge_compounds=True)
    return dict(words=len(ref.split()), errors=result['ins'] + result['del'] + result['sub'],
                substitutions=result['sub'], deletions=result['del'], insertions=result['ins'],
                chars=chars, char_errors=char_errors,
                utterance_error=int(bool(result['ins'] + result['del'] + result['sub'])))
