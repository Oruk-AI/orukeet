import importlib.util
import os
from pathlib import Path
import sys

from realign import mapped_normalize

metric_dir = Path(os.environ.get('ORUKEET_METRIC_CODE', Path(__file__).resolve().parents[2] / 'evaluation/unseen'))
sys.path.insert(0, str(metric_dir))
from metrics import normalize, PUNCT, CHAR_MAP


def test_greek_contextual_case_and_unicode_expansion():
    cases = ['  Ο ΠΡΟΕΔΡΟΣ: ΑΥΤΟΣ — αυτός!  ', 'Ι\u0308διος και α\u0301λλος',
             'İstanbul, l\'altra—città.\n ΝΟΜΟΣ', '**κείμενο**   (test)\t42%']
    for text in cases:
        source, normalized, indices = mapped_normalize(text, PUNCT, CHAR_MAP)
        assert normalized == normalize(source)
        assert len(indices) == len(normalized)
        assert indices == sorted(indices)
        assert all(0 <= i < len(source) for i in indices)


def test_human_span_mapping_preserves_the_source_words():
    source, normalized, indices = mapped_normalize('Prima frase.\nΚαλό παράδειγμα — και τέλος.', PUNCT, CHAR_MAP)
    begin = normalized.index('καλό')
    end = normalized.index(' και')
    assert source[indices[begin]:indices[end - 1] + 1] == 'Καλό παράδειγμα'
