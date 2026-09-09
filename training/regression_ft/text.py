"""Audited training-label formatting; benchmark references remain immutable.

Do not use this function as a replacement benchmark scoring normalizer. In
particular, NST's punctuation names are retained as words: deleting them would
assume that speakers never pronounced the dictation commands.
"""
import re
import unicodedata


def normalize_training_text(text, split):
    text = unicodedata.normalize('NFC', text)
    if split.startswith('nst_'):
        text = re.sub(r'\\(Punktum|Punkt|Komma|Spørgsmålstegn|Bindestreg|Utropstecken)',
                      lambda m: ' ' + m[1] + ' ', text)
        text = text.replace('( ... tyst under denna inspelning ...)', '')
    if split.startswith('gigaspeechbench_'):
        text = re.sub(r'\[(?:breath|laugh|sigh|Throat clear|cough|humph|chocking|hissing|applause)\]', ' ', text)
        text = re.sub(r'\((?:/?overlap|noise|sil|~)\)', ' ', text)
        text = text.replace('#', '')
    if split == 'eurospeech_it':
        text = re.sub(r'\(\*?(?:Applausi|Commenti)[^)]*\)', ' ', text)
        text = text.replace('+', ' più ').replace('°', '')
    if split == 'eurospeech_bg':
        text = re.sub(r'\((?:Ръкопляскания|Шум|шум|Реплики|реплики|Смях|Оживление|Показва)[^)]*\)', ' ', text)
        text = text.replace('§', ' параграф ').replace('ѝ', 'и').replace('ѝ', 'и')
    if split.endswith('_el'):
        text = text.translate(str.maketrans({'ς': 'σ', 'ϋ': 'υ', 'ΐ': 'ί', 'Ί': 'ί', '΄': "'", ';': '?'}))
    text = text.translate(str.maketrans({'’': "'", '‘': "'", '`': "'", '–': '-', '—': '-', '‑': '-', ';': ','}))
    text = text.translate(str.maketrans('', '', '"„“”«»*()[]'))
    if split == 'nst_da_da':
        text = text.replace('&', ' og ')
    text = re.sub(r'\s+([,.!?])', r'\1', text)
    return ' '.join(text.split())
