"""Recover human transcript spans near misaligned EuroSpeech recordings.

The frozen base model's existing hypotheses locate candidate spans. Training
labels are copied from the source's human transcripts; no predicted word is
inserted. Uncertain pairs are retained in an explicit quarantine manifest.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


CONFIG = dict(window_seconds=180, max_cer=.20, max_wer=.35,
              min_prediction_words_for_repair=8, min_prediction_chars_for_repair=35,
              minimum_word_length_ratio=.70, maximum_word_length_ratio=1.35,
              boundary_word_radius=2, candidate_model_used=False,
              label_source='Contiguous spans copied from published human transcripts')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def mapped_normalize(raw, punct, char_map):
    raw = unicodedata.normalize('NFC', raw)
    chars, positions = [], []
    # Whole-string casing matters for Greek capital sigma at word boundaries.
    translated = raw.translate(char_map)
    lowered = translated.lower()
    source_indices = [i for i, character in enumerate(translated) for _ in character.lower()]
    assert len(source_indices) == len(lowered)
    for c, i in zip(lowered, source_indices):
        c = ' ' if c.isspace() or punct.fullmatch(c) else c
        if c == ' ' and (not chars or chars[-1] == ' '):
            continue
        chars.append(c)
        positions.append(i)
    if chars and chars[-1] == ' ':
        chars.pop()
        positions.pop()
    return raw, ''.join(chars), positions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.root / 'unseen_20260907/code'))
    from metrics import normalize, counts, PUNCT, CHAR_MAP
    a.output.mkdir(parents=True, exist_ok=True)
    config_path = a.output / 'config.json'
    config_path.write_text(json.dumps(CONFIG, indent=2) + '\n')
    summary = dict(config=CONFIG, config_sha256=sha(config_path), script_sha256=sha(__file__),
                   purpose='Training-data alignment repair; original benchmark references remain unchanged.', sets={})
    for split in ['eurospeech_el', 'eurospeech_it']:
        source = a.root / 'unseen_20260907'
        raw = [json.loads(l) for l in (source / 'metadata/manifests' / (split + '.jsonl')).open()]
        pred_path = source / 'results/parakeet' / (split + '.jsonl')
        complete = json.loads((source / 'results' / (split + '.complete.json')).read_text())
        assert sha(pred_path) == complete['prediction_sha256']['parakeet']
        predictions = {r['uid']: r for r in [json.loads(l) for l in pred_path.open()]}
        assert len(predictions) == len(raw)
        sessions = defaultdict(list)
        for r in raw:
            sessions[r['source_session_id']].append(r)
        for session in sessions:
            sessions[session].sort(key=lambda r: (r['start_seconds'], r['uid']))
        times = {s: [r['start_seconds'] for r in rows] for s, rows in sessions.items()}
        accepted, rejected, tally = [], [], Counter()
        for index, row in enumerate(raw):
            pred = predictions[row['uid']]
            assert pred['model_sha256'] == '3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d'
            assert pred['reference_sha256'] == hashlib.sha256(row['text'].encode()).hexdigest()
            hyp = normalize(pred['pred_text'])
            hw = hyp.split()

            def assess(text):
                c = counts(text, pred['pred_text'])
                cer = c['char_errors'] / max(c['chars'], 1)
                wer = c['errors'] / max(c['words'], 1)
                ratio = c['words'] / max(len(hw), 1)
                good = (bool(c['words']) and bool(hw) and cer <= CONFIG['max_cer']
                        and wer <= CONFIG['max_wer'] and CONFIG['minimum_word_length_ratio'] <= ratio <= CONFIG['maximum_word_length_ratio'])
                return good, cer, wer, c

            original_ok, old_cer, old_wer, _ = assess(row['text'])
            result = dict(uid=row['uid'], split=split, original_reference_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
                          teacher_prediction_sha256=hashlib.sha256(pred['pred_text'].encode()).hexdigest(),
                          original_teacher_cer=old_cer, original_teacher_wer=old_wer)
            if original_ok and not pred['failure']:
                result.update(status='assigned_pair_verified', text=row['text'], human_sources=[dict(uid=row['uid'], start=0, end=len(row['text']))],
                              aligned_teacher_cer=old_cer, aligned_teacher_wer=old_wer)
                accepted.append(result)
                tally[result['status']] += 1
                continue
            if pred['failure'] or len(hw) < CONFIG['min_prediction_words_for_repair'] or len(hyp) < CONFIG['min_prediction_chars_for_repair']:
                result['status'] = 'quarantined_insufficient_alignment_evidence'
                rejected.append(result)
                tally[result['status']] += 1
                continue
            session = row['source_session_id']
            lo = bisect_left(times[session], row['start_seconds'] - CONFIG['window_seconds'])
            hi = bisect_right(times[session], row['start_seconds'] + CONFIG['window_seconds'])
            neighbors = sessions[session][lo:hi]
            raw_context = '\n'.join(unicodedata.normalize('NFC', r['text']) for r in neighbors)
            context, normalized, positions = mapped_normalize(raw_context, PUNCT, CHAR_MAP)
            assert normalized == normalize(context)
            alignment = fuzz.partial_ratio_alignment(hyp, normalized)
            if alignment is None:
                result['status'] = 'quarantined_no_alignment'
                rejected.append(result)
                tally[result['status']] += 1
                continue
            words = list(re.finditer(r'\S+', normalized))
            starts = [w.start() for w in words]
            first = max(0, bisect_right(starts, alignment.dest_start) - 1)
            last = bisect_left(starts, alignment.dest_end)
            options = []
            radius = CONFIG['boundary_word_radius']
            for begin in range(max(0, first - radius), min(len(words), first + radius + 1)):
                for end in range(max(begin + 1, last - radius), min(len(words), last + radius) + 1):
                    left, right = positions[words[begin].start()], positions[words[end - 1].end() - 1] + 1
                    label = context[left:right]
                    good, cer, wer, c = assess(label)
                    target_words = normalize(label).split()
                    edge_ok = bool(set(target_words[:3]) & set(hw[:3])) and bool(set(target_words[-3:]) & set(hw[-3:]))
                    if good and edge_ok:
                        options.append((cer, wer, abs(len(target_words) - len(hw)), left, right, label))
            if not options:
                result['status'] = 'quarantined_no_reliable_human_span'
                rejected.append(result)
                tally[result['status']] += 1
                continue
            cer, wer, _, left, right, label = min(options)
            source_spans, cursor = [], 0
            for neighbor in neighbors:
                text = unicodedata.normalize('NFC', neighbor['text'])
                if cursor < right and cursor + len(text) > left:
                    source_spans.append(dict(uid=neighbor['uid'], start=max(0, left - cursor), end=min(len(text), right - cursor)))
                cursor += len(text) + 1
            reconstructed = '\n'.join(unicodedata.normalize('NFC', next(r['text'] for r in neighbors if r['uid'] == span['uid']))[span['start']:span['end']]
                                      for span in source_spans)
            assert reconstructed == label, 'Label does not reconstruct from human sources'
            result.update(status='realigned_human_span', text=label, human_sources=source_spans,
                          aligned_teacher_cer=cer, aligned_teacher_wer=wer,
                          new_reference_sha256=hashlib.sha256(label.encode()).hexdigest())
            accepted.append(result)
            tally[result['status']] += 1
            if index % 250 == 0:
                print('ALIGNMENT', split, index, dict(tally), flush=True)
        for suffix, rows in [('accepted', accepted), ('quarantined', rejected)]:
            (a.output / f'{split}-{suffix}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        summary['sets'][split] = dict(rows=len(raw), accepted=len(accepted), quarantined=len(rejected), statuses=dict(tally),
                                     teacher_prediction_file_sha256=sha(pred_path),
                                     accepted_sha256=sha(a.output / f'{split}-accepted.jsonl'),
                                     quarantined_sha256=sha(a.output / f'{split}-quarantined.jsonl'))
        (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('ALIGNMENT_COMPLETE', split, summary['sets'][split], flush=True)


if __name__ == '__main__':
    main()
