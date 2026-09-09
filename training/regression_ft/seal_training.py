"""Create the final finite training manifests without modifying the benchmark."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import tarfile

import sentencepiece as spm

from text import normalize_training_text


# The independent ASR audit reported additional boundary phrases in these pairs.
# Quarantining disputed training labels is distinct from scoring the audit itself;
# its original predictions and full, unfiltered statistics are retained.
BOUNDARY_QUARANTINE = {
    'eurospeech:italy_15_27_2008592_2019040',
    'eurospeech:italy_18_88_21341680_21353248',
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--alignment', type=Path, required=True)
    p.add_argument('--check', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(a.plan.read_text())
    assert sha(a.parent) == plan['parent_sha256']
    with tarfile.open(a.parent) as archive:
        tokenizers = [r for r in archive.getmembers() if r.name.endswith('_tokenizer.model')]
        assert len(tokenizers) == 1
        proto = archive.extractfile(tokenizers[0]).read()
    tokenizer = spm.SentencePieceProcessor(model_proto=proto)
    prepared = json.loads(a.prepared.read_text())
    assert prepared['complete'] and prepared['plan_sha256'] == sha(a.plan)
    alignment = json.loads((a.alignment / 'summary.json').read_text())
    check = json.loads((a.check / 'summary.json').read_text())
    assert check['status'] == 'complete' and check['rows'] == 128
    assert check['predictions_sha256'] == sha(a.check / 'predictions.jsonl')
    check_protocol = json.loads((a.check / 'protocol.json').read_text())
    assert check['protocol_sha256'] == sha(a.check / 'protocol.json')
    manifests, quarantined, text_audits = [], [], {}
    for entry in prepared['manifests']:
        assert sha(entry['path']) == entry['sha256']
        rows = [json.loads(l) for l in Path(entry['path']).open()]
        split = entry['split']
        if split in alignment['sets']:
            path = a.alignment / (split + '-accepted.jsonl')
            repair_sha = sha(path)
            assert repair_sha == alignment['sets'][split]['accepted_sha256'] == check_protocol['input_sha256'][split]
            accepted = {r['uid']: r for r in [json.loads(l) for l in path.open()]}
            filtered = []
            for row in rows:
                if row['uid'] not in accepted or row['uid'] in BOUNDARY_QUARANTINE:
                    quarantined.append(dict(uid_sha256=hashlib.sha256(row['uid'].encode()).hexdigest(), split=split,
                                            duration=row['duration'], reason='boundary_disagreement' if row['uid'] in BOUNDARY_QUARANTINE else 'unresolved_source_alignment'))
                    continue
                repair = accepted[row['uid']]
                assert repair['original_reference_sha256'] == hashlib.sha256(row['text'].encode()).hexdigest()
                row['original_text'] = row['text']
                row['text'] = repair['text']
                row['alignment_status'] = repair['status']
                row['human_reference_spans'] = repair['human_sources']
                row['reference_repair_sha256'] = repair_sha
                filtered.append(row)
            rows = filtered
        assert rows
        formatted, before, after = [], Counter(), Counter()
        changed, max_tokens, total_tokens = 0, 0, 0
        for row in rows:
            row['evaluation_text'] = row['text']
            row['text'] = normalize_training_text(row['text'], split)
            changed += row['text'] != row['evaluation_text']
            for content, counter in [(row['evaluation_text'], before), (row['text'], after)]:
                counter.update(t.surface for t in tokenizer.encode_as_immutable_proto(content).pieces if t.id == tokenizer.unk_id())
            tokens = tokenizer.encode(row['text'])
            if not tokens or not any(c.isalnum() for c in row['text']):
                quarantined.append(dict(uid_sha256=hashlib.sha256(row['uid'].encode()).hexdigest(), split=split,
                                        duration=row['duration'], reason='no_lexical_target_after_annotation_cleanup'))
                continue
            assert tokenizer.unk_id() not in tokens, (split, row['uid'], dict(after))
            max_tokens = max(max_tokens, len(tokens))
            total_tokens += len(tokens)
            row['training_reference_sha256'] = hashlib.sha256(row['text'].encode()).hexdigest()
            row['training_run'] = plan['run']
            formatted.append(row)
        text_audits[split] = dict(input_rows=len(rows), output_rows=len(formatted), changed_rows=changed,
                                 unknown_surfaces_before=dict(before), unknown_surfaces_after=dict(after),
                                 max_target_tokens=max_tokens, total_target_tokens=total_tokens)
        rows = formatted
        path = a.output / (split + '.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows))
        manifests.append(dict(split=split, path=str(path), sha256=sha(path), rows=len(rows),
                              hours=sum(r['duration'] for r in rows) / 3600,
                              original_rows=entry['rows'], original_manifest_sha256=entry['sha256'],
                              alignment_status='human_spans_repaired_with_quarantine' if split in alignment['sets'] else 'no_known_alignment_defect'))
    plan = copy.deepcopy(plan)
    plan.update(status='sealed_for_training', source_plan_sha256=sha(a.plan),
                training_rows=sum(r['rows'] for r in manifests), training_hours=sum(r['hours'] for r in manifests),
                training_splits=len(manifests), quarantined_rows=len(quarantined),
                quarantined_hours=sum(r['duration'] for r in quarantined) / 3600,
                alignment_policy='Preserve all pairs from 22 unaffected splits. For Greek and Italian EuroSpeech, use accepted human transcript spans and quarantine unresolved or known incomplete labels.',
                alignment_summary_sha256=sha(a.alignment / 'summary.json'),
                independent_alignment_audit_sha256=sha(a.check / 'summary.json'),
                training_scope='One complete pass over all 24 selected datasets after the recorded alignment repair and quarantine. Frozen Gabor rows remain immutable; all other parameters train.',
                training_manifests=manifests,
                diagnostics='Fixed hash-selected fit checks and nontraining controls, 256 examples per split when available. All three models receive identical input files.',
                seal_script_sha256=sha(__file__))
    write(a.output / 'text-normalization.json', dict(splits=text_audits,
          script_sha256=sha(Path(__file__).with_name('text.py')), tokenizer_proto_sha256=hashlib.sha256(proto).hexdigest(),
          evaluation_references_preserved=True, unknown_training_tokens=0,
          policy='Preserve case and words; remove explicit non-speech annotations and formatting delimiters; retain NST punctuation names as words; collapse whitespace; map documented unsupported Greek/Bulgarian orthography and punctuation to the unchanged tokenizer. Quarantine labels with no lexical target.'))
    plan['text_normalization_sha256'] = sha(a.output / 'text-normalization.json')
    assert plan['training_rows'] + plan['quarantined_rows'] == plan['selected_rows']
    assert plan['training_splits'] == len(plan['sets']) == 24
    write(a.output / 'training-plan.json', plan)
    ready = dict(complete=True, plan_sha256=plan['source_plan_sha256'],
                 original_preparation_sha256=sha(a.prepared), rows=plan['training_rows'], hours=plan['training_hours'],
                 manifests=manifests, quarantined_rows=len(quarantined), original_float_pcm_verified_for_every_row=True)
    write(a.output / 'prepared.json', ready)
    write(a.output / 'quarantine.json', quarantined)
    print('SEALED', plan['training_rows'], plan['training_hours'], 'hours;', len(quarantined), 'quarantined;', len(manifests), 'splits')


if __name__ == '__main__':
    main()
