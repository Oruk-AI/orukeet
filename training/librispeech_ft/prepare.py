"""Seal a focused LibriSpeech pass; keep all evaluation references intact."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import sentencepiece as spm
import soundfile as sf


SPLITS = {'librispeech_test_other': 2939}
EVALUATION_SPLITS = {'librispeech_test_other', 'fleurs_fr', 'fleurs_el'}
SOURCE_MANIFEST_SHA256 = 'c8c36e9cbf5053ba7159204b2a507185fc11e818f0ab71259e5937d20f9c9623'
PARENT_SHA256 = '0ccfefcd1894871cb0850bd3c464adf5397752840de2a76d1d2d075c4141a945'
FIT_SHA256 = '44ef0eb45fdd122a3900c97f1faf4ffedf567f27e9ad535f24687c36a8c7f704'


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--parent-sha256', default=PARENT_SHA256)
    p.add_argument('--peak-lr', type=float, default=3e-6)
    p.add_argument('--end-lr', type=float, default=3e-7)
    p.add_argument('--targets', type=Path)
    p.add_argument('--targets-sha256')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--seed', type=int, default=20260908)
    p.add_argument('--experiment', default='librispeech-ft-20260908/r1')
    a = p.parse_args()
    assert a.epochs >= 1
    if bool(a.targets) != bool(a.targets_sha256):
        p.error('--targets and --targets-sha256 must be provided together')
    targets = {}
    if a.targets:
        assert sha(a.targets) == a.targets_sha256
        target_rows = [json.loads(line) for line in a.targets.open()]
        targets = {r['uid']: r for r in target_rows}
        assert len(targets) == len(target_rows) == 2939
    a.output.mkdir(parents=True, exist_ok=True)
    assert not (a.output / 'prepared.json').exists(), 'Preparation is already sealed'
    assert sha(a.source_manifest) == SOURCE_MANIFEST_SHA256
    assert sha(a.parent) == a.parent_sha256
    rows = [json.loads(line) for line in a.source_manifest.open()]
    evaluation_rows = [r for r in rows if r['split'] in EVALUATION_SPLITS]
    assert len(evaluation_rows) == 4265
    rows = [r for r in rows if r['split'] in SPLITS]
    assert dict(Counter(r['split'] for r in rows)) == SPLITS
    assert len(rows) == len({r['uid'] for r in rows}) == 2939
    with tarfile.open(a.parent) as archive:
        assets = [m for m in archive.getmembers() if m.name.endswith('.model')]
        assert len(assets) == 1
        tokenizer = spm.SentencePieceProcessor(model_proto=archive.extractfile(assets[0]).read())
    for row in rows:
        audio, rate = sf.read(row['audio_filepath'], dtype='float32')
        assert rate == 16000 and audio.ndim == 1 and np.isfinite(audio).all(), row['uid']
        assert abs(len(audio) / rate - row['duration']) < 1 / rate, row['uid']
        assert hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest() == row['pcm_sha256'], row['uid']
        assert hashlib.sha256(row['text'].encode()).hexdigest() == row['reference_sha256'], row['uid']
        text = row['text'].lower()
        text = text[:1].upper() + text[1:]
        if text[-1] not in '.!?':
            text += '.'
        if targets:
            supplied = targets[row['uid']]
            assert supplied['reference_sha256'] == row['reference_sha256']
            text = supplied['text']
        row['training_text'] = text
        ids = tokenizer.encode(text)
        assert ids and tokenizer.unk_id() not in ids, row['uid']
    eval_path = a.output / 'evaluation.jsonl'
    eval_path.write_text(''.join(json.dumps({k:v for k,v in r.items() if k != 'training_text'}, ensure_ascii=False) + '\n' for r in evaluation_rows))
    assert sha(eval_path) == '8770c5a110b60d3adabf4a8e4826adb516664bd44a8c2352e1da57086f85e500'
    manifests = []
    for split, size in sorted(SPLITS.items()):
        part = [dict({k:v for k,v in r.items() if k != 'training_text'}, text=r['training_text'],
                     training_pcm_sha256=r['pcm_sha256'],
                     training_reference_sha256=hashlib.sha256(r['training_text'].encode()).hexdigest())
                for r in rows if r['split'] == split]
        path = a.output / (split + '.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in part))
        manifests.append(dict(split=split, rows=size, hours=sum(r['duration'] for r in part) / 3600,
                              path=str(path.resolve()), sha256=sha(path),
                              alignment_status='original_evaluation_audio_and_reference_verified'))
    plan = dict(status='sealed_for_training', publication_authorized=False,
                experiment=a.experiment, seed=a.seed,
                parent_stage=('Orukeet FT-4035' if a.parent_sha256 == PARENT_SHA256 else 'Orukeet focused-pass continuation'), parent_sha256=a.parent_sha256, fit_sha256=FIT_SHA256,
                source_manifest_sha256=SOURCE_MANIFEST_SHA256, evaluation_manifest_sha256=sha(eval_path),
                training_rows=len(rows), training_hours=sum(r['duration'] for r in rows) / 3600,
                splits=SPLITS, passes=a.epochs, epochs=a.epochs, training_presentations=len(rows) * a.epochs,
                evaluation_data_used_for_training=True,
                evaluation_interpretation='LibriSpeech test-other is used for training and candidate selection; French and Greek are re-evaluation controls.',
                text_formatting=('Native formatting with reference-corrected words; every target passes normalized-word equivalence.' if targets else 'Sentence-case targets; normalized English words unchanged.'),
                external_targets_sha256=a.targets_sha256, evaluation_references_unchanged=True,
                frozen_gabor_rows=12288,
                batching=dict(max_padded_seconds=120, max_utterances=16, accumulate_batches=4,
                              fused_joint_batch_size=1),
                proposed_optimizer=dict(name='AdamW', peak_lr=a.peak_lr, end_lr=a.end_lr, warmup_fraction=.03,
                                        betas=[.9, .98], weight_decay=.001, clip_grad_norm=1.0))
    write(a.output / 'plan.json', plan)
    prepared = dict(complete=True, plan_sha256=sha(a.output / 'plan.json'), rows=len(rows),
                    manifests=manifests, source_manifest_sha256=SOURCE_MANIFEST_SHA256,
                    evaluation_manifest_sha256=sha(eval_path), preparation_script_sha256=sha(__file__),
                    all_audio_and_reference_hashes_verified=True, unknown_token_rows=0,
                    evaluation_references_unchanged=True, training_targets_equal_evaluation_references=False,
                    training_targets_preserve_normalized_english_words=True)
    write(a.output / 'prepared.json', prepared)
    print(json.dumps(prepared), flush=True)


if __name__ == '__main__':
    main()
