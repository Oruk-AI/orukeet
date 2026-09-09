"""Seal deterministic fit checks and controls without consulting new predictions."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import soxr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--training-ready', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--per-split', type=int, default=256)
    a = p.parse_args()
    os.chdir(a.root)
    sys.path.insert(0, str(a.root / 'unseen_20260907/code'))
    import run as core
    import run_coverage as coverage
    from metrics import normalize
    a.output.mkdir(parents=True, exist_ok=True)
    preparation = json.loads(a.training_ready.read_text())
    assert preparation['complete']
    trained = {}
    for entry in preparation['manifests']:
        assert core.sha(entry['path']) == entry['sha256']
        trained[entry['split']] = core.read_rows(entry['path'])
    training_rows = [r for part in trained.values() for r in part]
    known_pcm = {r['benchmark_pcm_sha256'] for r in training_rows}
    known_text = {normalize(r[key]) for r in training_rows for key in ['text', 'evaluation_text', 'original_text'] if key in r}
    known_clusters = {r['cluster'] for r in training_rows if r.get('cluster_metadata_available', True)}
    order = lambda r: hashlib.sha256(('regression-check-20260918:' + r['uid']).encode()).hexdigest()
    candidates, selections, excluded = {}, {}, {}
    phases = [('primary', ''), ('coverage', 'coverage'), ('followup', 'alignment-followup')]
    for phase, directory in phases:
        experiment = a.root / 'unseen_20260907' / directory
        seal = json.loads((experiment / 'metadata/seal.json').read_text())
        for split, spec in sorted(seal['sets'].items()):
            if split in trained:
                rows = sorted(trained[split], key=order)[:a.per_split]
                role = 'training_exposed_fit_check'
            else:
                if split in ['eurospeech_el', 'eurospeech_it']:
                    excluded[split] = 'Known source alignment defect; no validated replacement in this run.'
                    continue
                raw = core.read_rows(spec['path'])
                assert core.sha(spec['path']) == spec['sha256']
                pcm = {r['uid']: r['float32_pcm_sha256'] for r in core.read_rows(experiment / 'audio-receipts' / (split + '.jsonl'))}
                eligible = [r for r in raw if pcm[r['uid']] not in known_pcm
                            and normalize(r['text']) not in known_text
                            and (not r.get('cluster_metadata_available', True) or r['cluster'] not in known_clusters)]
                rows = sorted(eligible, key=order)[:a.per_split]
                role = 'nontraining_control'
                excluded[split] = dict(rows_excluded_for_training_pcm_text_or_known_cluster_overlap=len(raw) - len(eligible))
            if not rows:
                continue
            candidates[split] = (phase, experiment, rows, role)
            selections[split] = dict(rows=len(rows), role=role,
                                     uid_sha256=[hashlib.sha256(r['uid'].encode()).hexdigest() for r in rows])
    protocol = dict(status='sealed_before_candidate_inference', training_ready_sha256=core.sha(a.training_ready),
                    selection='First 256 UIDs by SHA-256(regression-check-20260918:UID), or the full split if smaller.',
                    control_exclusions='Training waveform, normalized transcript, and known recording/speaker cluster matches.',
                    no_claim_of_new_unseen_benchmark=True, splits=selections, exclusions=excluded,
                    script_sha256=core.sha(__file__))
    core.atomic_json(a.output / 'protocol.json', protocol)

    receipts = {}
    def write_audio(row, data, sr, out, db):
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != 16000:
            data = soxr.resample(data, sr, 16000, quality='HQ')
        digest = hashlib.sha256(data.astype('<f4').tobytes()).hexdigest()
        assert digest == receipts[row['uid']]['float32_pcm_sha256']
        path = out / (hashlib.sha256(row['uid'].encode()).hexdigest() + '.flac')
        sf.write(path, data, 16000, subtype='PCM_16')
        decoded, rate = sf.read(path, dtype='float32')
        assert rate == 16000
        return dict(row, audio_filepath=str(path), actual_duration=len(decoded) / 16000,
                    benchmark_pcm_sha256=digest,
                    training_pcm_sha256=hashlib.sha256(decoded.astype('<f4').tobytes()).hexdigest(),
                    audio_file_sha256=core.sha(path), known_pcm_overlap=receipts[row['uid']]['known_pcm_overlap'])
    core.write_audio = write_audio
    result = []
    for split, (phase, experiment, rows, role) in candidates.items():
        if role == 'nontraining_control':
            selected_uids = {r['uid'] for r in rows}
            receipts = {r['uid']: r for r in core.read_rows(experiment / 'audio-receipts' / (split + '.jsonl'))}
            sources = json.loads((experiment / 'metadata/sources.json').read_text())
            materialize = core.materialize if phase == 'primary' else coverage.materialize
            rows = materialize(split, rows, sources, a.scratch, a.output, experiment / 'history/history.sqlite')
            assert {r['uid'] for r in rows} == selected_uids
        for row in rows:
            row['text'] = row.get('evaluation_text', row['text'])
            row['evaluation_role'] = role
            row['split'] = split
            row['duration'] = row['actual_duration']
        result.extend(rows)
        print('DIAGNOSTIC_READY', split, role, len(rows), flush=True)
    assert len({r['uid'] for r in result}) == len(result)
    manifest = a.output / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in result))
    core.atomic_json(a.output / 'prepared.json', dict(status='complete', rows=len(result),
                     roles=dict(Counter(r['evaluation_role'] for r in result)),
                     manifest_sha256=core.sha(manifest), protocol_sha256=core.sha(a.output / 'protocol.json')))


if __name__ == '__main__':
    main()
