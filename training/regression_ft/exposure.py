"""Write model-specific training exposure without changing historical benchmarks."""
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def export_exposure(experiment, run, candidate_sha):
    experiment, run = Path(experiment), Path(run)
    sys.path.insert(0, str(experiment.parent / 'unseen_20260907/code'))
    from metrics import normalize
    plan_path = experiment / 'training-ready/training-plan.json'
    plan = json.loads(plan_path.read_text())
    complete = json.loads((run / 'complete.json').read_text())
    from repair_export import verify_training_identity
    verify_training_identity(experiment, complete, candidate_sha)
    assert complete['run']['plan_sha256'] == sha(plan_path)
    assert complete['audit']['exactly_one_pass']
    target = run / 'training-exposure.jsonl.gz'
    seen, counts = set(), Counter()
    with target.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding='utf-8') as out:
            for spec in plan['training_manifests']:
                assert sha(spec['path']) == spec['sha256']
                for line in Path(spec['path']).open():
                    row = json.loads(line)
                    uid = digest(row['uid'])
                    assert uid not in seen
                    seen.add(uid)
                    assert digest(row['text']) == row['training_reference_sha256']
                    item = dict(uid_sha256=uid, split=row['split'],
                                benchmark_pcm_sha256=row['benchmark_pcm_sha256'],
                                training_pcm_sha256=row['training_pcm_sha256'],
                                training_text_sha256=row['training_reference_sha256'],
                                normalized_text_sha256={key: digest(normalize(row[key])) for key in
                                    ['text', 'evaluation_text', 'original_text'] if key in row},
                                cluster_sha256=digest(row['cluster']) if row.get('cluster_metadata_available', True) and row.get('cluster') else None)
                    out.write(json.dumps(item, sort_keys=True) + '\n')
                    counts[row['split']] += 1
    assert dict(counts) == complete['audit']['seen_by_split']
    assert len(seen) == plan['training_rows'] == complete['audit']['seen_rows']
    result = dict(status='complete', candidate_sha256=candidate_sha, parent_sha256=plan['parent_sha256'],
                  rows=len(seen), splits=dict(counts), path=str(target), sha256=sha(target),
                  plan_sha256=sha(plan_path), script_sha256=sha(__file__),
                  normalizer_sha256=sha(experiment.parent / 'unseen_20260907/code/metrics.py'),
                  prior_training_exposure='Inherit all training exposure of the parent checkpoint.',
                  reference_selection_exposure='The full original benchmark informed selection of training splits. Greek/Italian alignment also consulted nearby human transcript context.',
                  role='These records are training-exposed for this candidate. Do not report them as unseen evaluation.',
                  historical_benchmark_files_modified=False)
    (run / 'training-exposure.json').write_text(json.dumps(result, indent=2) + '\n')
    print('TRAINING_EXPOSURE_RECORDED', len(seen), candidate_sha, flush=True)
    return result
