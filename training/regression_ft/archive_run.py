"""Back up an audited experiment to the existing private HF model repository."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile

# The VM's root disk is nearly full; the launch environment points TMPDIR at
# the run's scratch filesystem. Keep upload cache traffic on that filesystem.
os.environ.setdefault('HF_XET_CACHE', str(Path(os.environ.get('TMPDIR', '/tmp')) / 'orukeet-hf-xet'))
from huggingface_hub import HfApi, CommitOperationAdd


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--optimizer', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--prefix', required=True)
    a = p.parse_args()
    assert a.prefix.startswith('experiments/regression-ft-20260907/') and '..' not in a.prefix
    completed = json.loads((a.run / 'complete.json').read_text())
    audit = json.loads(a.audit.read_text())
    evaluation = json.loads((a.evaluation / 'comparison.json').read_text())
    candidate_sha = sha(a.candidate)
    assert candidate_sha == audit['candidate_sha256'] == evaluation['models']['candidate']
    from repair_export import verify_training_identity
    verify_training_identity(a.experiment, completed, candidate_sha)
    assert audit['status'] == 'pass' and completed['audit']['exactly_one_pass']
    assert evaluation['status'] == 'complete'
    assert evaluation['models']['parent'] == audit['parent_sha256'] == completed['run']['parent_sha256']
    benchmark_seal = json.loads((a.experiment.parent / 'unseen_20260907/metadata/seal.json').read_text())
    assert evaluation['models']['parakeet'] == benchmark_seal['parakeet_sha256']
    from exposure import export_exposure
    export_exposure(a.experiment, a.run, candidate_sha)
    repo, api = 'oruk/orukeet', HfApi()
    before = api.model_info(repo)
    assert before.private is True
    assert not any(p.startswith(a.prefix + '/') for p in api.list_repo_files(repo))
    bundle = a.experiment / 'private-training-provenance.tar.gz'
    # Raw source references and predictions belong only in this private archive.
    with tarfile.open(bundle, 'w:gz', compresslevel=6) as archive:
        for name in ['plan.json', 'code', 'audio', 'alignment/human-spans', 'alignment/independent-check', 'training-ready']:
            source = a.experiment / name
            for path in sorted(source.rglob('*')) if source.is_dir() else [source]:
                if path.is_file() and path.suffix in {'.json', '.jsonl', '.py'}:
                    archive.add(path, arcname=str(path.relative_to(a.experiment)))
        gabor = a.experiment.parent / 'gabor_half_20260906'
        for name in ['frozen.py', 'fit.py', 'identity.py', 'fit-full/fits.json']:
            archive.add(gabor / name, arcname='gabor-source/' + name)
        for name in ['launch-v1.json', 'diagnostic-launch.json', 'quarantine-cache-retirement.json']:
            archive.add(a.experiment / name, arcname=name)
        for path in sorted((a.experiment / 'export-metadata').iterdir()):
            assert path.is_file() and path.suffix in {'.json', '.yaml'}
            archive.add(path, arcname=str(path.relative_to(a.experiment)))
        benchmark = a.experiment.parent / 'unseen_20260907'
        for phase in ['', 'coverage', 'alignment-followup']:
            for name in ['sources.json', 'seal.json']:
                archive.add(benchmark / phase / 'metadata' / name,
                            arcname=str(Path('benchmark-source') / phase / 'metadata' / name))
            for path in sorted((benchmark / phase / 'metadata/manifests').glob('*.jsonl')):
                archive.add(path, arcname=str(Path('benchmark-source') / phase / 'metadata/manifests' / path.name))
            for path in sorted((benchmark / phase / 'audio-receipts').glob('*.jsonl')):
                archive.add(path, arcname=str(Path('benchmark-source') / phase / 'audio-receipts' / path.name))
            archive.add(benchmark / phase / 'history/history.sqlite',
                        arcname=str(Path('benchmark-source') / phase / 'history/history.sqlite'))
            for name in ['eurospeech_el.jsonl', 'eurospeech_it.jsonl']:
                path = benchmark / phase / 'results/parakeet' / name
                if path.exists():
                    archive.add(path, arcname=str(Path('benchmark-source') / phase / 'results/parakeet' / name))
        for path in sorted((benchmark / 'code').glob('*.py')):
            archive.add(path, arcname='benchmark-source/code/' + path.name)
        for source in [a.run, a.evaluation]:
            for path in sorted(source.rglob('*')):
                if path.is_file() and path.suffix in {'.json', '.jsonl', '.gz', '.md'}:
                    archive.add(path, arcname=str(path.relative_to(a.experiment)))
        for name in ['protocol.json', 'prepared.json', 'manifest.jsonl']:
            archive.add(a.experiment / 'diagnostic' / name, arcname='diagnostic/' + name)
        archive.add(a.audit, arcname=str(a.audit.relative_to(a.experiment)))
    files = []
    for path, name in [(a.candidate, 'orukeet-regression-ft.nemo'), (a.optimizer, 'resume.pt'),
                       (bundle, bundle.name), (a.audit, 'export-audit.json'),
                       (a.evaluation / 'comparison.json', 'comparison.json')]:
        files.append(dict(path=str(path), repo_path=a.prefix + '/' + name,
                          sha256=sha(path), bytes=path.stat().st_size))
    operations = [CommitOperationAdd(path_in_repo=r['repo_path'], path_or_fileobj=r['path']) for r in files]
    readme = ('# Regression fine-tuning experiment\n\nPrivate, fully audited one-pass training candidate. '
              'All 12,288 fitted Gabor kernels remain exact. The optimizer and provenance bundle permit resumption and audit. '
              'Former evaluation splits used for this run are training-exposed. Follow-up results separate fit checks from nontraining controls. '
              'This experiment does not replace canonical release weights or application defaults.\n')
    operations.append(CommitOperationAdd(path_in_repo=a.prefix + '/README.md', path_or_fileobj=readme.encode()))
    commit = api.create_commit(repo_id=repo, repo_type='model', parent_commit=before.sha,
                               operations=operations, commit_message='Back up audited regression fine-tuning experiment privately')
    after = api.model_info(repo, revision=commit.oid, files_metadata=True)
    assert after.private is True and api.model_info(repo).private is True
    remote = {f.rfilename: f for f in after.siblings}
    for item in files:
        stored = remote[item['repo_path']]
        assert stored.size == item['bytes']
        if stored.lfs:
            assert stored.lfs.sha256 == item['sha256']
        else:
            data = Path(item['path']).read_bytes()
            assert stored.blob_id == hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    receipt = dict(status='archived-and-verified-private', repo=repo, revision=commit.oid, parent_revision=before.sha,
                   private=True, files=files, canonical_release_replaced=False)
    (a.experiment / 'private-archive.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
