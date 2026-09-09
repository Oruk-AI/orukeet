"""Archive the audited candidate in a new path of the existing private model repo."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import tarfile

os.environ.setdefault('HF_XET_CACHE', str(Path(os.environ.get('TMPDIR', '/tmp')) / 'orukeet-hf-xet'))
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--gabor-source', type=Path, required=True)
    p.add_argument('--verify-revision', help='Verify an existing upload without writing another commit.')
    p.add_argument('--parent-revision', help='Repository revision immediately before the existing upload.')
    a = p.parse_args()
    if bool(a.verify_revision) != bool(a.parent_revision):
        p.error('--verify-revision and --parent-revision must be provided together')
    plan = json.loads((a.experiment / 'prepared/plan.json').read_text())
    assert re.fullmatch(r'librispeech-ft-20260908/r[1-9][0-9]*', plan['experiment'])
    prefix, repo = 'experiments/' + plan['experiment'], 'oruk/orukeet'
    complete = json.loads((a.experiment / 'run/complete.json').read_text())
    audit = json.loads((a.experiment / 'export-audit.json').read_text())
    result = json.loads((a.experiment / 'scored/comparison.json').read_text())
    scoring_audit = json.loads((a.experiment / 'scored/hypotheses-audit.json').read_text())
    candidate_sha = sha(a.candidate)
    assert candidate_sha == audit['candidate_sha256'] == result['models']['orukeet_targeted']
    assert candidate_sha == complete['audit']['checkpoint_sha256']
    passes = plan.get('epochs', 1)
    assert complete['audit']['seen_rows'] == 2939 * passes
    assert complete['audit'].get('data_passes', 1) == passes
    if passes > 1:
        assert complete['audit']['every_record_presentations'] == passes
        assert complete['audit']['unique_training_rows'] == 2939
    assert audit['status'] == 'pass' and audit['frozen_gabor_rows_exact'] == 12288
    assert scoring_audit['status'] == 'passed' and result['status'] == 'complete'
    assert result['evaluation_data_used_for_training'] is True
    assert audit['parent_sha256'] == complete['run']['parent_sha256'] == plan['parent_sha256']
    assert result['models']['orukeet_ft4035'] == '0ccfefcd1894871cb0850bd3c464adf5397752840de2a76d1d2d075c4141a945'
    api = HfApi()
    before = api.model_info(repo, revision=a.parent_revision, files_metadata=True)
    assert before.private is True
    assert not any(f.rfilename.startswith(prefix + '/') for f in before.siblings)
    bundle = a.experiment / 'private-provenance.tar.gz'
    if not a.verify_revision:
        with tarfile.open(bundle, 'w:gz', compresslevel=6) as archive:
            for name in ['code', 'prepared', 'run', 'evaluation', 'scored', 'scoring-source', 'targets']:
                for path in sorted((a.experiment / name).rglob('*')):
                    if path.is_file() and path.suffix in {'.py', '.json', '.jsonl', '.gz', '.txt', '.md'}:
                        archive.add(path, arcname=str(path.relative_to(a.experiment)))
            for name in ['export-audit.json', 'launch.json', 'models.json', 'pipeline.json', 'training.log']:
                archive.add(a.experiment / name, arcname=name)
            for name in ['frozen.py', 'fit.py', 'identity.py', 'fit-full/fits.json']:
                archive.add(a.gabor_source / name, arcname='gabor-source/' + name)
    files = []
    for path in [a.candidate, bundle, a.experiment / 'export-audit.json',
                 a.experiment / 'scored/comparison.json', a.experiment / 'scored/hypotheses-audit.json']:
        files.append(dict(path=str(path), repo_path=prefix + '/' + path.name,
                          sha256=sha(path), bytes=path.stat().st_size))
    operations = [CommitOperationAdd(path_in_repo=f['repo_path'], path_or_fileobj=f['path']) for f in files]
    readme = ('# Focused LibriSpeech Orukeet pass\n\n'
              f"Private candidate: {passes} passes over all 2,939 LibriSpeech test-other recordings; {complete['audit']['optimizer_steps']} "
              f"AdamW updates, peak LR {plan['proposed_optimizer']['peak_lr']}, final LR {plan['proposed_optimizer']['end_lr']}. "
              'All 12,288 Gabor kernels remain byte-identical. LibriSpeech test-other is used for training, '
              'candidate selection, and re-evaluation. French and Greek measure changes relative to the canonical parent. '
              'Baseline provenance and decoding settings are recorded in comparison.json. The provenance archive contains exact '
              'manifests, transcripts, predictions, code, fit parameters, run records, and numeric audits. '
              'Canonical release files remain unchanged.\n')
    operations.append(CommitOperationAdd(path_in_repo=prefix + '/README.md', path_or_fileobj=readme.encode()))
    revision = a.verify_revision
    if not revision:
        revision = api.create_commit(repo_id=repo, repo_type='model', parent_commit=before.sha,
                                      operations=operations, commit_message='Archive focused LibriSpeech pass and matched evaluation privately').oid
    after = api.model_info(repo, revision=revision, files_metadata=True)
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
    attributes_added = []
    for prior in before.siblings:
        stored = remote[prior.rfilename]
        if prior.rfilename == '.gitattributes' and prior.blob_id != stored.blob_id:
            old = Path(hf_hub_download(repo, '.gitattributes', revision=before.sha)).read_bytes()
            new = Path(hf_hub_download(repo, '.gitattributes', revision=revision)).read_bytes()
            assert new.startswith(old), 'Existing LFS rules changed'
            attributes_added = new[len(old):].decode().splitlines()
            allowed = {f['repo_path'] + ' filter=lfs diff=lfs merge=lfs -text'
                       for f in files if remote[f['repo_path']].lfs}
            assert attributes_added and set(attributes_added) <= allowed, 'Unexpected new LFS rules'
            continue
        assert (prior.size, prior.blob_id) == (stored.size, stored.blob_id)
        if prior.lfs:
            assert prior.lfs.sha256 == stored.lfs.sha256
    assert set(remote) - {f.rfilename for f in before.siblings} == {f['repo_path'] for f in files} | {prefix + '/README.md'}
    expected_readme = readme.encode()
    assert remote[prefix + '/README.md'].blob_id == hashlib.sha1(b'blob ' + str(len(expected_readme)).encode() + b'\0' + expected_readme).hexdigest()
    receipt = dict(status='archived-and-verified-private', repo=repo, revision=revision,
                   parent_revision=before.sha, private=True, files=files,
                   canonical_release_replaced=False, preexisting_model_and_document_files_unchanged=True,
                   gitattributes_added_rules=attributes_added, verification_script_sha256=sha(Path(__file__)))
    (a.experiment / 'private-archive.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
