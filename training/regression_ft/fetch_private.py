"""Retain hash-verified local copies of this run's private model and optimizer."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    receipt = json.loads(a.receipt.read_text())
    assert receipt['status'] == 'archived-and-verified-private' and receipt['private'] is True
    assert receipt['repo'] == 'oruk/orukeet'
    api = HfApi()
    assert api.model_info(receipt['repo']).private is True
    archived = api.model_info(receipt['repo'], revision=receipt['revision'], files_metadata=True)
    parent = api.model_info(receipt['repo'], revision=receipt['parent_revision'], files_metadata=True)
    assert archived.private is True and parent.private is True
    identity = lambda f: (f.blob_id, f.size, f.lfs.sha256 if f.lfs else None)
    before = {f.rfilename: identity(f) for f in parent.siblings}
    after = {f.rfilename: identity(f) for f in archived.siblings}
    assert all(after.get(name) == value for name, value in before.items() if name != '.gitattributes')
    expected = {f['repo_path'] for f in receipt['files']} | {'experiments/regression-ft-20260907/r1/README.md'}
    assert after.keys() - before.keys() == expected
    storage_rules = []
    if before.get('.gitattributes') != after.get('.gitattributes'):
        contents = [Path(hf_hub_download(receipt['repo'], '.gitattributes', revision=receipt[key])).read_text()
                    for key in ['parent_revision', 'revision']]
        assert contents[1].startswith(contents[0])
        storage_rules = contents[1][len(contents[0]):].splitlines()
        assert storage_rules
        for rule in storage_rules:
            fields = rule.split()
            assert fields[0] in expected and fields[1:] == ['filter=lfs', 'diff=lfs', 'merge=lfs', '-text']
    a.output.mkdir(parents=True, exist_ok=True)
    files = []
    for item in receipt['files']:
        name = item['repo_path']
        assert name.startswith('experiments/regression-ft-20260907/r1/') and '..' not in name.split('/')
        path = a.output / name
        if not (path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256']):
            path = Path(hf_hub_download(receipt['repo'], name, revision=receipt['revision'], local_dir=a.output))
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
        files.append(dict(path=str(path.resolve()), sha256=item['sha256'], bytes=item['bytes']))
        print('VERIFIED_LOCAL', path.name, item['bytes'], flush=True)
    result = dict(status='verified', private_archive_receipt_sha256=sha(a.receipt),
                  repo=receipt['repo'], revision=receipt['revision'], files=files,
                  parent_revision=receipt['parent_revision'], canonical_release_files_unchanged=True,
                  existing_files_unchanged_except_verified_lfs_rules=True, added_storage_rules=storage_rules,
                  added_paths=sorted(expected))
    (a.output / 'local-artifacts.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
