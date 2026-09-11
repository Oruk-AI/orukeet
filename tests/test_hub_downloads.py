"""Verify Hub provenance and integrity without downloading model-sized fixtures."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('download_onnx', ROOT / 'examples/download_onnx.py')
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


@pytest.fixture
def release(tmp_path, monkeypatch):
    archive = tmp_path / 'fixture.tar.bz2'
    archive.write_bytes(b'ONNX release fixture')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({
        'archive': archive.name,
        'archive_bytes': archive.stat().st_size,
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    }))
    monkeypatch.setattr(downloader, 'MANIFEST_SHA256', hashlib.sha256(manifest.read_bytes()).hexdigest())
    calls = []

    def download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(manifest if filename == 'onnx/manifest.json' else archive)

    monkeypatch.setattr(downloader, 'hf_hub_download', download)
    return manifest, archive, calls


@pytest.mark.parametrize('offline', [False, True])
def test_fetches_manifest_and_weights_from_same_pinned_hub_revision(release, tmp_path, offline):
    _, archive, calls = release
    assert downloader.download(tmp_path, local_files_only=offline) == archive
    assert [call[:2] for call in calls] == [
        ('oruk/orukeet', 'onnx/manifest.json'),
        ('oruk/orukeet', 'onnx/' + archive.name),
    ]
    for _, _, options in calls:
        assert options == dict(revision='55a984d46f68323301837194ce647c702f55facc',
                               local_dir=tmp_path / 'models', local_files_only=offline)


def test_changed_manifest_rejected_before_weight_download(release, tmp_path):
    manifest, _, calls = release
    manifest.write_text('{}')
    with pytest.raises(ValueError, match='SHA-256 mismatch: manifest.json'):
        downloader.download(tmp_path)
    assert len(calls) == 1


@pytest.mark.parametrize('same_size', [False, True])
def test_damaged_archive_rejected(release, tmp_path, same_size):
    _, archive, _ = release
    archive.write_bytes(b'x' * (archive.stat().st_size if same_size else 3))
    with pytest.raises(ValueError, match='SHA-256 mismatch' if same_size else 'size mismatch'):
        downloader.download(tmp_path)


def test_missing_offline_cache_does_not_fall_back_to_network(tmp_path, monkeypatch):
    def missing(repo, filename, **options):
        assert options['local_files_only'] is True
        raise FileNotFoundError('Not cached')
    monkeypatch.setattr(downloader, 'hf_hub_download', missing)
    with pytest.raises(FileNotFoundError, match='Not cached'):
        downloader.download(tmp_path, local_files_only=True)
