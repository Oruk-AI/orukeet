import hashlib
import io
import json
import stat
import zipfile

import pytest

from orukeet import install


def bundle(monkeypatch, entries, *, digest=None):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, content in entries:
            archive.writestr(name, content)
    data = stream.getvalue()
    spec = {'archives': [{'name': 'runtime.zip', 'url': 'https://unused.invalid/sdk',
                          'sha256': digest or hashlib.sha256(data).hexdigest()}]}
    monkeypatch.setattr(install, 'runtime_spec', lambda device: spec)
    monkeypatch.setattr(install.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(data))
    return spec


def test_verified_install_is_reused_offline_and_corruption_is_rejected(tmp_path, monkeypatch):
    bundle(monkeypatch, [('sdk/lib/libnemo_speech_asr_c.so', b'native test fixture')])
    root = install.install_runtime('cpu', tmp_path)
    monkeypatch.setattr(install.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('Unexpected download'))
    assert install.install_runtime('cpu', tmp_path) == root
    (root / 'lib/libnemo_speech_asr_c.so').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='SHA-256'):
        install.install_runtime('cpu', tmp_path)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'C:/escape', r'..\escape'])
def test_archive_cannot_escape_staging(tmp_path, monkeypatch, name):
    bundle(monkeypatch, [(name, b'bad')])
    with pytest.raises(ValueError, match='Unsafe ZIP'):
        install.install_runtime('cpu', tmp_path)
    assert not list(tmp_path.iterdir())


def test_archive_symlinks_rejected(tmp_path, monkeypatch):
    entry = zipfile.ZipInfo('sdk/link')
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    bundle(monkeypatch, [(entry, '../../outside')])
    with pytest.raises(ValueError, match='symlink'):
        install.install_runtime('cpu', tmp_path)


def test_bad_hash_never_installed(tmp_path, monkeypatch):
    bundle(monkeypatch, [('sdk/lib/libnemo_speech_asr_c.so', b'fixture')], digest='0' * 64)
    with pytest.raises(ValueError, match='SHA-256'):
        install.install_runtime('cpu', tmp_path)
    assert not list(tmp_path.iterdir())


def test_incomplete_sdk_rejected(tmp_path, monkeypatch):
    bundle(monkeypatch, [('sdk/README', b'no library')])
    with pytest.raises(ValueError, match='lacks'):
        install.install_runtime('cpu', tmp_path)


@pytest.mark.parametrize('system,arch,device', [('darwin','arm64','metal'), ('darwin','x86_64','cpu'),
    ('linux','x86_64','cuda'), ('linux','aarch64','cpu'), ('win32','AMD64','vulkan')])
def test_pinned_platforms(monkeypatch, system, arch, device):
    monkeypatch.setattr(install.sys, 'platform', system)
    monkeypatch.setattr(install.platform, 'machine', lambda: arch)
    spec = install.runtime_spec(device)
    assert spec['archives'] and all(len(a['sha256']) == 64 for a in spec['archives'])


def test_explicit_gpu_is_not_replaced(monkeypatch):
    monkeypatch.setattr(install.shutil, 'which', lambda _: None)
    assert install.resolve_device('cuda') == 'cuda'


@pytest.mark.parametrize('artifact', ['source', 'q8', 'f16'])
def test_fetch_checks_exact_artifact(tmp_path, monkeypatch, artifact):
    import huggingface_hub
    bad = tmp_path / 'bad.gguf'
    bad.write_bytes(b'incorrect')
    def download(repo, path, **kwargs):
        assert repo == 'oruk/orukeet'
        assert kwargs['revision'] == '555136b50265a132d4cea0d35560c26fc4f657ab'
        assert path == {'source': 'orukeet-v0.1.0.nemo', 'q8': 'orukeet-v0.1.0-q8.gguf',
                        'f16': 'orukeet-v0.1.0-f16.gguf'}[artifact]
        return str(bad)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    with pytest.raises(ValueError, match='size mismatch'):
        install.fetch(artifact, tmp_path)


def test_historical_training_archives_are_not_current_model_downloads(tmp_path, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download',
                        lambda *a, **k: pytest.fail('Historical artifact must not download'))
    with pytest.raises(ValueError, match='Unknown Orukeet artifact'):
        install.fetch('training-inputs', tmp_path)
