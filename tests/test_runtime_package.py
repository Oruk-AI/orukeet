"""Release archives must round-trip through the real runtime installer."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

from orukeet import install


RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'
spec = importlib.util.spec_from_file_location('package_metal', RUNTIME / 'package-metal.py')
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


@pytest.fixture
def sdk(tmp_path):
    root = tmp_path / 'sdk'
    (root / 'lib').mkdir(parents=True)
    (root / 'lib/libnemo_speech_asr_c.dylib').write_bytes(b'native library fixture')
    return root


def test_archive_is_deterministic_and_installs_offline(sdk, tmp_path, monkeypatch):
    first, second = tmp_path / 'first.tar.gz', tmp_path / 'second.tar.gz'
    packager.archive_sdk(sdk, first, 'sdk')
    for path in [sdk, *sdk.rglob('*')]:
        os.utime(path, (1234567890, 1234567890))
    packager.archive_sdk(sdk, second, 'sdk')
    assert first.read_bytes() == second.read_bytes()
    entry = {'archives': [{'name': first.name, 'url': first.as_uri(),
                           'sha256': packager.digest(first)}]}
    monkeypatch.setattr(install, 'runtime_spec', lambda device: entry)
    installed = install.install_runtime('metal', tmp_path / 'cache')
    assert (installed / 'lib/libnemo_speech_asr_c.dylib').read_bytes() == b'native library fixture'
    receipt = json.loads((installed / 'receipt.json').read_text())
    assert receipt['files']
    monkeypatch.setattr(install.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('Network used'))
    assert install.install_runtime('metal', tmp_path / 'cache') == installed
    with pytest.raises(FileExistsError):
        packager.archive_sdk(sdk, first, 'sdk')


@pytest.mark.skipif(os.name == 'nt', reason='macOS SDK symlinks require Unix permissions')
def test_internal_sdk_symlink_survives_install(sdk, tmp_path, monkeypatch):
    target = sdk / 'lib/libnemo_speech_asr_c.dylib'
    real = target.with_name('libnemo_speech_asr_c.1.dylib')
    target.rename(real)
    target.symlink_to(real.name)
    archive = tmp_path / 'sdk.tar.gz'
    packager.archive_sdk(sdk, archive, 'sdk')
    entry = {'archives': [{'name': archive.name, 'url': archive.as_uri(),
                           'sha256': packager.digest(archive)}]}
    monkeypatch.setattr(install, 'runtime_spec', lambda device: entry)
    installed = install.install_runtime('metal', tmp_path / 'cache')
    link = installed / 'lib' / target.name
    assert link.is_symlink() and os.readlink(link) == real.name
    assert link.read_bytes() == real.read_bytes()


@pytest.mark.skipif(os.name == 'nt', reason='macOS SDK symlinks require Unix permissions')
def test_sdk_symlink_cannot_escape_archive(sdk, tmp_path):
    (sdk / 'lib/escape').symlink_to('../../outside')
    with pytest.raises(ValueError, match='escapes the package'):
        packager.archive_sdk(sdk, tmp_path / 'bad.tar.gz', 'sdk')
