"""Offline checks for source integrity and preservation of developer edits."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest


RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'
spec = importlib.util.spec_from_file_location('build_metal', RUNTIME / 'build-metal.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / 'runtime'
    shutil.copytree(RUNTIME, root)
    return root


def test_changed_patch_is_rejected(runtime):
    patch = next((runtime / 'ggml-patches').glob('*.patch'))
    patch.write_bytes(patch.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        builder.load_lock(runtime)


@pytest.mark.parametrize('change', ['extra', 'missing'])
def test_patch_inventory_must_match_lock(runtime, change):
    if change == 'extra':
        (runtime / 'ggml-patches/9999-unlisted.patch').write_text('unlisted')
    else:
        next((runtime / 'nemo-patches').glob('*.patch')).unlink()
    with pytest.raises(ValueError, match='patch list differs'):
        builder.load_lock(runtime)


def test_moving_source_reference_is_rejected(runtime):
    path = runtime / 'sources.lock.json'
    lock = json.loads(path.read_text())
    lock['sources']['nemo']['commit'] = 'main'
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match='full Git commit hashes'):
        builder.load_lock(runtime)


@pytest.fixture
def repository(tmp_path):
    source = tmp_path / 'repository'
    builder.run('git', 'init', '--quiet', source)
    (source / 'kernel.cpp').write_text('original\n')
    builder.run('git', '-C', source, 'add', 'kernel.cpp')
    builder.run('git', '-C', source, '-c', 'user.name=Test',
                '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false',
                'commit', '--quiet', '-m', 'Fixture')
    return source


def test_source_verification_preserves_unstaged_edits(repository):
    path = repository / 'kernel.cpp'
    path.write_text('developer edit\n')
    with pytest.raises(subprocess.CalledProcessError):
        builder.tree_state(repository)
    assert path.read_text() == 'developer edit\n'


def test_source_verification_rejects_added_files_even_if_ignored(repository):
    # CMake source globs can pick up ignored files too.
    (repository / '.git/info/exclude').write_text('extra.cpp\n')
    path = repository / 'extra.cpp'
    path.write_text('developer addition\n')
    with pytest.raises(ValueError, match='Unexpected source files'):
        builder.tree_state(repository)
    assert path.read_text() == 'developer addition\n'


def test_staged_edits_change_source_fingerprint(repository):
    original = builder.tree_state(repository)
    (repository / 'kernel.cpp').write_text('staged developer edit\n')
    builder.run('git', '-C', repository, 'add', 'kernel.cpp')
    changed = builder.tree_state(repository)
    assert changed['commit'] == original['commit']
    assert changed['tree'] != original['tree']
