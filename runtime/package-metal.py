#!/usr/bin/env python3
"""Prepare a versioned Metal SDK archive and installer entry; never publish."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_sdk(sdk):
    require(platform.system() == 'Darwin' and platform.machine() == 'arm64',
            'Package the Metal SDK on Apple silicon')
    receipt = json.loads((sdk / 'share/orukeet/build.json').read_text())
    lock = json.loads((ROOT / 'sources.lock.json').read_text())
    require(receipt['lock'] == lock, 'SDK source pins differ from the current runtime manifest')
    require(receipt['build_script_sha256'] == digest(ROOT / 'build-metal.py'),
            'SDK was built by a different build script; rerun runtime/build-metal.py')
    for patches in lock['patches'].values():
        for patch in patches:
            require(digest(ROOT / patch['path']) == patch['sha256'],
                    f"Patch differs from its pinned hash: {patch['path']}")
    required = ('lib/libnemo_speech_asr_c.dylib', 'lib/libggml-metal.dylib',
                'include/nemo_speech/asr.h', 'lib/cmake/NeMoSpeech/NeMoSpeechConfig.cmake',
                'share/licenses/nemo-speech/LICENSE', 'share/licenses/nemo-speech/NOTICE',
                'share/licenses/nemo-speech/THIRD_PARTY_NOTICES.md',
                'share/licenses/nemo-speech/third_party/ggml/LICENSE',
                'share/licenses/nemo-speech/third_party/sentencepiece/LICENSE')
    for name in required:
        require((sdk / name).is_file(), f'Incomplete SDK: missing {name}')
    libraries = {p.name: digest(p) for p in sorted((sdk / 'lib').glob('*.dylib'))}
    require(libraries == receipt['libraries'], 'SDK libraries differ from the build receipt')
    for path in sorted((sdk / 'lib').glob('*.dylib')):
        if path.is_symlink():
            continue
        arch = subprocess.check_output(['lipo', '-archs', str(path)], text=True).strip()
        require(arch == 'arm64', f'Unexpected architecture in {path.name}: {arch}')
        deps = subprocess.check_output(['otool', '-L', str(path)], text=True)
        for line in deps.splitlines()[1:]:
            dependency = line.strip().split(' (', 1)[0]
            if dependency.startswith('@rpath/'):
                require((sdk / 'lib' / dependency[len('@rpath/'):]).is_file(),
                        f'Unbundled dependency: {dependency}')
            else:
                require(dependency.startswith(('/usr/lib/', '/System/Library/')),
                        f'Nonportable dependency: {dependency}')
        commands = subprocess.check_output(['otool', '-l', str(path)], text=True)
        rpaths = re.findall(r'cmd LC_RPATH\s+cmdsize \d+\s+path (.+?) \(offset', commands)
        require(rpaths and all(p.startswith('@loader_path') for p in rpaths),
                f'Nonportable runtime search path in {path.name}')
    return receipt


def archive_sdk(sdk, destination, prefix):
    """Normalize metadata and preserve internal relative symlinks."""
    paths = [sdk] + sorted(sdk.rglob('*'))
    for path in paths:
        if path.is_symlink():
            require(not os.path.isabs(os.readlink(path)) and path.resolve().is_relative_to(sdk),
                    f'SDK symlink escapes the package: {path}')
        else:
            require(path.is_file() or path.is_dir(), f'Unsupported SDK entry: {path}')
    with destination.open('xb') as raw, gzip.GzipFile(
            filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=9) as compressed:
        with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as archive:
            for path in paths:
                name = prefix if path == sdk else f'{prefix}/{path.relative_to(sdk).as_posix()}'
                info = archive.gettarinfo(str(path), name)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                info.pax_headers = {}
                if info.isfile():
                    with path.open('rb') as source:
                        archive.addfile(info, source)
                else:
                    archive.addfile(info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, default=ROOT.parent / 'build/metal/sdk')
    parser.add_argument('--tag', required=True, help='Orukeet release tag, e.g. v0.1.1')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repository', default='Oruk-AI/orukeet')
    args = parser.parse_args()
    require(re.fullmatch(r'v\d+\.\d+\.\d+(?:[-.][A-Za-z0-9.-]+)?', args.tag),
            'Use an immutable version tag such as v0.1.1')
    require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repository),
            'Repository must be owner/name')
    sdk, output = args.sdk.resolve(), args.output_dir.resolve()
    require(not output.is_relative_to(sdk), 'Output directory must be outside the SDK')
    receipt = validate_sdk(sdk)
    output.mkdir(parents=True, exist_ok=True)
    prefix = f'nemo-speech-orukeet-{args.tag[1:]}-macos-arm64-metal'
    archive = output / (prefix + '.tar.gz')
    require(not archive.exists(), 'Archive already exists; select a new output directory or tag')
    with tempfile.TemporaryDirectory(prefix='package-', dir=output) as temp:
        candidate = Path(temp) / archive.name
        archive_sdk(sdk, candidate, prefix)
        candidate.rename(archive)
    checksum = digest(archive)
    spec = {
        'version': args.tag[1:], 'component_api': 1, 'platform': 'macos_arm64',
        'install_bytes': sum(p.stat().st_size for p in sdk.rglob('*')
                             if p.is_file() and not p.is_symlink()),
        'archives': [{
            'name': archive.name,
            'url': f'https://github.com/{args.repository}/releases/download/{args.tag}/{archive.name}',
            'sha256': checksum, 'size_bytes': archive.stat().st_size, 'extract': 'native-tar',
        }],
    }
    (output / 'metal-runtime.json').write_text(json.dumps(spec, indent=2) + '\n')
    (output / (archive.name + '.sha256')).write_text(f'{checksum}  {archive.name}\n')
    (output / 'metal-build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Prepared {archive}\nSHA-256: {checksum}\nInstaller entry: {output / "metal-runtime.json"}')
    print('The generated download URL becomes available after the release is published.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'error: {error}', file=sys.stderr)
        sys.exit(1)
