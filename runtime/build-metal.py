#!/usr/bin/env python3
"""Build Orukeet's patched Metal SDK from pinned sources (Python 3.9+)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent


def run(*args, cwd=None, capture=False):
    command = [str(arg) for arg in args]
    print('+ ' + shlex.join(command), flush=True)
    # The ASR SDK does not use the optional TTS assets stored in Git LFS.
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
    return subprocess.run(command, cwd=cwd, env=env, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def load_lock(root=ROOT):
    lock = json.loads((root / 'sources.lock.json').read_text())
    if lock['schema_version'] != 1:
        raise ValueError('Unsupported runtime lock schema')
    for source in lock['sources'].values():
        commit = source['commit']
        if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
            raise ValueError('Source revisions must be full Git commit hashes')
    for kind, patches in lock['patches'].items():
        expected = [item['path'] for item in patches]
        actual = sorted(str(p.relative_to(root)) for p in (root / f'{kind}-patches').glob('*.patch'))
        if expected != actual:
            raise ValueError(f'{kind} patch list differs from sources.lock.json')
        for item in patches:
            if sha256(root / item['path']) != item['sha256']:
                raise ValueError(f"Patch SHA-256 mismatch: {item['path']}")
    return lock


def checkout(destination, source, repository=None):
    destination.mkdir(parents=True, exist_ok=True)
    run('git', 'init', '--quiet', destination)
    run('git', '-C', destination, 'remote', 'add', 'origin', repository or source['repository'])
    run('git', '-C', destination, 'fetch', '--depth', '1', 'origin', source['commit'])
    run('git', '-C', destination, '-c', 'advice.detachedHead=false', 'checkout', '--quiet', 'FETCH_HEAD')
    if run('git', '-C', destination, 'rev-parse', 'HEAD', capture=True).strip() != source['commit']:
        raise ValueError(f'Wrong source revision at {destination}')


def tree_state(source):
    # Keep the applied patches staged. Reject later edits rather than resetting
    # a developer's changes, including additions picked up by CMake source globs.
    run('git', '-C', source, 'diff', '--exit-code', '--ignore-submodules=all')
    others = run('git', '-C', source, 'ls-files', '--others', capture=True).strip()
    if others:
        raise ValueError(f'Unexpected source files at {source}: {others}')
    return {
        'commit': run('git', '-C', source, 'rev-parse', 'HEAD', capture=True).strip(),
        'tree': run('git', '-C', source, 'write-tree', capture=True).strip(),
    }


def prepare(work, lock, repositories, root=ROOT):
    sources = work / 'sources'
    paths = {'nemo': sources / 'nemo', 'ggml': sources / 'nemo/ggml',
             'sentencepiece': sources / 'sentencepiece'}
    if sources.exists():
        receipt = json.loads((sources / 'prepared.json').read_text())
        if receipt['lock'] != lock:
            raise ValueError('Source pins or patches changed; select a new --work-dir')
        for name, path in paths.items():
            if tree_state(path) != receipt['trees'][name]:
                raise ValueError(f'{name} sources changed; select a new --work-dir')
        print('Pinned sources and complete patch series already prepared.', flush=True)
        return paths

    work.mkdir(parents=True, exist_ok=True)
    # A failed fetch or patch leaves no partially prepared source directory.
    with tempfile.TemporaryDirectory(prefix='prepare-', dir=work) as temp:
        staging = Path(temp) / 'sources'
        nemo, ggml, spm = staging / 'nemo', staging / 'nemo/ggml', staging / 'sentencepiece'
        checkout(nemo, lock['sources']['nemo'], repositories.get('nemo'))
        gitlink = run('git', '-C', nemo, 'ls-tree', 'HEAD', 'ggml', capture=True).split()
        if gitlink[:3] != ['160000', 'commit', lock['sources']['ggml']['commit']]:
            raise ValueError('ggml pin differs from the NeMo submodule revision')
        # A standalone checkout at the submodule path keeps it independent of
        # any adjacent developer checkout and avoids unrelated submodules.
        checkout(ggml, lock['sources']['ggml'], repositories.get('ggml'))
        checkout(spm, lock['sources']['sentencepiece'], repositories.get('sentencepiece'))
        baseline = sorted((nemo / 'ggml-patches').glob('*.patch'))
        if len(baseline) != 20:
            raise ValueError('Expected the 20 baseline ggml patches at the pinned NeMo revision')
        for patch in baseline:
            run('git', '-C', ggml, 'apply', '--index', patch)
        for kind, target in (('ggml', ggml), ('nemo', nemo)):
            for item in lock['patches'][kind]:
                run('git', '-C', target, 'apply', '--index', root / item['path'])
        trees = {name: tree_state(path) for name, path in
                 (('nemo', nemo), ('ggml', ggml), ('sentencepiece', spm))}
        write_json(staging / 'prepared.json', {'lock': lock, 'trees': trees})
        staging.rename(sources)
    return paths


def configure(source, build, *options):
    run('cmake', '-S', source, '-B', build, '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_OSX_ARCHITECTURES=arm64',
        '-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0', *options)


def build_sdk(work, paths, jobs, lock):
    spm_build, nemo_build, sdk = work / 'sentencepiece-build', work / 'nemo-build', work / 'sdk'
    configure(paths['sentencepiece'], spm_build, '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
              '-DSPM_BUILD_TEST=OFF',
              '-DSPM_ENABLE_SHARED=OFF', '-DSPM_ENABLE_TCMALLOC=OFF')
    run('cmake', '--build', spm_build, '--target', 'sentencepiece-static', '--parallel', jobs)
    configure(paths['nemo'], nemo_build,
              f'-DCMAKE_INSTALL_PREFIX={sdk}', '-DCMAKE_INSTALL_LIBDIR=lib',
              '-DBUILD_SHARED_LIBS=ON', '-DBUILD_TESTING=OFF',
              '-DNEMO_SPEECH_BUILD_ASR=ON', '-DNEMO_SPEECH_BUILD_DIAR=OFF',
              '-DNEMO_SPEECH_BUILD_TTS=OFF', '-DNEMO_SPEECH_BUILD_NMT=OFF',
              '-DNEMO_SPEECH_BUILD_S2S=OFF', '-DNEMO_SPEECH_BUILD_CLI=OFF',
              '-DNEMO_SPEECH_BUILD_MIC_CAPTURE=OFF', '-DNEMO_SPEECH_BUILD_HTTP=OFF',
              '-DNEMO_SPEECH_BUILD_GRPC=OFF', '-DNEMO_SPEECH_BUILD_TESTS=OFF',
              '-DNEMO_SPEECH_BUILD_EXAMPLES=OFF', '-DNEMO_SPEECH_BUILD_TOOLS=OFF',
              # Match the qualified Metal runtime: portable ASR graph operations
              # with backend optimizations supplied by the applied ggml patches.
              '-DNEMO_SPEECH_GGML_PATCHED=OFF', '-DGGML_METAL=ON',
              '-DGGML_METAL_EMBED_LIBRARY=ON', '-DGGML_NATIVE=OFF',
              '-DGGML_BLAS=ON', '-DGGML_BLAS_VENDOR=Apple', '-DGGML_CCACHE=OFF',
              '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_BUILD_TESTS=OFF',
              '-DGGML_BUILD_EXAMPLES=OFF',
              f'-DSENTENCEPIECE_LIB={spm_build}/src/libsentencepiece.a',
              f'-DSENTENCEPIECE_INCLUDE_DIR={paths["sentencepiece"]}/src')
    run('cmake', '--build', nemo_build, '--parallel', jobs)
    run('cmake', '--install', nemo_build)
    licenses = sdk / 'share/licenses/nemo-speech/third_party/sentencepiece'
    licenses.mkdir(parents=True, exist_ok=True)
    for original, name in [('LICENSE', 'LICENSE'), ('third_party/absl/LICENSE', 'absl-LICENSE'),
                           ('third_party/darts_clone/LICENSE', 'darts-clone-LICENSE'),
                           ('third_party/protobuf-lite/LICENSE', 'protobuf-lite-LICENSE')]:
        shutil.copyfile(paths['sentencepiece'] / original, licenses / name)
    provenance = sdk / 'share/orukeet'
    provenance.mkdir(parents=True, exist_ok=True)
    # Record which patched sources and actual binaries this local build used.
    write_json(provenance / 'build.json', {
        'lock': lock,
        'prepared_sources': json.loads((work / 'sources/prepared.json').read_text())['trees'],
        'build_script_sha256': sha256(Path(__file__).resolve()),
        'platform': platform.platform(),
        'cmake': run('cmake', '--version', capture=True).splitlines()[0],
        'compiler': run('c++', '--version', capture=True).splitlines()[0],
        'libraries': {p.name: sha256(p) for p in sorted((sdk / 'lib').glob('*.dylib'))},
    })
    for name in ('libnemo_speech_asr_c.dylib', 'libggml-metal.dylib'):
        if not (sdk / 'lib' / name).is_file():
            raise ValueError(f'Installed SDK is missing {name}')
    print(f'\nOptimized Metal SDK: {sdk}\nPass this directory as Orukeet\'s runtime argument.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=ROOT.parent / 'build/metal',
                        help='Owned source/build/SDK directory (default: build/metal)')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Fetch, verify and apply patches without compiling; works off macOS')
    parser.add_argument('--jobs', type=int, default=min(os.cpu_count() or 4, 8))
    for name in ('nemo', 'ggml', 'sentencepiece'):
        parser.add_argument(f'--{name}-repository',
                            help='Optional Git mirror/local repository; locked commit still required')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    lock = load_lock()
    if not args.prepare_only:
        if sys.platform != 'darwin' or platform.machine().lower() != 'arm64':
            parser.error('The Metal SDK requires native Apple silicon macOS; use --prepare-only elsewhere')
        for tool in ('git', 'cmake', 'ninja', 'xcrun'):
            if not shutil.which(tool):
                parser.error(f'Missing {tool}; see runtime/README.md for prerequisites')
        run('xcrun', '--find', 'clang')
    repositories = {name: getattr(args, name + '_repository')
                    for name in ('nemo', 'ggml', 'sentencepiece')}
    paths = prepare(args.work_dir.resolve(), lock, repositories)
    if args.prepare_only:
        print(f'Patched sources: {paths["nemo"]}')
    else:
        build_sdk(args.work_dir.resolve(), paths, args.jobs, lock)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'error: {error}', file=sys.stderr)
        sys.exit(1)
