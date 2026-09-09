"""Explicit, hash-checked downloads. Transcription itself never uses the network."""
import hashlib
import json
import platform
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile


def verify(path, expected):
    with Path(path).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError(f'SHA-256 mismatch: {Path(path).name}')


def resolve_device(device):
    if device != 'auto':
        return device
    if sys.platform == 'darwin' and platform.machine().lower() in ('arm64', 'aarch64'):
        return 'metal'
    if shutil.which('nvidia-smi'):
        try:
            if subprocess.run(['nvidia-smi', '-L'], capture_output=True, timeout=5).returncode == 0:
                return 'cuda'
        except (OSError, subprocess.TimeoutExpired):
            pass
    return 'cpu'


def runtime_spec(device):
    arch = platform.machine().lower()
    arch = 'arm64' if arch in ('arm64', 'aarch64') else 'x86_64' if arch in ('amd64', 'x86_64') else arch
    tag = ('win_amd64' if sys.platform == 'win32' and arch == 'x86_64' else
           'macos_' + arch if sys.platform == 'darwin' else
           'linux_' + ('aarch64' if arch == 'arm64' else arch) if sys.platform == 'linux' else 'unsupported')
    catalog = json.loads(Path(__file__).with_name('native_runtimes.json').read_text())
    try:
        return catalog['asr-nvidia-' + device]['platforms'][tag]
    except KeyError:
        raise ValueError(f'No pinned {device} runtime for {sys.platform}/{arch}') from None


def install_runtime(device, cache):
    spec = runtime_spec(device)
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    destination = Path(cache) / ('runtime-' + device + '-' + digest[:12])
    if (destination / 'receipt.json').is_file():
        receipt = json.loads((destination / 'receipt.json').read_text())
        if receipt.get('spec_sha256') == digest and receipt.get('files'):
            for name, sha in receipt['files'].items():
                if not (destination / name).resolve().is_relative_to(destination.resolve()):
                    raise ValueError('Unsafe runtime receipt path')
                verify(destination / name, sha)
            return destination
    Path(cache).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='install-', dir=cache) as temp:
        staging = Path(temp) / 'runtime'
        staging.mkdir()
        for item in spec['archives']:
            archive = Path(temp) / item['name']
            with urllib.request.urlopen(item['url'], timeout=60) as response, archive.open('wb') as stream:
                shutil.copyfileobj(response, stream)
            verify(archive, item['sha256'])
            if archive.name.endswith('.zip'):
                with zipfile.ZipFile(archive) as bundle:
                    for info in bundle.infolist():
                        path = PurePosixPath(info.filename.replace('\\', '/'))
                        if path.is_absolute() or '..' in path.parts or ':' in str(path):
                            raise ValueError('Unsafe ZIP member')
                        if stat.S_ISLNK(info.external_attr >> 16):
                            raise ValueError('Unsafe ZIP symlink')
                    bundle.extractall(staging)
            else:
                with tarfile.open(archive) as bundle:
                    bundle.extractall(staging, filter='data')
        # Published SDKs contain a single enclosing directory.
        children = list(staging.iterdir())
        if len(children) == 1 and children[0].is_dir() and children[0].name not in ('lib', 'bin'):
            staging = children[0]
        files = {}
        for file in staging.rglob('*'):
            if file.is_file():
                with file.open('rb') as stream:
                    files[str(file.relative_to(staging))] = hashlib.file_digest(stream, 'sha256').hexdigest()
        if not any(Path(name).name.startswith(('libnemo_speech_asr_c.', 'nemo_speech_asr_c.')) for name in files):
            raise ValueError('Runtime archive lacks the native speech library')
        (staging / 'receipt.json').write_text(json.dumps({'spec_sha256': digest, 'files': files}, indent=2))
        if destination.exists():
            raise FileExistsError(f'Incomplete runtime at {destination}; remove it before retrying')
        staging.rename(destination)
    return destination


def fetch(artifact, cache):
    from huggingface_hub import hf_hub_download
    catalog = json.loads(Path(__file__).with_name('artifacts.json').read_text())
    if artifact not in catalog['files']:
        raise ValueError(f'Unknown Orukeet artifact: {artifact}')
    spec = catalog['files'][artifact]
    path = Path(hf_hub_download(spec.get('repo_id', catalog['repo_id']), spec['path'], revision=spec.get('revision', catalog['revision']),
                              local_dir=Path(cache) / 'models'))
    if path.stat().st_size != spec['size']:
        raise ValueError('Model size mismatch')
    verify(path, spec['sha256'])
    return path
