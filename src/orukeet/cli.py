import argparse
import json
import sys
from pathlib import Path

from .install import fetch, install_runtime, resolve_device
from .model import Orukeet


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description='Orukeet multilingual speech recognition with frozen Gabor kernels')
    sub = parser.add_subparsers(dest='command', required=True)
    install = sub.add_parser('install', help='Download and verify Q8 model and native runtime')
    install.add_argument('--device', choices=['auto', 'cpu', 'metal', 'cuda', 'vulkan'], default='auto')
    install.add_argument('--cache', type=Path, default=Path.home() / '.cache/orukeet')
    install.add_argument('--output', type=Path, help='Write a UTF-8 installation receipt for offline use')
    download = sub.add_parser('fetch', help='Fetch an exact checkpoint for research or fine-tuning')
    catalog = json.loads(Path(__file__).with_name('artifacts.json').read_text())
    download.add_argument('artifact', choices=sorted(catalog['files']))
    download.add_argument('--cache', type=Path, default=Path.home() / '.cache/orukeet')
    transcribe = sub.add_parser('transcribe', help='Transcribe local files; performs no downloads')
    transcribe.add_argument('audio', nargs='+', type=Path)
    transcribe.add_argument('--model', type=Path, required=True)
    transcribe.add_argument('--runtime', type=Path, required=True)
    transcribe.add_argument('--device', choices=['cpu', 'metal', 'cuda', 'vulkan'], default='cpu')
    transcribe.add_argument('--sha256', help='Expected SHA-256 of a custom fine-tuned GGUF')
    args = parser.parse_args()
    if args.command == 'install':
        device = resolve_device(args.device)
        runtime = install_runtime(device, args.cache)
        model = fetch('q8', args.cache)
        receipt = json.dumps({'model': str(model.resolve()), 'runtime': str(runtime.resolve()), 'device': device}, indent=2)
        if args.output:
            args.output.write_text(receipt + '\n', encoding='utf-8')
        print(receipt)
    elif args.command == 'fetch':
        print(fetch(args.artifact, args.cache))
    else:
        with Orukeet(args.model, args.runtime, args.device, expected_sha256=args.sha256) as model:
            for path in args.audio:
                print(json.dumps(dict(file=str(path), **model.transcribe(path)), ensure_ascii=False))
