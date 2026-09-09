"""Use an installation receipt without downloading anything during transcription."""
import argparse
import json
import sys
from pathlib import Path

from orukeet import Orukeet


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path, nargs='+')
    parser.add_argument('--installation', type=Path, default=Path('installation.json'))
    args = parser.parse_args()
    config = json.loads(args.installation.read_text(encoding='utf-8-sig'))
    with Orukeet(config['model'], config['runtime'], device=config['device']) as model:
        for path in args.audio:
            print(json.dumps({'file': str(path), **model.transcribe(path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
