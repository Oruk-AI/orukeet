"""Replay captured stage-3 arguments in a prepared training environment.

Prints the command by default. --run starts training; it does not download data.
Use the historical root inside a container to preserve exact manifest hashes.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_ROOT = '/home/nathanroll/parakeet-ft'


def command(root, python=sys.executable):
    args=json.loads((ROOT/'evidence/training_argv.json').read_text())
    return [python,str(ROOT/'training/finetune_anchored.py'),
            *(arg.replace(HISTORICAL_ROOT,str(root)) for arg in args[2:])]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(HISTORICAL_ROOT))
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args();root=args.root.resolve();argv=command(root)
    print(shlex.join(argv),flush=True)
    if args.run:
        required=['Speech/examples/asr/speech_to_text_finetune.py',
                  'models/ft/parakeet-tdt-0.6b-v3-ft-cv-fleurs.nemo',
                  'manifests/balanced_global_20260905/input_cfg_with_accents.yaml',
                  'manifests/balanced_global_20260905/selection_dev.jsonl']
        for relative in required:
            if not (root/relative).is_file():raise FileNotFoundError(root/relative)
        output=root/'models/ft/stage3_anchored_accents_20260905.nemo'
        if output.exists():raise FileExistsError(f'Preserve the existing checkpoint: {output}')
        env=os.environ.copy();env['PARAKEET_ROOT']=str(root)
        subprocess.run(argv,env=env,check=True)


if __name__=='__main__':main()
