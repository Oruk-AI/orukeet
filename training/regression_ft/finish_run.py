"""Run the authorized audits and private backup after training completes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    a = p.parse_args()
    experiment = a.experiment.resolve()
    root, code = experiment.parent, experiment / 'code'
    launch = json.loads((experiment / 'launch-v1.json').read_text())
    args = launch['argv']
    value = lambda flag: args[args.index(flag) + 1]
    run = Path(value('--output'))
    candidate = Path(value('--checkpoint-dir')) / 'orukeet-regression-ft.nemo'
    audit = experiment / 'export-audit.json'
    evaluation = experiment / 'evaluation'
    state_path = experiment / 'finish-progress.json'

    def record(stage, **fields):
        data = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), **fields)
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=2) + '\n')
        temp.replace(state_path)
        print(json.dumps(data), flush=True)

    def alive(pid):
        try:
            return Path('/proc', str(pid), 'stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
        except FileNotFoundError:
            return False

    def execute(stage, argv):
        record(stage, argv=argv)
        with (experiment / ('finish-' + stage + '.log')).open('ab') as log:
            subprocess.run(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        record('waiting_for_training_and_diagnostic_preparation')
        while not (run / 'complete.json').exists() or alive(launch['pid']):
            if not alive(launch['pid']) and not (run / 'complete.json').exists():
                raise RuntimeError('Training exited without complete.json; inspect train-v1.log')
            time.sleep(10)
        prepared = experiment / 'diagnostic/prepared.json'
        while not prepared.exists():
            diagnostic = json.loads((experiment / 'diagnostic-launch.json').read_text())
            if not alive(diagnostic['pid']):
                raise RuntimeError('Diagnostic preparation exited before completion')
            time.sleep(10)
        assert json.loads(prepared.read_text())['status'] == 'complete'
        execute('repair-export', [sys.executable, str(code / 'repair_export.py'),
                                 '--parent', value('--parent'), '--candidate', str(candidate),
                                 '--complete', str(run / 'complete.json'),
                                 '--output', str(experiment / 'export-metadata')])
        execute('audit', [sys.executable, str(code / 'audit_export.py'), '--parent', value('--parent'),
                         '--candidate', str(candidate), '--plan', value('--plan'), '--fits', value('--fits'),
                         '--gabor-code', value('--gabor-code'), '--parameter-inventory', str(run / 'gradient-coverage.json'),
                         '--output', str(audit)])
        execute('evaluation', [sys.executable, str(code / 'evaluate.py'),
                              '--manifest', str(experiment / 'diagnostic/manifest.jsonl'),
                              '--base', str(root / 'models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo'),
                              '--parent', value('--parent'), '--candidate', str(candidate),
                              '--metric-code', str(root / 'unseen_20260907/code'), '--output', str(evaluation)])
        execute('summary', [sys.executable, str(code / 'summarize.py'),
                           '--manifest', str(experiment / 'diagnostic/manifest.jsonl'),
                           '--evaluation', str(evaluation), '--output', str(evaluation)])
        execute('archive', [sys.executable, str(code / 'archive_run.py'), '--experiment', str(experiment),
                           '--run', str(run), '--candidate', str(candidate),
                           '--optimizer', str(candidate.parent / 'resume.pt'), '--audit', str(audit),
                           '--evaluation', str(evaluation), '--prefix', 'experiments/regression-ft-20260907/r1'])
        record('complete', private_archive=str(experiment / 'private-archive.json'))
    except Exception as error:
        record('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
