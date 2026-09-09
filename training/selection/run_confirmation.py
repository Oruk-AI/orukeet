#!/usr/bin/env python3
"""Run the sealed gate only for the preselected eligible candidate."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    args = p.parse_args()
    root = args.root
    os.chdir(root)
    log = (root/'logs/goal_v2_development.log').read_text(errors='replace')
    if 'DEVELOPMENT_EVALUATION_COMPLETE' not in log:
        raise RuntimeError('Development evaluations have not completed')
    decision_path = root/'eval/goal_v2/development_decision.json'
    if not decision_path.exists():
        subprocess.run(['python3','goal_v2/select_development.py','--root',str(root)],check=True)
    decision = json.loads(decision_path.read_text())
    if decision['status'] != 'selected':
        raise RuntimeError('No eligible candidate; confirmation stays unopened')
    registry_path = root/'manifests/goal_v2_confirmation/sealed_metric_registry.json'
    registry = json.loads(registry_path.read_text())
    protocol_path = root/'goal_v2/protocol.json'
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if registry['protocol_sha256'] != protocol_hash or decision['protocol_sha256'] != protocol_hash:
        raise RuntimeError('Protocol changed after selection or sealing')
    manifests = []
    for name,entry in registry['sets'].items():
        path = Path(entry['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['manifest_sha256']:
            raise RuntimeError('Manifest changed: '+name)
        manifests.append(str(path))
    exposure = {'opened_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'selected_model_path':decision['selected']['model_path'],
        'selected_model_sha256':decision['selected']['model_sha256'],
        'protocol_sha256':protocol_hash,'registry_sha256':hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        'development_decision_sha256':hashlib.sha256(decision_path.read_bytes()).hexdigest(),
        'policy':'This confirmation is consumed by this selected candidate; never reuse it as a fresh gate after outcome-informed model changes.'}
    with (root/'eval/goal_v2/confirmation_exposure_receipt.json').open('x') as f:
        json.dump(exposure,f,indent=2)
        f.write('\n')
    models = [('base',str(root/'models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo')),
              ('selected',decision['selected']['model_path'])]
    for label,model in models:
        env = dict(os.environ,NEMO_NAME='goal-v2-confirm-'+label)
        subprocess.run(['./nemo.sh','python','evaluate_20260905.py','--model',model,
                        '--output',str(root/'eval/goal_v2'/(label+'_confirmation'))]+manifests,env=env,check=True)
    env = dict(os.environ,NEMO_NAME='goal-v2-confirm-statistics')
    subprocess.run(['./nemo.sh','env','OPENBLAS_NUM_THREADS=4','OMP_NUM_THREADS=4',
                    'python','goal_v2/compare_confirmation.py','--root',str(root)],env=env,check=True)
    print('GOAL_V2_CONFIRMATION_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
