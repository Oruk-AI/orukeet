#!/usr/bin/env python3
"""Apply predeclared weight averaging to a newly trained ASR checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_checkpoint import read
import torch


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    from nemo.collections.asr.models import ASRModel
    root = args.root
    protocol = json.loads(args.protocol.read_text())
    base = root / 'models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo'
    tuned = root / 'models/ft/stage3_anchored_accents_20260905.nemo'
    if sha(base) != protocol['baseline_sha256'] or sha(tuned) != protocol['new_training_parent_sha256']:
        raise ValueError('Parent checkpoint mismatch')
    a, at = read(base)
    b, bt = read(tuned)
    if a.keys() != b.keys() or at != bt:
        raise ValueError('Incompatible states or tokenizer artifacts')
    for k in a:
        if a[k].shape != b[k].shape or not torch.isfinite(a[k]).all() or not torch.isfinite(b[k]).all():
            raise ValueError('Invalid tensor: ' + k)
        if (not a[k].is_floating_point() or k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))) and not torch.equal(a[k], b[k]):
            raise ValueError('Incompatible fixed buffer: ' + k)
    model = ASRModel.restore_from(str(base), map_location='cpu').eval()
    for alpha in protocol['candidate_family']['fine_tuned_weight_fractions']:
        name = f'stage3_baseblend_a{round(alpha * 100):03d}_20260905'
        path = root / 'models/ft' / (name + '.nemo')
        if path.exists():
            raise FileExistsError(path)
        with torch.no_grad():
            state = model.state_dict()
            for k, v in state.items():
                v.copy_(a[k].lerp(b[k], alpha) if a[k].is_floating_point() else a[k])
        model.save_to(str(path))
        receipt = {'name': name, 'alpha': alpha, 'sha256': sha(path),
                   'base_sha256': protocol['baseline_sha256'],
                   'trained_parent_sha256': protocol['new_training_parent_sha256'],
                   'tokenizer_artifacts': at, 'model_path': str(path),
                   'operation': '(1-alpha)*stock + alpha*new-fine-tune',
                   'inference_architecture_unchanged': True}
        path.with_suffix('.lineage.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print('BLEND_COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
