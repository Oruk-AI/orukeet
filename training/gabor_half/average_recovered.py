"""Average trained parameters while preserving audited Gabor rows and buffers."""
import argparse
from collections import OrderedDict
import copy
import io
import json
import math
from pathlib import Path
import tarfile

import torch
import yaml

from identity import sha, SOURCE_SHA


def comparable_metadata(state):
    metadata = getattr(state, '_metadata', None)
    if metadata is None:
        return None
    result = dict(metadata)
    # R2 exported this stateless training augmentation child; later runs
    # disabled it. Its presence does not describe any stored model tensor.
    training_child = 'spec_augmentation.spec_augment'
    if training_child in result:
        if result[training_child] != {'version': 1} or any(
                name == training_child or name.startswith(training_child + '.') for name in state):
            raise ValueError('Augmentation metadata is not the known stateless training child')
        del result[training_child]
    return result


def average_states(states, weights, parameters, fixed_rows):
    if len(states) < 2 or len(states) != len(weights):
        raise ValueError('Expected at least two states and one weight per state')
    if any(not math.isfinite(w) or w <= 0 for w in weights) or abs(math.fsum(weights) - 1) > 1e-12:
        raise ValueError('Expected a strict convex average')
    left = states[0]
    if any(left.keys() != state.keys() for state in states) or not parameters <= left.keys():
        raise ValueError('Tensor keys differ or a parameter is missing')
    if not fixed_rows.keys() <= parameters:
        raise ValueError('Fixed rows must belong to materialized parameter tensors')
    metadata = getattr(left, '_metadata', None)
    if any(comparable_metadata(left) != comparable_metadata(state) for state in states):
        raise ValueError('Module state versions differ')
    result = OrderedDict()
    if metadata is not None:
        result._metadata = copy.deepcopy(metadata)
    for name, value in left.items():
        values = [state[name] for state in states]
        if any(value.shape != other.shape or value.dtype != other.dtype for other in values):
            raise ValueError('Tensor layout differs: ' + name)
        if any(not torch.isfinite(other).all() for other in values):
            raise ValueError('Nonfinite input: ' + name)
        if name not in parameters:
            if any(not torch.equal(value, other) for other in values):
                raise ValueError('Non-parameter buffer differs: ' + name)
            result[name] = value.clone()
            continue
        if not value.is_floating_point():
            raise ValueError('Non-floating parameter: ' + name)
        indices = fixed_rows.get(name)
        if indices is not None and any(not torch.equal(value[indices], other[indices]) for other in values):
            raise ValueError('Fixed Gabor rows differ: ' + name)
        accumulator = torch.zeros_like(value, dtype=torch.float64)
        for other, weight in zip(values, weights):
            accumulator.add_(other.double(), alpha=weight)
        averaged = accumulator.to(value.dtype)
        if indices is not None:
            averaged[indices] = value[indices]
        if not torch.isfinite(averaged).all():
            raise ValueError('Nonfinite output: ' + name)
        result[name] = averaged
    return result


def average_state(left, right, alpha, parameters, fixed_rows):
    return average_states([left, right], [1 - alpha, alpha], parameters, fixed_rows)


def load(path):
    with tarfile.open(path) as archive:
        weights = [m for m in archive if m.name.endswith('model_weights.ckpt')]
        if len(weights) != 1:
            raise ValueError('Expected one weight dictionary')
        state = torch.load(io.BytesIO(archive.extractfile(weights[0]).read()),
                           map_location='cpu', weights_only=True)
        assets = sorted(archive.extractfile(m).read() for m in archive.getmembers()
                        if m.isfile() and m.name.endswith(('.vocab', '.model', 'vocab.txt')))
        config_members = [m for m in archive.getmembers() if m.name.endswith('model_config.yaml')]
        if len(config_members) != 1:
            raise ValueError('Expected one model configuration')
        config = yaml.safe_load(archive.extractfile(config_members[0]).read())
        inference_config = {key: config.get(key) for key in
                            ('target', 'sample_rate', 'preprocessor', 'encoder', 'decoder', 'joint', 'decoding')}
    return state, assets, inference_config


def main():
    parser = argparse.ArgumentParser()
    for key in ('parents', 'audits'):
        parser.add_argument('--' + key, type=Path, nargs='+', required=True)
    parser.add_argument('--weights', type=float, nargs='+', required=True)
    for key in ('fits', 'coverage', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    assert len(args.parents) == len(args.audits) == len(args.weights) >= 2
    assert not args.output.exists() and not args.output.with_suffix('.json').exists()
    identities = []
    for path, audit_path in zip(args.parents, args.audits):
        audit = json.loads(audit_path.read_text())
        digest = sha(path)
        assert audit['status'] == 'pass' and audit['original_sha256'] == SOURCE_SHA
        assert audit['candidate_sha256'] == digest and audit['frozen_gabor_rows_exact'] == 12288
        identities.append(digest)
    assert len(set(identities)) == len(identities)
    assert sha(args.fits) == '44ef0eb45fdd122a3900c97f1faf4ffedf567f27e9ad535f24687c36a8c7f704'
    fits = json.loads(args.fits.read_text())
    fixed = {}
    for row in fits['records']:
        if row['selected']:
            fixed.setdefault(row['name'], []).append(row['channel'])
    assert sum(map(len, fixed.values())) == 12288 and len(fixed) == 24
    coverage = json.loads(args.coverage.read_text())
    assert len(coverage) == 651 and all(v['trainable'] and v['has_gradient'] and v['finite'] for v in coverage.values())
    parameters = {name.replace('.parametrizations.weight.original', '.weight') for name in coverage}
    assert len(parameters) == 651
    loaded = [load(path) for path in args.parents]
    states = [item[0] for item in loaded]
    assert loaded[0][1] and all(item[1:] == loaded[0][1:] for item in loaded)
    from fit import reconstruct
    import numpy as np
    for name, indices in fixed.items():
        rows = [r for r in fits['records'] if r['name'] == name and r['selected']]
        assert indices == [r['channel'] for r in rows]
        expected = torch.from_numpy(reconstruct([r['params'] for r in rows]).astype(np.float32)).unsqueeze(1)
        assert all(torch.equal(state[name][indices], expected) for state in states)
    result = average_states(states, args.weights, parameters, fixed)
    data = io.BytesIO()
    torch.save(result, data)
    data.seek(0)
    with tarfile.open(args.parents[0]) as source, args.output.open('xb') as output:
        with tarfile.open(fileobj=output, mode='w') as destination:
            for member in source:
                if member.name.endswith('model_weights.ckpt'):
                    member.size = data.getbuffer().nbytes
                    destination.addfile(member, data)
                else:
                    destination.addfile(member, source.extractfile(member) if member.isfile() else None)
    receipt = {
        'method': 'Convex average of audited trained states; not an inference ensemble',
        'source_sha256': SOURCE_SHA, 'parents_sha256': identities,
        'weights': args.weights, 'output_sha256': sha(args.output),
        'fits_sha256': sha(args.fits), 'coverage_sha256': sha(args.coverage),
        'frozen_rows_copied_exactly': 12288, 'parameter_tensors_averaged': len(parameters),
        'unchanged_buffers': len(result) - len(parameters), 'tokenizer_assets_equal': True,
        'inference_configuration_equal': True,
        'state_metadata_policy': 'Preserve the exporting parent metadata; require matching common module versions. '
                                 'Allow presence of the known stateless training augmentation child only when it owns no stored tensor.',
        'stateless_augmentation_metadata_present': [
            'spec_augmentation.spec_augment' in (getattr(state, '_metadata', None) or {}) for state in states],
        'precision': 'One ordered float64 weighted sum, rounded once to the original storage dtype; fixed rows copied',
        'scope': 'New experimental candidate; independent checkpoint and accuracy audits still required'}
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
