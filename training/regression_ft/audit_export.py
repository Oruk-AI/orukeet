"""Audit a portable candidate directly from the two on-disk NeMo archives."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    with tarfile.open(path) as archive:
        member = next(m for m in archive.getmembers() if m.name.endswith('model_weights.ckpt'))
        state = torch.load(io.BytesIO(archive.extractfile(member).read()), map_location='cpu', weights_only=True)
        assets = sorted(hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                        for m in archive.getmembers() if m.isfile()
                        and any(m.name.endswith(s) for s in ['.model', '.vocab', 'vocab.txt']))
    return state, assets


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--fits', type=Path, required=True)
    p.add_argument('--gabor-code', type=Path, required=True)
    p.add_argument('--parameter-inventory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.gabor_code))
    from fit import reconstruct, SOURCE_SHA
    torch.set_num_threads(8)
    plan = json.loads(a.plan.read_text())
    assert sha(a.parent) == plan['parent_sha256']
    before, ba = load(a.parent)
    after, aa = load(a.candidate)
    assert ba == aa and before.keys() == after.keys()
    fit = json.loads(a.fits.read_text())
    assert fit['source_sha256'] == SOURCE_SHA and fit['selected'] == 12288 and fit['kernels'] == 24576
    selected, fixed_rows = 0, {}
    for name in sorted({r['name'] for r in fit['records']}):
        rows = sorted((r for r in fit['records'] if r['name'] == name and r['selected']), key=lambda r: r['channel'])
        indices = [r['channel'] for r in rows]
        if not indices:
            continue
        expected = torch.from_numpy(reconstruct([r['params'] for r in rows]).astype('float32')).unsqueeze(1)
        assert torch.equal(after[name][indices], before[name][indices]), name
        assert torch.equal(after[name][indices], expected), name
        selected += len(indices)
        fixed_rows[name] = indices
    assert selected == 12288
    changed, norms = [], {}
    bn_suffixes = ('running_mean', 'running_var', 'num_batches_tracked')
    signal_buffers = {'preprocessor.featurizer.window', 'preprocessor.featurizer.fb'}
    fixed_buffers = {n for n in before if n.endswith(bn_suffixes)} | signal_buffers
    assert fixed_buffers <= before.keys()
    inventory = json.loads(a.parameter_inventory.read_text())
    parameter_names = {n.replace('.parametrizations.weight.original', '.weight') for n in inventory}
    assert len(parameter_names) == len(inventory) == 651
    assert all(all(v.values()) for v in inventory.values())
    assert parameter_names == before.keys() - fixed_buffers, 'Parameter/buffer inventory mismatch'
    for name, value in after.items():
        assert value.shape == before[name].shape and value.dtype == before[name].dtype, name
        if value.is_floating_point():
            assert torch.isfinite(value).all(), name
        if name in fixed_buffers:
            assert torch.equal(value, before[name]), name
            continue
        # Window, mel filterbank, and BN statistics are non-parameter buffers.
        # Every tensor in the verified parameter inventory must actually update.
        assert name in parameter_names
        assert not torch.equal(value, before[name]), 'Unchanged trainable tensor: ' + name
        changed.append(name)
        delta = value.float() - before[name].float()
        norms[name] = dict(l2=float(torch.linalg.vector_norm(delta)), max_abs=float(delta.abs().max()))
    result = dict(status='pass', parent_sha256=sha(a.parent), candidate_sha256=sha(a.candidate),
                  plan_sha256=sha(a.plan), script_sha256=sha(__file__),
                  frozen_gabor_rows_exact=selected, frozen_coefficients_exact=selected * 9,
                  gabor_values_match_parent_and_original_functions=True,
                  tokenizer_assets_equal=True, tensor_keys_shapes_dtypes_equal=True,
                  all_floating_tensors_finite=True, batchnorm_buffers_unchanged=True,
                  signal_preprocessing_buffers_unchanged=True,
                  parameter_inventory_sha256=sha(a.parameter_inventory), fixed_buffer_tensors=len(fixed_buffers),
                  changed_parameter_tensors=len(changed), every_other_parameter_tensor_changed=True,
                  tensor_update_norms=norms)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'tensor_update_norms'}))


if __name__ == '__main__':
    main()
