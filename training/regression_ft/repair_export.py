"""Restore NeMo's inference metadata without changing any model weight bytes."""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

import yaml


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def inspect(path):
    members, config = {}, None
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            stream = archive.extractfile(member)
            if member.name.endswith('model_config.yaml'):
                config = stream.read()
            else:
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(1 << 20), b''):
                    h.update(block)
                members[member.name] = dict(bytes=member.size, sha256=h.hexdigest())
    assert config is not None
    return config, members


def repair(parent, candidate, complete, output):
    parent, candidate, output = Path(parent), Path(candidate), Path(output)
    original = candidate.with_name(candidate.stem + '.training-export.nemo')
    receipt_path = output / 'repair.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt['source_sha256'] == complete['audit']['checkpoint_sha256'] == sha(original)
        assert receipt['candidate_sha256'] == sha(candidate)
        assert receipt['parent_sha256'] == complete['run']['parent_sha256'] == sha(parent)
        assert receipt['all_non_config_members_byte_identical']
        return receipt
    assert sha(candidate) == complete['audit']['checkpoint_sha256']
    assert sha(parent) == complete['run']['parent_sha256']
    parent_text, _ = inspect(parent)
    source_text, source_members = inspect(candidate)
    parent_cfg, source_cfg = yaml.safe_load(parent_text), yaml.safe_load(source_text)
    validation = parent_cfg['validation_ds']
    assert isinstance(validation, dict) and validation['manifest_filepath'] is None
    assert validation['sample_rate'] == 16000 and validation['use_start_end_token'] is False
    assert source_cfg['validation_ds'] is None
    target_cfg = copy.deepcopy(source_cfg)
    target_cfg['validation_ds'] = validation
    target_text = yaml.safe_dump(target_cfg, sort_keys=False, allow_unicode=True).encode()
    temporary = candidate.with_suffix('.repair.tmp')
    with tarfile.open(candidate) as source, tarfile.open(temporary, 'w') as target:
        for member in source.getmembers():
            if member.name.endswith('model_config.yaml'):
                replacement = copy.copy(member)
                replacement.size = len(target_text)
                target.addfile(replacement, io.BytesIO(target_text))
            else:
                target.addfile(member, source.extractfile(member) if member.isfile() else None)
    checked_text, checked_members = inspect(temporary)
    checked_cfg = yaml.safe_load(checked_text)
    assert checked_members == source_members
    assert checked_cfg == target_cfg
    checked_cfg['validation_ds'] = None
    assert checked_cfg == source_cfg
    receipt = dict(status='pass', source_sha256=sha(candidate), candidate_sha256=sha(temporary),
                   parent_sha256=sha(parent), script_sha256=sha(__file__),
                   source_path=str(original), candidate_path=str(candidate),
                   change='Restore parent validation_ds metadata required by NeMo transcribe; manifest remains null.',
                   all_non_config_members_byte_identical=True, all_other_config_values_equal=True,
                   non_config_members=source_members)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'training-model-config.yaml').write_bytes(source_text)
    (output / 'inference-model-config.yaml').write_bytes(target_text)
    if original.exists():
        assert sha(original) == receipt['source_sha256']
    else:
        os.link(candidate, original)
    os.replace(temporary, candidate)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def verify_training_identity(experiment, complete, candidate_sha):
    receipt = json.loads((Path(experiment) / 'export-metadata/repair.json').read_text())
    assert receipt['status'] == 'pass'
    assert receipt['source_sha256'] == complete['audit']['checkpoint_sha256']
    assert receipt['candidate_sha256'] == candidate_sha
    assert receipt['parent_sha256'] == complete['run']['parent_sha256']
    assert receipt['all_non_config_members_byte_identical'] and receipt['all_other_config_values_equal']


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--complete', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(repair(a.parent, a.candidate, json.loads(a.complete.read_text()), a.output)), flush=True)
