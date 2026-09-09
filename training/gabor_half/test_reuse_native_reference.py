import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from reuse_native_reference import ORIGINAL_Q8, reuse_reference


def fixture(tmp_path):
    source = tmp_path / 'saved'
    source.mkdir()
    protocol = {'device': 'cuda', 'decoder': 'fixed', 'runtime_library_sha256': {'runtime.so': 'runtime'},
                'binding_sha256': 'binding', 'audio_path_sha256': 'audio',
                'evaluation_script_sha256': 'evaluator', 'scope': 'old candidate precision'}
    metadata = {'model_sha256': ORIGINAL_Q8, 'normalizer': 'legacy-compatible-nfc-v1',
                'decoding': protocol, 'sets': {'slice': {'manifest_sha256': 'manifest'}}}
    (source / 'results.json').write_text(json.dumps(metadata))
    (source / 'slice_hypotheses.jsonl').write_text('{"synthetic_record":true}\n')
    registry = {'sets': {'slice': {'manifest_sha256': 'manifest', 'rows': 1}}}
    return source, protocol, metadata, registry


def test_identical_predictions_and_metadata_scope_provenance(tmp_path):
    source, protocol, metadata, registry = fixture(tmp_path)
    target = tmp_path / 'new' / 'original'
    current = dict(protocol, scope='new candidate precision')
    receipt = reuse_reference(source, target, current, registry, full=True)
    assert (target / 'slice_hypotheses.jsonl').read_bytes() == (source / 'slice_hypotheses.jsonl').read_bytes()
    assert json.loads((source / 'results.json').read_text()) == metadata
    assert receipt['original_protocol'] == protocol and receipt['comparison_protocol'] == current


@pytest.mark.parametrize('field,value', [('device', 'cpu'), ('decoder', 'changed'),
    ('audio_path_sha256', 'other'), ('runtime_library_sha256', {'runtime.so': 'other'})])
def test_rejects_changed_execution_before_creating_output(tmp_path, field, value):
    source, protocol, _, registry = fixture(tmp_path)
    target = tmp_path / 'new'
    with pytest.raises(ValueError, match='protocol'):
        reuse_reference(source, target, dict(protocol, **{field: value}), registry, full=True)
    assert not target.exists()


@pytest.mark.parametrize('change', ['model', 'manifest', 'rows', 'extra_file'])
def test_rejects_wrong_reference_or_membership(tmp_path, change):
    source, protocol, metadata, registry = fixture(tmp_path)
    if change == 'model': metadata['model_sha256'] = 'different-model'
    if change == 'manifest': metadata['sets']['slice']['manifest_sha256'] = 'different-speech'
    if change == 'rows': registry['sets']['slice']['rows'] = 2
    if change == 'extra_file': (source / 'extra_hypotheses.jsonl').write_text('{}\n')
    (source / 'results.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        reuse_reference(source, tmp_path / 'new', protocol, registry, full=True)


def test_development_registry_schema(tmp_path):
    source, protocol, _, _ = fixture(tmp_path)
    registry = {'records': [{'name': 'slice', 'sha256': 'manifest', 'rows': 1}]}
    receipt = reuse_reference(source, tmp_path / 'new', protocol, registry, full=False)
    assert set(receipt['prediction_sha256']) == {'slice'}
