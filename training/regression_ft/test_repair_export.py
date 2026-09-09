"""Check the metadata repair on an archive with opaque, immutable weight bytes."""
import io
import tarfile

import yaml

from repair_export import inspect, repair, sha


def test_repair_preserves_weights_and_is_idempotent(tmp_path):
    def archive(path, config):
        with tarfile.open(path, 'w') as out:
            for name, data in [('model_config.yaml', yaml.safe_dump(config).encode()),
                               ('model_weights.ckpt', b'\x00opaque weight bytes\xff'),
                               ('tokenizer.model', b'original tokenizer')]:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                out.addfile(member, io.BytesIO(data))
    parent, candidate = tmp_path / 'parent.nemo', tmp_path / 'candidate.nemo'
    validation = dict(manifest_filepath=None, sample_rate=16000, use_start_end_token=False, batch_size=16)
    archive(parent, dict(validation_ds=validation, decoding={'strategy': 'greedy_batch'}))
    archive(candidate, dict(validation_ds=None, decoding={'strategy': 'greedy_batch'}))
    complete = dict(audit={'checkpoint_sha256': sha(candidate)}, run={'parent_sha256': sha(parent)})
    before = inspect(candidate)[1]
    first = repair(parent, candidate, complete, tmp_path / 'metadata')
    assert first == repair(parent, candidate, complete, tmp_path / 'metadata')
    assert inspect(candidate)[1] == before
    assert yaml.safe_load(inspect(candidate)[0])['validation_ds'] == validation
    assert sha(tmp_path / 'candidate.training-export.nemo') == complete['audit']['checkpoint_sha256']
