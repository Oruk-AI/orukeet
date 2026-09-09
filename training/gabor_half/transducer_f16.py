"""Keep the seven Parakeet TDT transducer matrices in F16 during a Q8 export."""
import hashlib
import io
from pathlib import Path
import runpy
import sys
import tarfile

from identity import sha

EXPECTED = {
    'decoder.prediction.dec_rnn.lstm.ih_l0.weight',
    'decoder.prediction.dec_rnn.lstm.hh_l0.weight',
    'decoder.prediction.dec_rnn.lstm.ih_l1.weight',
    'decoder.prediction.dec_rnn.lstm.hh_l1.weight',
    'joint.enc.weight', 'joint.pred.weight', 'joint.joint_net.2.weight',
}


def convert(source, destination, converter):
    sys.path.insert(0, str(converter))
    from conversion import asr
    from gguf import GGMLQuantizationType as Type
    original = asr._pick_dtype
    previous_argv = sys.argv
    changed = set()

    def pick(name, shape, linear_qtype, default_qtype):
        chosen, reason = original(name, shape, linear_qtype, default_qtype)
        if name.startswith(('decoder.', 'joint.')) and chosen == Type.Q8_0:
            changed.add(name)
            return Type.F16, 'declared transducer precision'
        return chosen, reason

    try:
        asr._pick_dtype = pick
        sys.argv = [str(converter / 'convert_model.py'), str(source),
                    '--outfile', str(destination), '--outtype', 'q8_0']
        entry = runpy.run_path(str(converter / 'convert_model.py'), run_name='orukeet_private_conversion')
        if entry['main']() != 0:
            raise RuntimeError('Upstream converter failed')
    finally:
        asr._pick_dtype = original
        sys.argv = previous_argv
    if changed != EXPECTED:
        raise ValueError('Unexpected transducer matrix population: ' + repr(sorted(changed)))
    return asr


def audit(source, baseline, candidate, converter_module):
    import numpy as np
    import torch
    from gguf import GGUFReader, GGMLQuantizationType as Type
    before = {t.name: t for t in GGUFReader(str(baseline)).tensors}
    after = {t.name: t for t in GGUFReader(str(candidate)).tensors}
    if before.keys() != after.keys():
        raise ValueError('Mixed export changed tensor membership')
    expected_changes = {n for n, t in before.items()
                        if n.startswith(('decoder.', 'joint.')) and t.tensor_type == Type.Q8_0}
    if expected_changes != EXPECTED:
        raise ValueError('Baseline has an unexpected transducer layout')
    with tarfile.open(source) as archive:
        members = [m for m in archive.getmembers() if m.name.endswith('model_weights.ckpt')]
        if len(members) != 1:
            raise ValueError('Expected one source state dictionary')
        state = torch.load(io.BytesIO(archive.extractfile(members[0]).read()),
                           map_location='cpu', weights_only=True)
    source_heads = {}
    for name, tensor in state.items():
        mapped = converter_module.remap_for_rnnt(name)
        if mapped in EXPECTED:
            source_heads[mapped] = converter_module.reshape_for_emission(
                mapped, tensor.detach().cpu().float().numpy()).astype(np.float16)
    if source_heads.keys() != EXPECTED:
        raise ValueError('Source is missing a declared transducer matrix')
    changes = []
    for name, old in before.items():
        new = after[name]
        if not np.array_equal(old.shape, new.shape):
            raise ValueError('Mixed export changed tensor shape: ' + name)
        if name in EXPECTED:
            expected = source_heads[name]
            if new.tensor_type != Type.F16 or new.data.tobytes() != expected.tobytes():
                raise ValueError('F16 transducer differs from rounded source: ' + name)
            changes.append({'name': name, 'old_type': 'Q8_0', 'new_type': 'F16',
                            'scalar_parameters': int(expected.size),
                            'source_f16_sha256': hashlib.sha256(expected.tobytes()).hexdigest()})
        elif old.tensor_type != new.tensor_type or old.data.tobytes() != new.data.tobytes():
            raise ValueError('Undeclared tensor change: ' + name)
    return {'baseline_q8_sha256': sha(baseline), 'changed_tensors': changes,
            'unchanged_tensor_count': len(before) - len(changes),
            'unchanged_tensors_byte_exact': True, 'source_f16_tensors_byte_exact': True,
            'size_increase_bytes': Path(candidate).stat().st_size - Path(baseline).stat().st_size,
            'policy': 'Only the seven previously Q8 predictor/LSTM and joint linear matrices '
                      'use F16 directly from the same trained source. All other tensor '
                      'payloads and types remain byte-identical to the baseline Q8 export.'}
