"""Assign every trainable tensor to an explicit recovery learning rate."""
import math


BASE_GROUPS = {'lower_encoder', 'upper_encoder', 'other', 'decoder', 'joint'}


def parameter_group(name, conv_neighbors=False):
    if name.startswith('encoder.layers.'):
        layer = int(name.split('.')[2])
        if not 0 <= layer < 24:
            raise ValueError('Expected a 24-layer encoder')
        if conv_neighbors and name.split('.')[3] == 'conv':
            return 'conv_neighbors'
        return 'upper_encoder' if layer >= 18 else 'lower_encoder'
    if name.startswith('decoder.'):
        return 'decoder'
    if name.startswith('joint.'):
        return 'joint'
    return 'other'


def build_groups(named_parameters, rates):
    if set(rates) not in (BASE_GROUPS, BASE_GROUPS | {'conv_neighbors'}):
        raise ValueError('Recovery rates must cover every parameter group')
    if any(not math.isfinite(v) or v <= 0 for v in rates.values()):
        raise ValueError('Every group needs a finite positive learning rate')
    groups = {name: [] for name in rates}
    seen = set()
    for name, parameter in named_parameters:
        if not parameter.requires_grad or id(parameter) in seen:
            raise ValueError('Expected unique trainable parameters; fixed taps belong in buffers')
        seen.add(id(parameter))
        groups[parameter_group(name, 'conv_neighbors' in rates)].append(parameter)
    return [{'params': parameters, 'lr': rates[name], 'name': name}
            for name, parameters in groups.items() if parameters]
