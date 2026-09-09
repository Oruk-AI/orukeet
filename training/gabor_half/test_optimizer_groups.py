import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from optimizer_groups import BASE_GROUPS, build_groups, parameter_group


def test_all_real_parameter_names_are_covered_without_frozen_rows():
    receipt = json.loads((Path(__file__).parent / 'results/r9-gradient-coverage.json').read_text())
    parameters = [(name, SimpleNamespace(requires_grad=True)) for name in receipt]
    rates = {name: 1e-6 for name in BASE_GROUPS}
    old = build_groups(parameters, rates)
    assert sum(len(g['params']) for g in old) == 651
    rates['conv_neighbors'] = 2e-5
    groups = build_groups(parameters, rates)
    assert {g['name']: len(g['params']) for g in groups} == {
        'conv_neighbors': 120, 'lower_encoder': 378, 'upper_encoder': 126,
        'other': 12, 'decoder': 9, 'joint': 6}
    assert len({id(p) for g in groups for p in g['params']}) == 651
    assert parameter_group('encoder.layers.0.conv.depthwise_conv.free_weight', True) == 'conv_neighbors'
    assert parameter_group('encoder.layers.23.norm_conv.weight', True) == 'upper_encoder'


def test_invalid_or_frozen_optimizer_membership_is_rejected():
    parameter = SimpleNamespace(requires_grad=True)
    rates = {name: 1e-6 for name in BASE_GROUPS}
    for invalid in ({'other': 1e-6}, dict(rates, decoder=0), dict(rates, other=float('nan'))):
        with pytest.raises(ValueError):
            build_groups([('decoder.weight', parameter)], invalid)
    with pytest.raises(ValueError):
        build_groups([('decoder.a', parameter), ('decoder.b', parameter)], rates)
    parameter.requires_grad = False
    with pytest.raises(ValueError):
        build_groups([('decoder.weight', parameter)], rates)
