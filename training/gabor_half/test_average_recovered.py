from collections import OrderedDict
import pytest
torch = pytest.importorskip('torch')

from average_recovered import average_state, average_states


def fixture():
    left = {'conv': torch.tensor([[1., 3.], [2., 6.]]), 'bias': torch.tensor([2.]),
            'running_mean': torch.tensor([4.]), 'count': torch.tensor(9)}
    right = {name: value.clone() for name, value in left.items()}
    right['conv'][1] = torch.tensor([6., 10.])
    right['bias'][0] = 6.
    return left, right


def test_analytic_rows_buffers_and_parameter_average():
    left, right = fixture()
    result = average_state(left, right, .25, {'conv', 'bias'}, {'conv': [0]})
    assert torch.equal(result['conv'], torch.tensor([[1., 3.], [3., 7.]]))
    assert torch.equal(result['bias'], torch.tensor([3.]))
    assert torch.equal(result['running_mean'], left['running_mean'])
    assert torch.equal(result['count'], left['count'])
    assert torch.equal(left['conv'][1], torch.tensor([2., 6.]))


@pytest.mark.parametrize('change', ['fixed', 'buffer', 'shape', 'missing', 'nonfinite'])
def test_incompatible_states_cannot_silently_be_averaged(change):
    left, right = fixture()
    if change == 'fixed': right['conv'][0, 0] += 1
    if change == 'buffer': right['count'] += 1
    if change == 'shape': right['bias'] = torch.ones(2)
    if change == 'missing': del right['bias']
    if change == 'nonfinite': right['bias'][0] = float('nan')
    with pytest.raises(ValueError):
        average_state(left, right, .5, {'conv', 'bias'}, {'conv': [0]})


def test_module_versions_survive_serialization_preparation():
    left, right = [OrderedDict(value) for value in fixture()]
    left._metadata = {'conv': {'version': 2}}
    right._metadata = {'conv': {'version': 2}}
    result = average_state(left, right, .5, {'conv', 'bias'}, {'conv': [0]})
    assert result._metadata == left._metadata and result._metadata is not left._metadata
    right._metadata['conv']['version'] = 1
    with pytest.raises(ValueError):
        average_state(left, right, .5, {'conv', 'bias'}, {'conv': [0]})


def test_three_parents_are_averaged_in_one_sum():
    left, middle = fixture()
    right = {name: value.clone() for name, value in middle.items()}
    right['conv'][1] += 4
    right['bias'] += 4
    result = average_states([left, middle, right], [.125, .375, .5],
                            {'conv', 'bias'}, {'conv': [0]})
    assert torch.equal(result['conv'], torch.tensor([[1., 3.], [7.5, 11.5]]))
    assert torch.equal(result['bias'], torch.tensor([7.5]))


def test_invalid_mixture_weights_are_rejected():
    left, right = fixture()
    for weights in [[1.], [0., 1.], [-.1, 1.1], [.5, .6], [float('nan'), .5]]:
        with pytest.raises(ValueError):
            average_states([left, right], weights, {'conv', 'bias'}, {'conv': [0]})


def test_only_stateless_training_augmentation_metadata_may_differ():
    left, right = [OrderedDict(value) for value in fixture()]
    left._metadata = {'conv': {'version': 2}, 'spec_augmentation.spec_augment': {'version': 1}}
    right._metadata = {'conv': {'version': 2}}
    result = average_state(left, right, .5, {'conv', 'bias'}, {'conv': [0]})
    assert result._metadata == left._metadata
    for state in [left, right]:
        state['spec_augmentation.spec_augment.state'] = torch.tensor([1.])
    with pytest.raises(ValueError):
        average_state(left, right, .5, {'conv', 'bias'}, {'conv': [0]})
