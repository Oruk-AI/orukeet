import copy
from pathlib import Path
import pytest
from prepare_slice_recovery import curriculum, leaves


def fixture():
    groups = []
    for language in ['bg', 'lv', 'fi', 'en']:
        groups.append({'type': 'group', 'weight': .2, 'tags': {'lang': language},
                       'input_cfg': [{'type': 'group', 'weight': .5, 'tags': {'src': source},
                                      'input_cfg': [{'type': 'nemo', 'manifest_filepath': f'{source}_{language}.jsonl',
                                                     'weight': 1.}]} for source in ['cv', 'fleurs']]})
    groups.append({'type': 'group', 'weight': .2, 'tags': {'cohort': 'accent_extension'},
                   'input_cfg': [{'type': 'nemo', 'manifest_filepath': 'accent.jsonl', 'weight': 1.}]})
    return groups


def test_exact_source_targets_and_original_membership_preserved():
    original = fixture()
    saved = copy.deepcopy(original)
    decision = {'status': 'fail', 'sets': {'cv_bg_dev_selection': {'delta_pp': .6},
                                         'fleurs_lv_dev_selection': {'delta_pp': .7},
                                         'cv_fi_dev_selection': {'delta_pp': .5}}}
    result, targets = curriculum(original, decision)
    assert targets == [('cv', 'bg'), ('fleurs', 'lv')]
    assert leaves(result) == leaves(original)
    assert original == saved
    assert result[0]['weight'] == .6
    assert sum(g['weight'] for g in result[1:3]) == .2
    assert leaves(result[1:3]) == {Path('cv_bg.jsonl'), Path('fleurs_lv.jsonl')}
    result[0]['input_cfg'][0]['weight'] = 999
    assert original == saved


def test_rejects_unjustified_curriculum():
    for decision in [{'status': 'pass', 'sets': {'cv_bg_dev_selection': {'delta_pp': .6}}},
                     {'status': 'fail', 'sets': {'cv_bg_dev_selection': {'delta_pp': .5}}}]:
        with pytest.raises(ValueError):
            curriculum(fixture(), decision)
