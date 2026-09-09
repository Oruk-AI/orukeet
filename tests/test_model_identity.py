"""Keep public model names attached to one audited source checkpoint."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def test_current_formats_share_the_audited_frozen_gabor_source():
    catalog = read('src/orukeet/artifacts.json')
    stages = read('release/model-stages.json')
    model = stages['gabor']
    audit = read(model['freeze_audit'])
    source = catalog['files']['source']['sha256']
    assert source == stages['canonical']['source_sha256'] == audit['candidate_sha256']
    assert audit['frozen_gabor_rows_exact'] == 12288
    assert set(catalog['files']) == {'source', 'q8', 'f16'}
    for format, path in [('q8', model['native_lineage']),
                         ('f16', model['native_f16']['lineage'])]:
        lineage = read(path)
        assert lineage['candidate_sha256'] == source
        assert lineage['gguf_sha256'] == catalog['files'][format]['sha256']
        assert lineage['gguf_bytes'] == catalog['files'][format]['size']
        assert lineage['fitted_rows_exact_after_f16_rounding'] == 12288


def test_release_checkpoint_matches_the_technical_report():
    catalog = read('src/orukeet/artifacts.json')
    model = read('report/model.json')
    validation = read('report/current-benchmark-validation.json')
    assert model['sha256'] == catalog['files']['source']['sha256']
    assert validation['models']['orukeet'] == model['sha256']
    for entry in catalog['files'].values():
        assert entry['source_sha256'] == model['sha256']


def test_current_cards_identify_the_same_model_as_the_installer():
    catalog = read('src/orukeet/artifacts.json')
    for path in ['README.md', 'MODEL_CARD.md', 'hub/README.md',
                 'launch/publication/github-README.md',
                 'launch/publication/github-MODEL_CARD.md',
                 'launch/publication/huggingface-README.md']:
        # Private preparation copies are absent from the clean public export.
        if path not in {'README.md', 'MODEL_CARD.md'} and not (ROOT / path).exists():
            continue
        card = (ROOT / path).read_text(encoding='utf-8')
        assert catalog['revision'] in card, path
        assert all(f['sha256'] in card for f in catalog['files'].values()), path
        assert '\x0c' not in card, path
        assert 'defaults remain unchanged' not in card, path
