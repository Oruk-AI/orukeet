"""Reuse recorded reference inference only under the identical native protocol."""
import json
from pathlib import Path
import shutil

from identity import sha

ORIGINAL_Q8 = '8967e04bd73fd8e88bccbfe97d0ec952beac73b1b3b3cba4b5c03316fcd8baa0'


def reuse_reference(source, destination, protocol, registry, *, full):
    source, destination = Path(source), Path(destination)
    metadata_path = source / 'results.json'
    original_metadata_sha = sha(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    if metadata['model_sha256'] != ORIGINAL_Q8:
        raise ValueError('Cached reference model identity changed')
    if metadata['normalizer'] != 'legacy-compatible-nfc-v1':
        raise ValueError('Cached reference normalizer changed')
    # The larger-suite scope names the candidate precision; it is descriptive.
    comparable = lambda value: {k: v for k, v in value.items() if k != 'scope'}
    if comparable(metadata['decoding']) != comparable(protocol):
        raise ValueError('Cached reference native protocol changed')
    records = registry['sets'] if full else {r['name']: r for r in registry['records']}
    if set(metadata['sets']) != set(records):
        raise ValueError('Cached reference membership changed')
    files = {p.name.removesuffix('_hypotheses.jsonl'): p
             for p in source.glob('*_hypotheses.jsonl')}
    if set(files) != set(records):
        raise ValueError('Cached reference prediction files changed')
    fingerprints = {}
    for name, record in records.items():
        expected = record['manifest_sha256' if full else 'sha256']
        if metadata['sets'][name]['manifest_sha256'] != expected:
            raise ValueError('Cached reference manifest changed: ' + name)
        if sum(1 for _ in files[name].open()) != record['rows']:
            raise ValueError('Cached reference row count changed: ' + name)
        fingerprints[name] = sha(files[name])
    destination.mkdir(parents=True, exist_ok=False)
    for name, path in files.items():
        target = destination / path.name
        shutil.copyfile(path, target)
        if sha(target) != fingerprints[name] or sha(path) != fingerprints[name]:
            raise ValueError('Cached reference changed while copying: ' + name)
    if sha(metadata_path) != original_metadata_sha:
        raise ValueError('Cached reference metadata changed while copying')
    receipt = {'source': str(source), 'source_metadata_sha256': original_metadata_sha,
               'model_sha256': ORIGINAL_Q8, 'prediction_sha256': fingerprints,
               'original_protocol': metadata['decoding'], 'comparison_protocol': protocol,
               'scope': 'Recorded reference inference reused without rerunning ASR. Model, '
                        'normalizer, manifests, row counts and complete native protocol match; '
                        'only the descriptive candidate-precision scope may differ. Every '
                        'prediction file is copied byte-for-byte and rehashed.'}
    (destination.parent / 'reference-reuse.json').write_text(json.dumps(receipt, indent=2) + '\n')
    metadata['decoding'] = protocol
    (destination / 'results.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return receipt
