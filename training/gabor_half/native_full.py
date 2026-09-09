"""Paired native deployment regression for an already source-qualified candidate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from identity import sha, SOURCE_SHA
from metric_qualification import qualify
from reuse_native_reference import reuse_reference

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'evaluation'))
from compare_oruk_export import compare
import compare_oruk_export


def runtime_hashes(runtime):
    files = {path.name: sha(path) for path in sorted((runtime / 'lib').iterdir())
             if path.is_file() and ('.so' in path.name or path.suffix == '.dylib')}
    if not files:
        raise ValueError('No native libraries found')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'candidate', 'lineage', 'source-comparison', 'development-decision',
                 'evaluator', 'manifests', 'registry', 'runtime', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'metal', 'cuda', 'vulkan'], required=True)
    parser.add_argument('--reference-cache', type=Path)
    args = parser.parse_args()
    lineage = json.loads(args.lineage.read_text())
    source = json.loads(args.source_comparison.read_text())
    development = json.loads(args.development_decision.read_text())
    original_sha = sha(args.original)
    candidate_sha = sha(args.candidate)
    if original_sha != '8967e04bd73fd8e88bccbfe97d0ec952beac73b1b3b3cba4b5c03316fcd8baa0':
        raise ValueError('Original Q8 identity changed')
    if lineage['gguf_sha256'] != candidate_sha or lineage['fitted_rows_exact_after_f16_rounding'] != 12288:
        raise ValueError('Native candidate identity or frozen-row audit failed')
    if not source['release_qualified'] or source['candidate_model_sha256'] != lineage['candidate_sha256']:
        raise ValueError('Source candidate has not qualified')
    if source['reference_model_sha256'] != SOURCE_SHA:
        raise ValueError('Source comparison reference changed')
    if development['status'] != 'pass' or development['candidate_sha256'] != candidate_sha:
        raise ValueError('Native development comparison has not passed')
    if development['original_sha256'] != original_sha:
        raise ValueError('Native development reference changed')
    if development['registry_sha256'] != '249a56d2c2c8f688e4d0654810ae1be69bfe6c3a9b8afdbd051d1c1804895b85':
        raise ValueError('Native development membership changed')
    registry = json.loads(args.registry.read_text())
    if sha(args.registry) != 'c6fee4d0cb2c2071d41148f545048d2791dcc23ec49db2dd299fbcaaa283b253':
        raise ValueError('Larger evaluation registry changed')
    if source['registry_sha256'] != sha(args.registry):
        raise ValueError('Source comparison membership changed')
    available_manifests = {path.stem for path in args.manifests.glob('*.jsonl')}
    if not set(registry['sets']) <= available_manifests:
        raise ValueError('A registered manifest is missing')
    for name, record in registry['sets'].items():
        if sha(args.manifests / (name + '.jsonl')) != record['manifest_sha256']:
            raise ValueError('Manifest changed: ' + name)
    import orukeet.audio
    import orukeet.nvidia
    protocol = {
        'device': args.device, 'runtime_library_sha256': runtime_hashes(args.runtime),
        'evaluation_script_sha256': sha(args.evaluator),
        'binding_sha256': sha(orukeet.nvidia.__file__), 'audio_path_sha256': sha(orukeet.audio.__file__),
        'registered_manifest_count': len(registry['sets']),
        'unselected_manifest_files': sorted(available_manifests - set(registry['sets'])),
        'decoder': 'Native TDT greedy defaults; automatic language; word offsets and punctuation enabled',
        'scope': ('Paired native comparison with recorded original Q8 and freshly evaluated candidate '
                  if args.reference_cache else 'Fresh paired native replay of original Q8 and candidate ')
                 + lineage.get('export_precision', 'q8_0') + ' on the same exposed larger suite'}
    if development['protocol']['device'] != args.device:
        raise ValueError('Native device changed since development')
    for key in ('runtime_library_sha256', 'evaluation_script_sha256', 'binding_sha256', 'audio_path_sha256'):
        if development['protocol'][key] != protocol[key]:
            raise ValueError('Native development protocol changed: ' + key)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    # The source folder also holds unregistered slices. The file evaluator
    # globs its input directory, so give it only the registry's exact membership.
    staged_manifests = args.output / 'manifests'
    staged_manifests.mkdir()
    for name, record in registry['sets'].items():
        path = staged_manifests / (name + '.jsonl')
        path.write_bytes((args.manifests / path.name).read_bytes())
        if sha(path) != record['manifest_sha256']:
            raise ValueError('Staged manifest changed: ' + name)
    for label, model in [('original', args.original), ('candidate', args.candidate)]:
        if label == 'original' and args.reference_cache:
            reuse_reference(args.reference_cache, args.output / label, protocol, registry, full=True)
        else:
            with (args.output / (label + '.log')).open('x') as log:
                subprocess.run([sys.executable, str(args.evaluator), '--model', str(model),
                                '--runtime', str(args.runtime), '--device', args.device,
                                '--manifests', str(staged_manifests), '--output', str(args.output / label)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        if runtime_hashes(args.runtime) != protocol['runtime_library_sha256']:
            raise ValueError('Native libraries changed during evaluation')
        metadata_path = args.output / label / 'results.json'
        metadata = json.loads(metadata_path.read_text())
        if metadata['model_sha256'] != (original_sha if label == 'original' else candidate_sha):
            raise ValueError('Evaluated model identity changed')
        metadata['decoding'] = protocol
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    result = qualify(compare(args.output / 'original', args.output / 'candidate', registry), 'pass', False)
    result['native_accuracy_qualified'] = result.pop('release_qualified')
    result['source_qualified'] = True
    result['interpretation'] = ('Matched native decoding and audio path on previously exposed '
        'recordings. Original Q8; candidate ' + lineage.get('export_precision', 'q8_0')
        + '. No new generalization or latency claim.')
    result['registry_sha256'] = sha(args.registry)
    result['lineage_sha256'] = sha(args.lineage)
    result['source_comparison_sha256'] = sha(args.source_comparison)
    result['development_decision_sha256'] = sha(args.development_decision)
    result['statistics_implementation_sha256'] = sha(compare_oruk_export.__file__)
    result['protocol'] = protocol
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in
                     ('status', 'native_accuracy_qualified', 'primary', 'english', 'failures')}), flush=True)


if __name__ == '__main__':
    main()
