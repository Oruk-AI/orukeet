"""Qualify one source-passing Gabor export without changing any default model."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from identity import sha, SOURCE_SHA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--precision', choices=['q8_0', 'f16', 'q8_0_t16'], default='q8_0')
    parser.add_argument('--reference-run', help='Completed native run whose original predictions may be reused')
    args = parser.parse_args()
    if not re.fullmatch(r'[ra]\d+-\d{4}', args.label):
        raise ValueError('Expected an audited recovery export label')
    root = Path('/home/nathanroll/parakeet-ft')
    exp = root / 'gabor_half_20260906'
    inference = root / 'inference_20260905'
    run, variant = args.label.split('-')
    model = exp / ('orukeet_gabor_half_' + run + '_20260906') / (
        args.label + '.nemo' if run.startswith('a') else 'step-' + variant + '.nemo')
    audit_path = exp / (args.label + '-audit.json')
    source_path = exp / ('full-' + args.label) / 'comparison.json'
    source = json.loads(source_path.read_text())
    audit = json.loads(audit_path.read_text())
    if source['release_qualified'] is not True or source['development_status'] != 'pass':
        raise ValueError('Source candidate has not qualified; native evaluation was not started')
    assert source['reference_model_sha256'] == SOURCE_SHA == audit['original_sha256']
    assert audit['status'] == 'pass' and audit['frozen_gabor_rows_exact'] == 12288
    assert source['candidate_model_sha256'] == audit['candidate_sha256'] == sha(model)
    original = inference / 'oruk-parakeet-v3-20260905.q8_0.gguf'
    assert sha(original) == '8967e04bd73fd8e88bccbfe97d0ec952beac73b1b3b3cba4b5c03316fcd8baa0'
    reference_run = None
    reference_receipt = None
    if args.reference_run:
        if not re.fullmatch(r'[ra]\d+-\d{4}(?:-f16)?', args.reference_run):
            raise ValueError('Invalid reference run label')
        reference_run = exp / ('native-' + args.reference_run)
        dev = json.loads((reference_run / 'development/comparison.json').read_text())
        full = json.loads((reference_run / 'larger/comparison.json').read_text())
        assert dev['original_sha256'] == full['reference_model_sha256'] == sha(original)
        assert dev['registry_sha256'] == '249a56d2c2c8f688e4d0654810ae1be69bfe6c3a9b8afdbd051d1c1804895b85'
        assert full['registry_sha256'] == 'c6fee4d0cb2c2071d41148f545048d2791dcc23ec49db2dd299fbcaaa283b253'
        reference_receipt = {'run': args.reference_run, 'new_reference_inference': False,
            'comparison_sha256': {phase: sha(reference_run / phase / 'comparison.json')
                                 for phase in ['development', 'larger']},
            'input_sha256': {str(p.relative_to(reference_run)): sha(p)
                            for phase in ['development', 'larger']
                            for p in sorted((reference_run / phase / 'original').iterdir()) if p.is_file()},
            'scope': 'Reuse only original predictions under an identical runtime, decoding, audio path, '
                     'normalizer and registered membership; verify all prediction bytes when copying. '
                     'Fresh candidate inference and every original accuracy gate remain required.'}
    development = exp / 'development'
    registry = json.loads((development / 'registry.json').read_text())
    assert sha(development / 'registry.json') == '249a56d2c2c8f688e4d0654810ae1be69bfe6c3a9b8afdbd051d1c1804895b85'
    records = {record['name']: record for record in registry['records']}
    assert len(records) == len(registry['records']) == 51
    assert sum(record['rows'] for record in records.values()) == 5100
    assert set(path.stem for path in development.glob('*.jsonl')) == set(records)
    for name, record in records.items():
        assert sha(development / (name + '.jsonl')) == record['sha256']
    fixture_manifest = exp / 'native-latency-fixtures.json'
    fixtures = json.loads(fixture_manifest.read_text())['fixtures']
    assert [row['name'] for row in fixtures] == ['jfk.wav', 'fr_fr.source.wav', 'es_419.source.wav', 'lv_lv.source.wav']
    for row in fixtures:
        assert sha(exp / 'fixtures' / row['name']) == row['sha256']
    assert shutil.disk_usage(exp).free >= 6_000_000_000
    suffix = {'q8_0': '', 'f16': '-f16', 'q8_0_t16': '-t16'}[args.precision]
    output = exp / ('native-' + args.label + suffix)
    output.mkdir(exist_ok=False)
    declaration = {
        'label': args.label, 'source_comparison_sha256': sha(source_path),
        'source_model_sha256': sha(model), 'original_q8_sha256': sha(original),
        'device': 'cuda', 'new_training': False, 'default_changed': False,
        'export_precision': args.precision,
        'reference_reuse': reference_receipt,
        'order': ['Native export and exact fixed-row audit', 'paired 51-slice native development',
                  'paired 61-slice native larger regression', 'paired warm fixture latency'],
        'stopping_rule': 'Stop if either native accuracy comparison fails. Keep every result. '
                         'Successful accuracy checks permit latency measurement, not automatic promotion.',
        'latency_scope': 'Four existing short English/French/Spanish/Latvian fixtures, two '
                         'reversed-order blocks, two warmups and ten timed calls per model per '
                         'fixture per block. This does not qualify other devices.',
        'latency_fixtures_sha256': sha(fixture_manifest),
        'code_sha256': {name: sha(exp / name) for name in [
            'export_native.py', 'native_development.py', 'native_full.py', 'benchmark_native.py',
            'evaluate_oruk_export.py', 'compare.py', 'compare_oruk_export.py', 'metric_qualification.py',
            'run_native_qualification.py', 'reuse_native_reference.py']}}
    if args.precision == 'q8_0_t16':
        declaration['code_sha256']['transducer_f16.py'] = sha(exp / 'transducer_f16.py')
        baseline = exp / ('native-' + args.label) / 'export/orukeet-gabor-half.q8_0.gguf'
        baseline_development = exp / ('native-' + args.label) / 'development/comparison.json'
        assert json.loads(baseline_development.read_text())['status'] == 'fail'
        declaration['mixed_precision'] = {
            'baseline_q8_sha256': sha(baseline), 'baseline_development_sha256': sha(baseline_development),
            'policy': 'One variant retains Q8 throughout the encoder and stores the seven '
                      'predictor/LSTM and joint linear matrices in F16 from the same trained source. '
                      'Audit every other tensor against the original Q8 bytes. No new training '
                      'or language-specific weight changes. Stop at any accuracy failure.'}
    (output / 'decision.json').write_text(json.dumps(declaration, indent=2) + '\n')
    pythonpath = str(exp / 'export-deps') + ':' + str(inference / 'orukeet-wheel')

    def run_tool(script, arguments, cpu=False):
        def verify_reference_snapshot():
            if reference_receipt:
                for name, digest in reference_receipt['input_sha256'].items():
                    if sha(reference_run / name) != digest:
                        raise ValueError('Recorded reference changed: ' + name)
        verify_reference_snapshot()
        command = ['./nemo.sh', 'env', 'OPENBLAS_NUM_THREADS=2', 'OMP_NUM_THREADS=2',
                   'PYTHONPATH=' + pythonpath]
        if cpu:
            command.append('CUDA_VISIBLE_DEVICES=')
        command.extend(['python', str(exp / script), *map(str, arguments)])
        with (output / (script.removesuffix('.py') + '.log')).open('x') as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ, NEMO_NAME='orukeet-native-' + args.label + suffix + '-' + script.removesuffix('.py')))
        verify_reference_snapshot()

    export = output / 'export'
    run_tool('export_native.py', ['--candidate', model, '--audit', audit_path,
        '--fits', exp / 'fit-full/fits.json', '--converter', inference / 'converter', '--output', export,
        '--outtype', args.precision,
        *(['--baseline-q8', baseline] if args.precision == 'q8_0_t16' else [])], cpu=True)
    candidate = export / ('orukeet-gabor-half.' + args.precision + '.gguf')
    lineage = export / 'lineage.json'
    evaluator = exp / 'evaluate_oruk_export.py'
    runtime = inference / 'cuda'
    common = ['--original', original, '--candidate', candidate, '--lineage', lineage,
              '--evaluator', evaluator, '--runtime', runtime, '--device', 'cuda']
    run_tool('native_development.py', [*common, '--manifests', development, '--output', output / 'development',
        *(['--reference-cache', reference_run / 'development/original'] if reference_run else [])])
    result = json.loads((output / 'development/comparison.json').read_text())
    if result['status'] != 'pass':
        print('NATIVE_DEVELOPMENT_FAILED', flush=True)
        return
    run_tool('native_full.py', [*common, '--source-comparison', source_path,
        '--development-decision', output / 'development/comparison.json',
        '--manifests', root / 'manifests/goal_v2_confirmation',
        '--registry', root / 'manifests/goal_v2_confirmation/sealed_metric_registry.json',
        '--output', output / 'larger',
        *(['--reference-cache', reference_run / 'larger/original'] if reference_run else [])])
    result = json.loads((output / 'larger/comparison.json').read_text())
    if not result['native_accuracy_qualified']:
        print('NATIVE_LARGER_FAILED', flush=True)
        return
    run_tool('benchmark_native.py', ['--original', original, '--candidate', candidate,
        '--runtime', runtime, '--device', 'cuda', '--output', output / 'latency-cuda.json',
        *[exp / 'fixtures' / row['name'] for row in fixtures]])
    print('NATIVE_ACCURACY_AND_LATENCY_MEASUREMENTS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
