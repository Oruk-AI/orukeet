"""Trace audited recovery ancestry without reading model weights or transcripts."""
import argparse
import json
from pathlib import Path
import re

from identity import sha, SOURCE_SHA


def trace(label, results):
    if not re.fullmatch(r'[ra]\d+-\d{4}', label):
        raise ValueError('Expected an audited export label')
    audits = {}
    for path in results.glob('*-audit.json'):
        if re.fullmatch(r'[ra]\d+-\d{4}-audit.json', path.name):
            record = json.loads(path.read_text())
            audits[record['candidate_sha256']] = (path.name.removesuffix('-audit.json'), path, record)
    target = json.loads((results / (label + '-audit.json')).read_text())['candidate_sha256']
    nodes = {SOURCE_SHA: {'label': 'original-before-surgery', 'optimizer_updates_in_this_node': 0,
                         'parents': [], 'longest_recovery_path_updates': 0}}

    def visit(digest):
        if digest in nodes:
            return nodes[digest]['longest_recovery_path_updates']
        name, audit_path, audit = audits[digest]
        assert audit['status'] == 'pass' and audit['original_sha256'] == SOURCE_SHA
        assert audit['frozen_gabor_rows_exact'] == 12288 and audit['all_tensors_finite']
        evidence = {audit_path.name: sha(audit_path)}
        if name.startswith('a'):
            average_path = results / (name + '-average.json')
            if not average_path.exists():
                average_path = results / (name + '.json')
            average = json.loads(average_path.read_text())
            assert average['output_sha256'] == digest
            assert average['frozen_rows_copied_exactly'] == 12288
            parents = average['parents_sha256']
            updates = 0
            extra = {'operation': 'parameter average', 'parent_weights': average['weights']}
            evidence[average_path.name] = sha(average_path)
        else:
            run, step = name.split('-')
            provenance_path = results / (run + '-provenance.json')
            provenance = json.loads(provenance_path.read_text())
            config = provenance['config']
            updates = int(step)
            assert 0 < updates <= config['trainer']['max_steps']
            parents = [provenance.get('initial_student_sha256', SOURCE_SHA)]
            extra = {'operation': 'recovery training', 'seed': config['seed'],
                     'declared_run_steps': config['trainer']['max_steps']}
            evidence[provenance_path.name] = sha(provenance_path)
        longest = updates + max(visit(parent) for parent in parents)
        nodes[digest] = {'label': name, 'optimizer_updates_in_this_node': updates,
                         'parents': parents, 'longest_recovery_path_updates': longest,
                         'evidence_sha256': evidence, **extra}
        return longest

    longest = visit(target)
    return {'label': label, 'candidate_sha256': target, 'longest_recovery_path_updates': longest,
            'nodes': nodes, 'scope': 'Audited weight ancestry only. The path includes prior recovery '
            'before parameter averaging; it is not a single uninterrupted optimizer run. It excludes '
            'foundation training, pre-surgery adaptation and experiments outside this ancestry. '
            'No unique audio-hours or independent generalization claim is inferred.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--results', type=Path, default=Path(__file__).with_name('results'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = trace(args.label, args.results)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'nodes'}))
