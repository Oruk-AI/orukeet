"""Hand-counted corpus aggregation and exact pairing of identical hypotheses."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DiagnosticSummary(unittest.TestCase):
    def test_weighted_corpus_rates_and_paired_zero_interval(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            rows, numeric = [], []
            for i, (role, words, errors) in enumerate([
                ('training_exposed_fit_check', 1, [1, 0, 0]),
                ('training_exposed_fit_check', 3, [2, 1, 1]),
                ('nontraining_control', 10, [2, 3, 1]),
            ]):
                uid = str(i)
                split = role
                rows.append(dict(uid=uid, split=split, lang='en', evaluation_role=role, cluster=uid))
                models = {label: dict(words=words, chars=2 * words, errors=e, char_errors=2 * e)
                          for label, e in zip(['parakeet', 'parent', 'candidate'], errors)}
                numeric.append(dict(record_sha256=hashlib.sha256(uid.encode()).hexdigest(), split=split,
                                    evaluation_role=role, counts={'standard_english': models}))
            manifest = p / 'manifest.jsonl'
            manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            (p / 'comparison.json').write_text(json.dumps(dict(status='complete', rows=3,
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())))
            with gzip.open(p / 'numeric-evidence.jsonl.gz', 'wt') as f:
                f.write(''.join(json.dumps(r) + '\n' for r in numeric))
            subprocess.run([sys.executable, str(Path(__file__).with_name('summarize.py')),
                            '--manifest', str(manifest), '--evaluation', str(p), '--output', str(p), '--bootstrap', '100'],
                           check=True, capture_output=True)
            d = json.loads((p / 'summary.json').read_text())['roles']
            fit = d['training_exposed_fit_check']
            self.assertEqual(fit['parakeet_wer'], 75.)
            self.assertEqual(fit['candidate_wer'], 25.)
            self.assertEqual(fit['candidate_minus_parent_wer']['paired_cluster_bootstrap_95ci_pp'], [0., 0.])
            self.assertEqual(d['nontraining_control']['candidate_wer'], 10.)


if __name__ == '__main__':
    unittest.main()
