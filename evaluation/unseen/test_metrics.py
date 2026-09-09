"""Check scientific invariants before evaluating any model outputs."""
import importlib.util
from pathlib import Path
import unittest
from metrics import counts,normalize,ENGLISH

class MetricTest(unittest.TestCase):
    def test_edit_accounting(self):
        c=counts('one two three','one four three five')
        self.assertEqual((c['errors'],c['substitutions'],c['deletions'],c['insertions'],c['words']),(2,1,0,1,3))
    def test_failure_keeps_denominator(self):
        c=counts('one two three','')
        self.assertEqual((c['words'],c['errors'],c['deletions']),(3,3,3))
    def test_empty_normalized_reference_preserves_insertions(self):
        c=counts('...','hello');self.assertEqual((c['words'],c['errors']),(0,1))
    def test_english_formatting(self):
        self.assertEqual(counts('twenty one','21',ENGLISH)['errors'],0)
        self.assertGreater(counts('twenty one','21')['errors'],0)
    def test_legacy_equivalence(self):
        roots=[Path(__file__).resolve().parents[2]/'training/evaluate.py',Path('/home/nathanroll/parakeet-ft/evaluate_20260905.py')]
        path=next(p for p in roots if p.exists());spec=importlib.util.spec_from_file_location('legacy',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        texts=['«Straße» 12.5%','Γιάννης: κόσμος;','Ёлка и ѝ','ţară; ştiință','He\'s twenty-one!','naïve café…','你好，世界','foo\nbar / baz','Hyphens—en–minus-','İstanbul αςσ']
        for s in texts:self.assertEqual(normalize(s),m.normalize(s),s)
        for a in texts:
            for b in texts:self.assertEqual(counts(a,b)['errors'],m.distance(m.normalize(a).split(),m.normalize(b).split()))

if __name__=='__main__':unittest.main()
