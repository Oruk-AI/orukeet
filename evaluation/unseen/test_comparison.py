import unittest
import numpy as np
from compare import aggregate,paired_bootstrap,macro_endpoint


def record(cluster,b,c,words=10):
    def metric(e):return {'errors':e,'words':words,'chars':words*5,'char_errors':e,'utterance_error':int(e>0)}
    return {'cluster':cluster,'duration':3,'parakeet':metric(b),'orukeet':metric(c)}

class ComparisonTest(unittest.TestCase):
    def test_identical_predictions_have_zero_interval(self):
        rows=[record('a',1,1),record('b',4,4),record('c',2,2)]
        _,d,ci=paired_bootstrap(rows,1000)
        self.assertTrue(np.all(d==0));self.assertEqual(ci['wer_delta_pp_95ci'],[0,0])
    def test_many_utterances_one_speaker_do_not_create_independence(self):
        _,_,ci=paired_bootstrap([record('one speaker',1,0)]*100,1000)
        self.assertEqual(ci['clusters'],1);self.assertIn('reason',ci)
    def test_missing_group_metadata_does_not_invent_independence(self):
        rows=[dict(record('a',1,0),cluster_metadata_available=False),dict(record('b',3,2),cluster_metadata_available=False)]
        _,_,ci=paired_bootstrap(rows,1000)
        self.assertIsNone(ci['clusters']);self.assertIn('grouping is unavailable',ci['reason'])

    def test_pairing_and_sign(self):
        _,d,_=paired_bootstrap([record(str(i),4,2) for i in range(10)],1000)
        self.assertTrue(np.all(d==-20))
    def test_corpus_wer_uses_reference_word_counts(self):
        result=aggregate([record('a',1,0,1),record('b',0,2,99)])
        self.assertEqual(result['parakeet']['wer'],1);self.assertEqual(result['orukeet']['wer'],2)
    def test_macro_keeps_languages_equal(self):
        a=aggregate([record('a',1,0,1)]);b=aggregate([record('b',0,0,99)])
        m=macro_endpoint(['a','b'],{'a':a,'b':b},{})
        self.assertEqual(m['parakeet_wer'],50);self.assertEqual(m['orukeet_wer'],0)

if __name__=='__main__':unittest.main()
