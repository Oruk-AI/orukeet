import json
from pathlib import Path
import tempfile
import unittest

from compare_confirmation import compare
from seal_confirmation import cluster
from select_development import decide


class EvaluationContract(unittest.TestCase):
    def test_parallel_sentences_and_multilingual_speakers_share_clusters(self):
        self.assertEqual(cluster({'src':'fleurs','lang':'en','fleurs_id':'12'}),
                         cluster({'src':'fleurs','lang':'de','fleurs_id':'12'}))
        self.assertEqual(cluster({'src':'cv','lang':'en','speaker_id':'abc'}),
                         cluster({'src':'cv','lang':'de','speaker_id':'abc'}))
        self.assertNotEqual(cluster({'src':'cv','speaker_id':'abc'}),
                            cluster({'src':'voxpopuli','speaker_id':'abc'}))

    def test_global_pairing_preserves_identical_across_language_distributions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry = {'sets':{},'primary_languages':{},'families':{'cv':[]},
                        'english_corpora':{'core':['cv_en'],'learner':['cv_en'],'dialect':['cv_en']}}
            for model in ['base','candidate']:
                folder = root/model
                folder.mkdir()
                results = {'normalizer':'fixed','decoding':{'strategy':'greedy_batch'},'model_sha256':model,'sets':{}}
                for lang in ['en','de']:
                    name = 'cv_'+lang
                    registry['sets'][name] = {'manifest_sha256':name,'rows':2,'language':lang,'source':'cv'}
                    registry['primary_languages'][lang] = {'sets':[name]}
                    results['sets'][name] = {'manifest_sha256':name}
                    with (folder/(name+'_hypotheses.jsonl')).open('w') as f:
                        for i in [0,1]:
                            row = {'audio_filepath':lang+str(i),'text':'reference','words':10,'errors':(2 if model=='base' else 1)+i,
                                   'src':'cv','speaker_id':str(i),'accent':'unknown'}
                            f.write(json.dumps(row)+'\n')
                (folder/'results.json').write_text(json.dumps(results))
            registry['families']['cv'] = ['cv_en','cv_de']
            output = compare(root/'base',root/'candidate',registry,
                             {'confirmation':{'minimum_relative_primary_improvement_over_stock':.1}},replicates=100)
            self.assertEqual(output['bootstrap']['distinct_global_clusters'],2)
            self.assertEqual(output['metrics']['set:cv_en']['paired_delta_95ci'],
                             output['metrics']['set:cv_de']['paired_delta_95ci'])
            self.assertAlmostEqual(output['primary']['delta_pp'],-10)
            self.assertTrue(all(abs(v+10)<1e-10 for v in output['primary']['paired_delta_95ci']))
            self.assertEqual(output['status'],'passed')

    def test_selection_rejects_english_regression_despite_primary_gain(self):
        def result(names,wer,model):
            return {'model_sha256':model,'model_path':model+'.nemo','normalizer':'fixed','decoding':{},
                    'sets':{name:{'manifest_sha256':name,'slices':{'all':{'wer':wer,'rows':10,'words':100}}} for name in names}}
        names = ['cv_'+str(i) for i in range(24)]+['fleurs_'+str(i) for i in range(25)]
        extras = ['english_dialects_dev','speechocean762_dev','libri_en_dev_other']
        base = [result(names,10,'base'),result(extras[:2],10,'base'),result(extras[2:],10,'base')]
        candidate = result(names+extras,8,'candidate')
        candidate['sets']['libri_en_dev_other']['slices']['all']['wer']=10.21
        protocol = {'development':{'maximum_source_language_regression_pp':.5,
            'maximum_each_english_guardrail_regression_pp':.2,'minimum_relative_primary_improvement_over_stock':.1}}
        decision = decide(base,[(.5,candidate)],protocol)
        self.assertEqual(decision['status'],'continue_development')
        self.assertEqual(decision['candidates'][0]['failures'][0]['set'],'libri_en_dev_other')
        candidate['sets']['libri_en_dev_other']['manifest_sha256']='changed'
        with self.assertRaises(ValueError):
            decide(base,[(.5,candidate)],protocol)


if __name__ == '__main__':
    unittest.main()
