"""Guard against changing benchmark audio to fit inaccurate metadata."""
import hashlib,sqlite3,tempfile,unittest
from pathlib import Path
import numpy as np
import soundfile as sf
from run import write_audio
from audit_history import fingerprint
from audit_decoded_history import decoded
from compare import historical_pcm_overlap

class AudioPreparationTest(unittest.TestCase):
    def test_stored_pcm16_positive_survives_decoding_and_requantization(self):
        samples=np.array([-32768,-12345,0,12345,32767],dtype='<i2')
        stored=hashlib.sha256(samples.tobytes()).hexdigest()
        db=sqlite3.connect(':memory:');db.execute('CREATE TABLE exposure(kind TEXT,fingerprint TEXT)')
        db.execute('INSERT INTO exposure VALUES (?,?)',('pcm_sha256',fingerprint(stored)))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'historical.wav';sf.write(path,samples,16000,subtype='PCM_16')
            repair=decoded((stored,path));self.assertEqual(repair['status'],'verified')
            data,sr=sf.read(path,dtype='float32')
            result=write_audio({'uid':'known-historical','duration':len(samples)/sr},data,sr,Path(tmp),db)
            self.assertFalse(result['known_pcm_overlap'])
            self.assertTrue(historical_pcm_overlap(result,{repair['float32_pcm_sha256']},{repair['legacy_pcm_sha256']}))
            result=write_audio({'uid':'novel-silence','duration':1},np.zeros(16000,dtype='float32'),16000,Path(tmp),db)
            self.assertFalse(historical_pcm_overlap(result,{repair['float32_pcm_sha256']},{repair['legacy_pcm_sha256']}))
        db.close()

    def test_trimmed_official_audio_keeps_samples_and_reference(self):
        data=(.1*np.sin(np.arange(48511,dtype=np.float32)*.05)).astype(np.float32)
        row={'uid':'synthetic-duration-regression','duration':20.0,'text':'unchanged reference'}
        db=sqlite3.connect(':memory:');db.execute('CREATE TABLE exposure(kind TEXT,fingerprint TEXT)')
        with tempfile.TemporaryDirectory() as tmp:
            result=write_audio(row,data,16000,Path(tmp),db)
            actual,sr=sf.read(result['audio_filepath'],dtype='float32')
            self.assertTrue(np.array_equal(actual,data));self.assertEqual(sr,16000)
            self.assertEqual(result['text'],row['text']);self.assertEqual(result['duration'],20.0)
            self.assertAlmostEqual(result['actual_duration'],3.0319375)
            self.assertTrue(result['declared_duration_mismatch']);self.assertFalse(result['known_pcm_overlap'])
        db.close()

if __name__=='__main__':unittest.main()
