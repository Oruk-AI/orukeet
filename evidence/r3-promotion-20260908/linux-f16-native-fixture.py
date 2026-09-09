import array,hashlib,json,sys,time
from pathlib import Path
import wave
import numpy as np
from orukeet.nvidia import NvidiaRecognizer
import orukeet.nvidia
root=Path('/home/nathanroll/parakeet-ft');dest=Path('/dev/shm/orukeet-r3-native-20260908')
model=dest/'orukeet-v0.1.0rc1-f16.gguf';runtime=root/'inference_20260905/cuda';fixture=root/'gabor_half_20260906/fixtures/jfk.wav'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(model)=='de53fb8ec251fb07ade15baabe17b00774ae3f1112f8618b062337f90fb49194'
with wave.open(str(fixture), 'rb') as audio:
 assert audio.getframerate()==16000 and audio.getnchannels()==1 and audio.getsampwidth()==2
 samples=np.frombuffer(audio.readframes(audio.getnframes()),dtype='<i2').astype(np.float32)/32768.0
results=[]
for device in ['cuda','cpu']:
 engine=NvidiaRecognizer(str(runtime),str(model),device)
 try:
  calls=[]
  for i in range(2):
   start=time.perf_counter();r=engine.transcribe(array.array('f',samples),'auto');calls.append(dict(seconds=time.perf_counter()-start,text=r['text']))
   assert 'ask not what your country can do for you' in r['text'].lower()
  assert calls[0]['text']==calls[1]['text']
 finally:engine.close()
 results.append(dict(device=device,calls=calls))
receipt=dict(status='passed',model='Orukeet r3 F16',model_sha256=sha(model),fixture_sha256=sha(fixture),python=sys.version.split()[0],binding_sha256=sha(Path(orukeet.nvidia.__file__)),runtime_library_sha256={p.name:sha(p) for p in sorted((runtime/'lib').iterdir()) if p.is_file() and '.so' in p.name},results=results,scope='Two repeated JFK transcriptions per device in the existing Linux CUDA container, using the same CUDA SDK for CPU selection. No timing comparison, standalone CPU package or cross-platform app claim.')
(dest/'linux-f16-native-smoke.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
