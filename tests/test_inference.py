import hashlib
import json
import os
from pathlib import Path
import wave

import numpy as np
import pytest

from orukeet import Orukeet
from orukeet.audio import windows
from orukeet.process import SpeechProcess
import orukeet.model as model_module


def wav(path, seconds=2, rate=16000, channels=1, silence=False):
    samples = np.zeros(int(seconds*rate), dtype=np.int16) if silence else (
        np.sin(np.arange(int(seconds*rate)) * (2*np.pi*220/rate)) * 3000).astype(np.int16)
    samples = np.repeat(samples[:,None], channels, axis=1)
    with wave.open(str(path), 'wb') as stream:
        stream.setparams((channels, 2, rate, 0, 'NONE', 'not compressed'))
        stream.writeframes(samples.tobytes())


def test_resampling_windows_preserve_length_and_timeline(tmp_path):
    audio = tmp_path / 'épreuve.wav'
    wav(audio, seconds=66, rate=48000, channels=2)
    parts = list(windows(str(audio)))
    assert len(parts) == 3
    assert sum(len(x) for _, x in parts) == 66*16000
    total = 0
    for start, samples in parts:
        assert start == total / 16000
        assert samples.size <= 30*16000 and np.isfinite(samples).all()
        total += len(samples)


def test_hash_mismatch_prevents_worker_start(tmp_path, monkeypatch):
    model = tmp_path / 'wrong.gguf'
    model.write_bytes(b'bad')
    monkeypatch.setattr(model_module, 'SpeechProcess', lambda: pytest.fail('Worker should not start'))
    with pytest.raises(ValueError, match='SHA-256'):
        Orukeet(model, tmp_path)


def test_long_audio_offsets_silence_and_lifecycle(tmp_path, monkeypatch):
    calls = []
    class Worker:
        def request(self, op, **kwargs):
            calls.append(op)
            return {} if op == 'load' else {'text': 'Example.', 'segments': [dict(text='Example.', start=.2,end=1.)]}
        def close(self):
            calls.append('close')
    monkeypatch.setattr(model_module, 'SpeechProcess', Worker)
    model = tmp_path/'fixture.gguf'; model.write_bytes(b'fixture')
    audio = tmp_path/'long.wav'; wav(audio, seconds=66)
    silent = tmp_path/'silence.wav'; wav(silent, silence=True)
    with Orukeet(model,tmp_path,expected_sha256=hashlib.sha256(b'fixture').hexdigest()) as engine:
        result=engine.transcribe(audio)
        assert len(result['segments']) == 3
        assert [s['start'] for s in result['segments']] == pytest.approx([p[0]+.2 for p in windows(str(audio))])
        assert result['language'] is None
        with pytest.raises(FileNotFoundError, match='Local audio file'):
            engine.transcribe('https://example.invalid/audio.wav')
        before=len(calls)
        assert engine.transcribe(silent) == dict(text='',segments=[],language=None)
        assert len(calls) == before
    with pytest.raises(RuntimeError, match='closed'):
        engine.transcribe(silent)
    assert calls.count('load') == 1


def test_real_worker_protocol_error_recovers():
    worker=SpeechProcess()
    try:
        with pytest.raises(RuntimeError, match='Unknown worker operation'):
            worker.request('invalid',timeout=10)
        with pytest.raises(RuntimeError, match='No model loaded'):
            worker.request('transcribe',audio_path='unused',timeout=10)
    finally:
        worker.close()
    assert worker.process.poll() is not None


@pytest.mark.skipif(not os.getenv('ORUKEET_TEST_MODEL'), reason='Requires explicit local model/runtime/audio fixtures')
def test_released_model_offline_reuse_and_silence(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE','1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    silent=tmp_path/'silence.wav';wav(silent,silence=True)
    with Orukeet(os.environ['ORUKEET_TEST_MODEL'],os.environ['ORUKEET_TEST_RUNTIME'],
                 device=os.environ.get('ORUKEET_TEST_DEVICE','cpu')) as model:
        pid=model._worker.process.pid
        first=model.transcribe(os.environ['ORUKEET_TEST_AUDIO'])
        second=model.transcribe(os.environ['ORUKEET_TEST_AUDIO'])
        assert 'country' in first['text'].lower() and first == second
        assert model._worker.process.pid == pid and first['language'] is None
        assert all(0 <= s['start'] <= s['end'] <= 11.1 for s in first['segments'])
        assert model.transcribe(silent)['text'] == ''
