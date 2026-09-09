from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import threading

import numpy as np

from .audio import windows
from .process import SpeechProcess


class Orukeet:
    """Own a serial, persistent recognizer process. Use as a context manager.

    Construction validates the released Q8 file once. Warm transcriptions reuse
    the worker. `close()` also interrupts an in-flight request from another thread.
    To load your own export, supply its SHA-256 through `expected_sha256`.
    """
    def __init__(self, model: str | Path, runtime: str | Path, device='cpu', *, expected_sha256=None):
        if device not in ('cpu', 'metal', 'cuda', 'vulkan'):
            raise ValueError('Choose cpu, metal, cuda, or vulkan')
        spec = json.loads(Path(__file__).with_name('artifacts.json').read_text())['files']['q8']
        with Path(model).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != (expected_sha256 or spec['sha256']):
            raise ValueError('Model SHA-256 differs from the requested artifact')
        self.model_sha256, self.device = actual, device
        self._closed = False
        self._lock = threading.Lock()
        self._worker = SpeechProcess()
        try:
            self._worker.request('load', model_path=str(Path(model).resolve()),
                                 runtime=str(Path(runtime).resolve()), device=device, timeout=300)
        except BaseException:
            self.close()
            raise

    def transcribe(self, audio: str | Path) -> dict:
        """Decode a file into text and segments in seconds on the original timeline.

        Audio is mixed to mono and resampled to 16 kHz. Long inputs use bounded
        24–30 second windows. The native API exposes no language ID/confidence.
        """
        with self._lock, tempfile.TemporaryDirectory(prefix='orukeet-') as temp:
            if self._closed:
                raise RuntimeError('Recognizer is closed; construct a new Orukeet instance')
            # PyAV also accepts network URLs. This API promises local, offline
            # transcription, so validate the file before handing it to PyAV.
            audio = Path(audio)
            if not audio.is_file():
                raise FileNotFoundError(f'Local audio file does not exist: {audio}')
            texts, segments = [], []
            path = Path(temp) / 'audio.f32'
            for offset, samples in windows(str(audio)):
                if not samples.size or np.max(np.abs(samples)) <= .00025:
                    continue
                samples.tofile(path)
                result = self._worker.request('transcribe', audio_path=str(path), timeout=300)
                texts.append(result['text'])
                segments.extend(dict(segment, start=offset + segment['start'], end=offset + segment['end'])
                                for segment in result['segments'])
            return dict(text=' '.join(texts).strip(), segments=segments, language=None)

    def close(self):
        self._closed = True
        self._worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
