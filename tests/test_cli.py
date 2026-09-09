import json
import sys

from orukeet import cli


def test_install_writes_portable_receipt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, 'resolve_device', lambda device: 'cpu')
    monkeypatch.setattr(cli, 'install_runtime', lambda *a: tmp_path / 'runtime')
    monkeypatch.setattr(cli, 'fetch', lambda *a: tmp_path / 'model.gguf')
    receipt = tmp_path / 'installación.json'
    monkeypatch.setattr(sys, 'argv', ['orukeet', 'install', '--output', str(receipt)])
    cli.main()
    data = json.loads(receipt.read_text(encoding='utf-8'))
    assert data == {'device': 'cpu', 'model': str(tmp_path / 'model.gguf'), 'runtime': str(tmp_path / 'runtime')}


def test_multilingual_json_survives_legacy_windows_stdout(monkeypatch):
    import io
    from contextlib import nullcontext
    stream = io.BytesIO()
    output = io.TextIOWrapper(stream, encoding='cp1252')
    monkeypatch.setattr(sys, 'stdout', output)
    class Model:
        def transcribe(self, path):
            return {'text': 'Привіт, світе.', 'segments': [], 'language': None}
    monkeypatch.setattr(cli, 'Orukeet', lambda *a, **k: nullcontext(Model()))
    monkeypatch.setattr(sys, 'argv', ['orukeet', 'transcribe', 'clip.wav', '--model', 'model.gguf', '--runtime', '.'])
    cli.main()
    output.flush()
    assert json.loads(stream.getvalue().decode('utf-8'))['text'] == 'Привіт, світе.'
    output.detach()
