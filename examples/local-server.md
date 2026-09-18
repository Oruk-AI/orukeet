# Local transcription server

This example keeps one Orukeet recognizer loaded and serves completed recordings
at `POST /v1/audio/transcriptions`. It supports OpenAI-style multipart uploads and
JSON or plain-text responses. It does not stream partial transcripts.

## Install and start

Use Python 3.12+ on a platform supported by the [native runtime](../docs/usage.md).
No Oruk account or API key is needed for local use.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install https://github.com/Oruk-AI/orukeet/releases/download/v0.1.1/orukeet-0.1.1-py3-none-any.whl
curl -fLO https://raw.githubusercontent.com/Oruk-AI/orukeet/main/examples/local_server.py
curl -fLO https://raw.githubusercontent.com/Oruk-AI/orukeet/main/examples/local-server-requirements.txt
python -m pip install -r local-server-requirements.txt
orukeet install --device cpu --cache ./orukeet-cache --output installation.json
python local_server.py --installation installation.json
```

If you already have a valid `installation.json` from the local guide, reuse it
and skip installation. `--device auto` selects an available accelerated runtime
when installing; the server uses the device recorded in the receipt.

Installation verifies the native runtime and the official, pinned Q8 GGUF from
[Hugging Face](https://huggingface.co/oruk/orukeet). The actual GGUF request uses
Hugging Face's normal download accounting. Starting the server and transcribing
use local files, with no model download or audio upload to Oruk. Keep the model,
runtime, receipt and accompanying weight licenses together. The weights use
CC BY-SA 4.0; the example code is MIT.

Wait for `Orukeet ready; persistent native worker PID ...` and Uvicorn's startup
message. The default address is `http://127.0.0.1:8000`. Keep this process running:
one server process owns one model worker, which is reused for every recording.
Do not add Uvicorn `--workers` or `--reload`; each process would load another model.

## Send a recording

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=orukeet -F file=@recording.wav
# {"text":"Your transcription."}

curl --fail-with-body http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=orukeet -F file=@recording.wav -F response_format=text
```

WAV, FLAC, MP3, OGG, M4A/AAC and WebM audio are decoded locally with PyAV, mixed
to mono and resampled to 16 kHz. Upload names are never used as filesystem paths.
Playlists and external media references are rejected. The endpoint defaults to
JSON with a `text` field. Silence returns an empty string.

For the OpenAI Python client (`python -m pip install openai`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
with open("recording.wav", "rb") as audio:
    result = client.audio.transcriptions.create(model="orukeet", file=audio)
print(result.text)
```

`whisper-1` is accepted as a compatibility alias for **the same Orukeet model**.
Language is detected automatically: omit `language` or use `auto`. Forced
language, translation, nonempty prompts, timestamps and streaming are unsupported
and rejected. `temperature=0`/`0.0` and the legacy `temperature_inc=0.2` (or zero)
are accepted for client compatibility; they do not change greedy decoding.

## Emacs / whisper.el

`whisper.el`'s OpenAI mode uses a base URL **without** `/v1`; it adds the path
itself. Set a nonempty dummy key because that client expects one:

```elisp
(setq whisper-server-mode 'openai
      whisper-openai-api-baseurl "http://127.0.0.1:8000/"
      whisper-openai-api-key "local"
      whisper-openai-model "orukeet"
      whisper-language "auto"
      whisper-translate nil)
```

For Sacha Chua's custom queue, which builds its own URL and uses `whisper-model`:

```elisp
(setq whisper-server-host "127.0.0.1"
      whisper-server-port 8000
      whisper-model "orukeet"
      whisper-language "auto"
      whisper-translate nil
      sacha-whisper-url-format "http://%s:%s/v1/audio/transcriptions")
```

Sacha's existing custom queue does not retry HTTP 429 automatically. Submit one
segment at a time or add retry/backoff before using it with this bounded server;
otherwise an overloaded request is reported as an error.

These snippets configure completed-file requests. They do not start the server
or turn Orukeet into a streaming recognizer. Review the returned text before
using it as a command.

## Limits and failures

The default limits are 25 MiB for the entire multipart request, five minutes of
decoded audio, and two admitted requests total (one working, one queued).
Normalization and recognition run on one background thread, while `/health`
remains responsive. The native runtime also splits long audio into bounded
windows; there is no unbounded batch held in memory.

A full queue returns HTTP 429 with `Retry-After: 1`. Clients should retry or send
one recording at a time. A disconnected request keeps its slot until its native
work finishes, then deletes the upload. This prevents abandoned requests from
building an unbounded inference queue. Normal shutdown drains accepted work and
closes the model.

Change the limits with `--max-upload-mb`, `--max-seconds` and `--max-requests`.
Increasing request capacity increases disk use and queue latency, not throughput.
Invalid audio/options return 400; oversized uploads return 413. Inference failures
return 500 instead of a successful empty transcript. Requests from browser origins
are rejected. This is a local example, not a production multi-user service.

To require a real bearer key, set `ORUKEET_API_KEY` before starting the server and
configure that key in your client. Binding to a non-loopback address requires
that setting; use your own TLS/access controls if you expose it beyond your machine.

## Verify changes

From a source checkout after installing the example requirements:

```sh
python -m pip install pytest httpx
python -m pytest tests/test_local_server.py
```

These tests cover persistent-model reuse, JSON/text output, stereo normalization,
invalid audio and options, upload/duration limits, cancellation and bounded work.
They use a fake recognizer for deterministic HTTP/lifecycle checks; they are not
accuracy or speed benchmarks. Real-runtime receipts should record the package,
model SHA-256, device and actual outputs separately.

[Recorded CPU HTTP checks](../evidence/local-server-20260917.json) cover the real
OpenAI client, repeated inference, audio formats and offline serving with outbound
network access blocked. These establish the checked integration behavior, not an
accuracy or speed advantage.
