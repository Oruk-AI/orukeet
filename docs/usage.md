# Use Orukeet for speech recognition

Orukeet is a general-purpose ASR model. The same weights can serve a file
transcription tool, a media pipeline, a background job or an interactive
application. The current package accepts local audio files; your application
owns uploads, storage, scheduling and user access.

## Choose an artifact

| Artifact | Use | Runtime |
| --- | --- | --- |
| Q8 GGUF, 714 MB | Compact native inference | Pinned NeMo-Speech.cpp SDK, installed by `orukeet install` |
| F16 GGUF, 1.30 GB | Half-precision native inference | The same pinned NeMo-Speech.cpp SDK |
| Source `.nemo`, 2.51 GB | NeMo inference, adaptation and model inspection | The separate NeMo training/inference environment |

Q8 is the default distribution format. A GGUF file does not imply compatibility
with llama.cpp or a generic language-model loader. This export requires the
speech runtime specified in [artifact metadata](../src/orukeet/artifacts.json).
The `.nemo` checkpoint preserves the tokenizer and configuration, but is not
a complete optimizer-state snapshot.

To select F16, use the catalog's filename and checksum explicitly. This installs
the same runtime as the Q8 quickstart and loads the verified F16 export:

```python
from pathlib import Path
import json
import orukeet
from orukeet import Orukeet
from orukeet.install import fetch, install_runtime, resolve_device

cache = Path("./orukeet-cache")
device = resolve_device("auto")
runtime = install_runtime(device, cache)
model_path = fetch("f16", cache)
catalog = json.loads(Path(orukeet.__file__).with_name("artifacts.json").read_text())
with Orukeet(model_path, runtime, device,
             expected_sha256=catalog["files"]["f16"]["sha256"]) as asr:
    print(asr.transcribe("recording.wav")["text"])
```

The release has one quantization, Q8, plus an F16 representation and the NeMo
source. All three contain r3.

## Files and batch jobs

Follow the [installation quickstart](../README.md#run-speech-recognition).
From a source checkout, you can pass several files to the example runner:

```sh
python examples/transcribe.py interview.wav lecture.flac --installation installation.json
```

The runner loads one recognizer and emits one JSON object per file. It decodes
each file to mono 16 kHz audio, uses bounded windows for long inputs and returns
text plus segment times in seconds on the original recording's timeline.
Files run sequentially; this command is not a vectorized GPU batch decoder.
It makes no downloads during recognition.

## Application workers

Keep the model loaded across requests. Construction verifies the model hash
and starts the native worker, so putting it inside a per-request handler adds
unnecessary startup cost.

```python
import json
from pathlib import Path
from orukeet import Orukeet

config = json.loads(Path("installation.json").read_text(encoding="utf-8-sig"))
with Orukeet(config["model"], config["runtime"], device=config["device"]) as asr:
    for audio_path in ["interview.wav", "lecture.flac"]:
        result = asr.transcribe(audio_path)
        print(result["text"])
        for segment in result["segments"]:
            print(segment["start"], segment["end"], segment["text"])
```

Use one instance per application worker and account for the model memory of
each worker. An instance serializes requests. `close()` can interrupt a running
request; construct a new instance before using it again. A service can place
this API behind its own job queue or HTTP endpoint. The package does not ship
a production web server, authentication or concurrent request scheduler.

Segment times are useful building blocks for media tools; caption layout and
speaker labels belong to the application. The native API returns `language`
as `null`, and does not expose calibrated confidence or speaker diarization.
Do not silently turn a missing value into a detected language.

## Source-model inference and fine-tuning

For the fitted, frozen kernels, download the [Gabor NeMo source](gabor-source.md).
`orukeet fetch source` downloads the same fitted-Gabor Orukeet checkpoint. In the
[recorded NeMo environment](../training/README.md), restore that local path:

```python
from nemo.collections.asr.models import ASRModel

model = ASRModel.restore_from("/path/to/orukeet-v0.1.0.nemo", map_location="cpu")
model.eval()
```

The [reference evaluator](../training/evaluate.py) contains the actual NeMo
transcription/decoding setup; use its settings to reproduce source results.
The training guide covers initialization, the anchored continuation, parameter
blending and the states recovered from earlier experiments. Q8 is an inference
export; start adaptation from the source checkpoint in the NeMo stack.

## Model and format measurements

All current downloads derive from r3. The [model card](../MODEL_CARD.md) and [report](technical-report.md) describe the source and native measurements, language set and input/output behavior.

## Download sources and counting

All released weights are hosted by [oruk/orukeet on Hugging Face](https://huggingface.co/oruk/orukeet).
The native installer and `orukeet fetch source`, `orukeet fetch q8`, and
`orukeet fetch f16` use `hf_hub_download` with pinned revisions and SHA-256 checks.
GitHub supplies the Python package and native SDK, which contain no model weights.

For ONNX, use [the repository downloader](../examples/download_onnx.py) as shown
in the [quickstart](../README.md#sherpa-onnx-inference). It downloads the pinned
`onnx/manifest.json` and uses it to verify the archive. The manifest also provides
hashes for each extracted file.

[Hugging Face counts downloads on its servers](https://huggingface.co/docs/hub/models-download-stats).
GGUF downloads are counted directly; the repository's NeMo library metadata
counts `.nemo` and `.json` files, including the ONNX manifest. A direct ONNX
archive download without that manifest is not covered by these published rules.
Repository clones and Python package installs alone are not model downloads.
Cached weights and offline transcription remain available without forced
downloads or requests made only to increment a counter.
