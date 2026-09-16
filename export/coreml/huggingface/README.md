# Orukeet r3 Core ML preview

Portable Core ML bundles for Apple Silicon, macOS 14+, and **FluidAudio 0.15.5**.
These are the same archives as the
[GitHub preview release](https://github.com/Oruk-AI/orukeet/releases/tag/coreml-taptalk-preview-20260915),
published here for TapTalk and other applications to download.

| Archive | Profile | Bytes |
|:--|:--|--:|
| [orukeet-r3-coreml-greedy.zip](https://huggingface.co/oruk/orukeet/resolve/coreml-taptalk-preview-20260915/coreml/orukeet-r3-coreml-greedy.zip?download=true) | Recommended for ordinary greedy decoding | 466,579,943 |
| [orukeet-r3-coreml-baseline.zip](https://huggingface.co/oruk/orukeet/resolve/coreml-taptalk-preview-20260915/coreml/orukeet-r3-coreml-baseline.zip?download=true) | Retains top-64 outputs for language hints/reranking | 466,579,851 |

[SHA256SUMS.txt](https://huggingface.co/oruk/orukeet/resolve/coreml-taptalk-preview-20260915/coreml/SHA256SUMS.txt)
contains the archive hashes. Each archive includes a `bundle.json` with per-file
SHA-256 hashes, conversion receipts, attribution, and the weight license.

## Download

```sh
python -m pip install 'huggingface-hub>=0.34,<2'
```

```python
import hashlib
from pathlib import Path
from zipfile import ZipFile
from huggingface_hub import hf_hub_download

archive = Path(hf_hub_download(
    repo_id="oruk/orukeet",
    filename="coreml/orukeet-r3-coreml-greedy.zip",
    revision="coreml-taptalk-preview-20260915",
))
expected = "beccdc6f18c4b10527a764f6e3ab12e3e11b969220c0cee175b3bb7eaa94290e"
with archive.open("rb") as stream:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != expected:
    raise ValueError("Core ML archive checksum mismatch")
with ZipFile(archive) as bundle:
    bundle.extractall("models")
print(Path("models/orukeet-r3-coreml-greedy").resolve())
```

For an application downloader, use the archive URL above and verify its hash
before extraction. The archive has one top-level directory,
`orukeet-r3-coreml-greedy/` (or `orukeet-r3-coreml-baseline/`).

## Load with Swift

Each bundle contains `Preprocessor.mlpackage`, `Encoder.mlpackage`,
`Decoder.mlpackage`, `JointDecisionv3.mlpackage`, and `parakeet_vocab.json`.
Compile all four packages on the destination Mac during installation;
machine-specific `.mlmodelc` caches are deliberately excluded.

The `OrukeetCoreML` library is in `export/coreml/benchmark` at
[Oruk-AI/orukeet commit 347f646](https://github.com/Oruk-AI/orukeet/tree/347f646cacda2e001865b6ac40ba5cbc7e90d1c9/export/coreml/benchmark).
Add that directory as a local Swift package dependency, then:

```swift
import OrukeetCoreML

try OrukeetLocalModels.compilePackages(from: downloadedBundle, to: installedCache)
let engine = OrukeetEngine(modelDirectory: installedCache)
try await engine.ensureLoaded()
let result = try await engine.transcribe(samples: mono16kSamples)
print(result.text)
```

`downloadedBundle` is the extracted profile directory; `installedCache` is a
fresh app-owned directory. Audio must be mono 16 kHz PCM. Keep the engine loaded
between recordings and keep the Orukeet cache separate from Parakeet's.
Use the explicit local loader rather than FluidAudio's NVIDIA model downloader.

[Full conversion and integration guide](https://github.com/Oruk-AI/orukeet/blob/347f646cacda2e001865b6ac40ba5cbc7e90d1c9/export/coreml/README.md)
· [Source PR #6](https://github.com/Oruk-AI/orukeet/pull/6)
· [TapTalk](https://github.com/vakharwalad23/tap-talk)

## Measurements and scope

On one M5 Max, greedy decoding reduced paired batch latency by 9.5% relative
to the Core ML baseline. Baseline and greedy returned identical text on 128
FLEURS recordings across eight languages. Core ML WER was 7.64%, compared with
8.55% for Parakeet Core ML and 7.29% for the uncompressed Orukeet source.
[Raw evidence and methodology](https://github.com/Oruk-AI/orukeet/tree/347f646cacda2e001865b6ac40ba5cbc7e90d1c9/evidence/coreml-taptalk-20260915).

The baseline uses 6-bit LUT/FP16 weights; the greedy profile removes unused
top-64 joint calculations. It is unsuitable for decoding paths that require
those outputs. The experimental precision profile is not included.

This is a preview. Other Macs, older OS versions, power use, and the full
25-language Core ML evaluation remain unqualified. FluidAudio's default
sliding-window TDT path buffers roughly 13 seconds before first text. This is
not the separate Parakeet EOU 120M live-typing model.

## License and attribution

Orukeet modified weights: CC BY-SA 4.0, retaining NVIDIA Parakeet attribution.
Core ML graphs derive from Fluid Inference's pinned Parakeet conversion.
FluidAudio runtime: Apache-2.0. Conversion and integration code: MIT.
The bundles retain `LICENSE-WEIGHTS`, `NOTICE.md`, and `COREML-NOTICE.txt`.

All weights derive from the r3 checkpoint with SHA-256
`031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`.
Baseline graph metadata retains the upstream names for graph identity; use
`bundle.json` to identify the weights as Orukeet.
