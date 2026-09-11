# Orukeet Metal runtime

This directory owns Orukeet's Metal optimization patches and the script that
applies them to pinned NeMo-Speech.cpp and ggml sources, then builds a native
ASR SDK. No Hoid SDK release or adjacent source checkout is required.

The execution path is:

```text
Orukeet Python API → persistent worker → NeMo-Speech C ABI
                                         ↓
                                FastConformer / TDT graph
                                         ↓
                                   patched ggml Metal
```

The Python binding in [`src/orukeet/nvidia.py`](../src/orukeet/nvidia.py)
loads `lib/libnemo_speech_asr_c.dylib` from the supplied runtime directory.
The SDK retains C ABI version 1 and SDK version 0.1.0. Its embedded Metal
shader source includes the patches below. These changes apply to native
GGUF inference; the sherpa-onnx export uses a separate runtime.

## Build

Use an Apple silicon Mac running natively as `arm64`, Python 3.9 or newer,
Git, CMake 3.26 or newer, Ninja, and the Xcode Command Line Tools. Orukeet's
Python package itself requires Python 3.12 or newer.

```sh
# Install missing build tools first:
xcode-select --install
brew install cmake ninja

# From the Orukeet repository root:
python3 runtime/build-metal.py
```

The first run downloads three pinned source revisions. It builds SentencePiece
statically, so a Homebrew SentencePiece or Abseil installation is unnecessary.
Only ggml is initialized from NeMo's submodule set; CLI microphone capture,
TTS, NMT, servers, and unrelated submodules are not needed by this SDK.

Output is kept in the ignored `build/metal/` directory:

| Path | Contents |
| --- | --- |
| `sources/nemo/` | Pinned NeMo source with the attention patch applied |
| `sources/nemo/ggml/` | Pinned ggml with all 22 patches applied |
| `sources/sentencepiece/` | Pinned tokenizer dependency |
| `sources/prepared.json` | Source pins, patch hashes and resulting Git trees |
| `nemo-build/`, `sentencepiece-build/` | CMake build directories |
| `sdk/` | Installed headers, libraries, CMake package and license notices |
| `sdk/share/orukeet/build.json` | Build provenance and installed library hashes |

Rerunning the command verifies the prepared sources and incrementally builds
the SDK. A changed lock file, patch, or prepared source tree is rejected; the
script never resets an existing checkout. After updating pins or patches,
choose a fresh directory with `--work-dir build/metal-next`. `--jobs 4`
limits compilation parallelism.

To inspect the applied patches without compiling (also supported on Linux):

```sh
python3 runtime/build-metal.py --prepare-only
git -C build/metal/sources/nemo diff --cached
git -C build/metal/sources/nemo/ggml diff --cached
```

Optional `--nemo-repository`, `--ggml-repository`, and
`--sentencepiece-repository` arguments accept Git mirrors or absolute paths
to local repositories. Each must contain its locked commit. The script fetches
committed objects into its own checkout; it does not copy working-tree edits
or modify the supplied repositories. This also supports preparation without
network access when all three repositories are already available locally.

## Use the locally built SDK

With Orukeet installed and the released Q8 GGUF model already on disk:

```sh
orukeet transcribe recording.wav \
  --model /path/to/orukeet-v0.1.0-q8.gguf \
  --runtime "$PWD/build/metal/sdk" \
  --device metal
```

Or pass `runtime="build/metal/sdk", device="metal"` to the `Orukeet`
Python constructor. Model weights are separate from the SDK and are not
downloaded by the build script. The existing optional integration test uses
`ORUKEET_TEST_RUNTIME` to select this SDK; see [Contributing](../CONTRIBUTING.md).

Package version 0.1.1 selects the optimized SDK through
[`native_runtimes.json`](../src/orukeet/native_runtimes.json). Its release archive
must be published before that installer entry can be used. Local builds do
not depend on publication. See [release preparation](RELEASE.md).

## Source pins and patch order

[`sources.lock.json`](sources.lock.json) is the source and patch manifest:

| Source | Commit |
| --- | --- |
| NVIDIA/NeMo-Speech.cpp | `a5b6953c4a579a2bbd1c0913ad8a85c2a4d99953` |
| ggml-org/ggml | `c03b4e2bcece5134827881af90242086daf75be5` |
| google/sentencepiece | `17d7580d6407802f85855d2cc9190634e2c95624` |

Preparation verifies that NeMo's ggml submodule pin matches the manifest,
then applies these patches with `git apply --index`, without merging conflicts:

1. NeMo's existing `ggml-patches/0001` through `0020`, in filename order.
   These come from the pinned baseline, including the Metal dynamic-K fix
   in `0018`; they are not duplicated in this directory.
2. [`0021-metal-asr-kernels.patch`](ggml-patches/0021-metal-asr-kernels.patch):
   flat im2col and F32 copy kernels, and improved short depthwise dot-product
   dispatch.
3. [`0022-metal-asr-fusions.patch`](ggml-patches/0022-metal-asr-fusions.patch):
   short-convolution and LSTM gate fusion, short Q8 matrix-vector dispatch,
   and accompanying ggml regression cases.
4. [`0001-metal-relative-position-cache.patch`](nemo-patches/0001-metal-relative-position-cache.patch)
   against NeMo: attention views, relative-shift views, and precomputed
   constant positional projections.

The ggml patches are copied byte-for-byte from
[Hoid PR #2](https://github.com/hoid-ai/NeMo-Speech.cpp/pull/2), at
`ed05e13681b330b080edb797cb87b055540c3666`. The NeMo patch is the exact diff
of `rel_pos_attention.cpp` and `.h` from the baseline to that commit.
The manifest verifies every local patch's SHA-256 before any source fetch.

The CMake flag `NEMO_SPEECH_GGML_PATCHED=OFF` matches the qualified Metal
build: it selects the portable ASR graph instead of CUDA-specific fused
operations. It does **not** disable the applied Metal backend patches.

## Optimizations and evidence

The positional cache precomputes constants, not audio-dependent activations.
It uses 95.9 MiB for Orukeet Q8, with a 128 MiB total cap, and applies to
5–512 encoder frames (about 41 seconds at this model's subsampling rate).
Other eligible-size checks and fallback paths remain in the patches.
Set `NEMO_SPEECH_METAL_POS_CACHE_DISABLE=1` before starting the worker to
disable the positional cache.

The original M4 Pro qualification measured native median latency falling
from 154.2 ms to 88.9 ms across 24 English clips of 5–30 seconds. The
full-model audit found matching transcripts and all tested final scores
within `atol=rtol=1e-5`. Scaled kernel stress cases had exceptions at tighter
tolerances; the numerical report preserves those limits. These measurements
describe the qualified PR build, not every machine or a newly compiled SDK.

- [Performance and reproduction commands](https://github.com/hoid-ai/NeMo-Speech.cpp/blob/ed05e13681b330b080edb797cb87b055540c3666/docs/development/orukeet-metal-followup-benchmark.md)
- [Full-model numerical audit](https://github.com/hoid-ai/NeMo-Speech.cpp/blob/ed05e13681b330b080edb797cb87b055540c3666/docs/development/orukeet-model-numerics.md)
- [Kernel numerical audit and limits](https://github.com/hoid-ai/NeMo-Speech.cpp/blob/ed05e13681b330b080edb797cb87b055540c3666/docs/development/orukeet-metal-numerics.md)

## Attribution

These optimizations originate in Hoid's NeMo-Speech.cpp contribution. NeMo's
source retains NVIDIA's Apache-2.0 notices and Jason Ni's MIT attribution
for code derived from parakeet.cpp. The NeMo patch remains under those source
terms; see [NeMo's license](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/a5b6953c4a579a2bbd1c0913ad8a85c2a4d99953/LICENSE) and the fetched source's
`THIRD_PARTY_NOTICES.md`. The ggml patches retain ggml's
[MIT license](https://github.com/ggml-org/ggml/blob/c03b4e2bcece5134827881af90242086daf75be5/LICENSE). The build script follows Orukeet's
[MIT license](../LICENSE).

The installed SDK preserves NeMo and ggml license files and adds the notices
for the statically linked SentencePiece and its bundled dependencies. Keep
these notices when redistributing a built SDK. Source pinning fixes the input
revisions; byte-identical binaries also require matching build toolchains.
