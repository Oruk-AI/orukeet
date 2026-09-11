# Prepare the prebuilt Metal runtime release

Orukeet's Python package and native SDK archive are attached to the same
GitHub release in `Oruk-AI/orukeet`. The package's Metal catalog entry pins
that archive's URL, SHA-256 and size. The SDK retains NeMo's 0.1.0 CMake
version and v1 C ABI; the Orukeet package release is 0.1.1. The r3 model
weights remain the existing v0.1.0 artifacts.

## Build and package

From an Apple silicon checkout, with the build prerequisites in
[README.md](README.md) installed:

```sh
python3 runtime/build-metal.py
python3 runtime/package-metal.py \
  --tag v0.1.1 --output-dir build/release-v0.1.1
```

The packager checks the source and patch manifest, build-script hash,
installed library hashes, ARM64 architecture, dynamic-library dependencies,
relative loader paths, and SDK completeness. It normalizes archive metadata
and preserves internal relative symlinks. Repackaging identical inputs in a
different output directory produces identical archive bytes.

It emits:

- `nemo-speech-orukeet-0.1.1-macos-arm64-metal.tar.gz`
- The archive's `.sha256` file
- `metal-runtime.json`, the complete replacement Metal platform entry
- `metal-build.json`, the build provenance already embedded in the SDK

The output directory must be outside the SDK. Existing archives are not
overwritten. Changed binaries require a new archive hash; after publication,
use a new version tag rather than replacing an existing archive.

The packager does not run inference, change the installer catalog, or publish
a release. Validate the candidate SDK with the optional native integration
test before publishing it:

```sh
ORUKEET_TEST_MODEL=/path/to/orukeet-v0.1.0-q8.gguf \
ORUKEET_TEST_RUNTIME="$PWD/build/metal/sdk" \
ORUKEET_TEST_AUDIO="$PWD/demos/fixtures/jfk.wav" \
ORUKEET_TEST_DEVICE=metal \
python -m pytest -q tests/test_inference.py::test_released_model_offline_reuse_and_silence
```

Test the extracted archive through the runtime installer too. The automated
archive tests in `tests/test_runtime_package.py` cover extraction, cache reuse,
symlinks and deterministic packaging; real inference needs the local model.

## Connect the package to its archive

Replace only `asr-nvidia-metal.platforms.macos_arm64` in
`src/orukeet/native_runtimes.json` with the generated `metal-runtime.json`.
Update the Python package version in `pyproject.toml` and
`src/orukeet/__init__.py`, then build its distributable files:

```sh
python -m pip install build
python -m build
```

The source distribution includes the runtime build and packaging scripts and
patches. The wheel includes the installer catalog. CPU, CUDA and Vulkan entries
continue to use their existing SDKs. The installer already keys its cache by
the complete runtime entry, so the new SDK gets its own cache directory.

Attach the SDK, checksum, catalog entry, provenance, Python wheel and source
distribution to a draft release targeting the reviewed commit. Publish the
release before merging an installer entry that points to its download URL;
draft assets cannot be downloaded anonymously by `orukeet install`.

## End-user installation after publication

With Python 3.12+ in an activated virtual environment:

```sh
python -m pip install --upgrade \
  https://github.com/Oruk-AI/orukeet/releases/download/v0.1.1/orukeet-0.1.1-py3-none-any.whl
orukeet install --device auto --cache ./orukeet-cache --output installation.json
python examples/transcribe.py recording.wav --installation installation.json
```

The last command uses `examples/transcribe.py` from a source checkout. An
application can instead pass the receipt's `model`, `runtime` and `device`
values to the `Orukeet` constructor. No CMake, Ninja or compiler is needed for
prebuilt installation. Downloading the SDK and the Q8 weights happens during
installation; recognition then runs locally.

Existing users must update their Python package and rerun `orukeet install`.
Their earlier `installation.json` still points to the old runtime until it is
regenerated. Publishing a GitHub release does not update PyPI automatically.
