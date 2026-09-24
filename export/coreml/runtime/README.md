# Pinned FluidAudio buffer optimization

Use the Oruk fork's reviewed buffer change on the existing FluidAudio 0.15.5 baseline. The production change is confined to `TdtDecoderState.swift`: contiguous logical elements use bulk fills/copies, padded views preserve neighboring storage, and other copy layouts snapshot the source before writing so overlapping views are safe. The existing cache reset contract is preserved.

The fork is [Oruk-AI/FluidAudio](https://github.com/Oruk-AI/FluidAudio), tag [`0.15.5-orukeet.1`](https://github.com/Oruk-AI/FluidAudio/tree/0.15.5-orukeet.1), commit `ddc95f4d03d5be12bf84eeb6e4bde5b724356d88`. No GitHub Release announcement was created. Exact base, runtime-change commit, published revision, source hash and patch hash are recorded in `fluidaudio-runtime-lock.json`. Pin the release revision or exact prerelease tag selected by the package; do not depend on the moving branch.

The [upstream optimization PR #941](https://github.com/FluidInference/FluidAudio/pull/941) was still open when checked on September 21, 2026. Its updated head fixes bulk-span overruns but retains a copy fallback that assumes disjoint storage, and also changes cache, warmup and diarizer behavior. The latest official release, 0.15.8, does not contain this optimization. This fork takes the earlier narrow optimization plus the reviewed shared-view correction, avoiding those additional changes.

To reproduce the source patch on a separate clean FluidAudio checkout:

```sh
git checkout --detach 19600a485baa4998812e4654b70d2bab8f2c9949
git apply /path/to/orukeet/export/coreml/runtime/fluidaudio-0.15.5-safe-buffers.patch
swift build -c release --target FluidAudio --jobs 4
```

To check the actual helper implementation using Command Line Tools, without XCTest or model weights:

```sh
python3 export/coreml/runtime/check_buffer_helpers.py \
  --checkout /path/to/patched/FluidAudio --output /tmp/orukeet-buffer-checks.json
```

The runner extracts the production helper from the supplied checkout and compiles it against native Core ML. It covers all four scalar types, contiguous singleton views with excess backing storage, strided gaps/tails, both overlap directions, different-layout overlap, cross-type conversion and the complete 240,000-element preprocessor reset. `buffer-checks.json` records the local successful check and exact production hash. No model is loaded or downloaded.

The fork also has 13 added XCTest methods and a manual `Orukeet buffer safety` workflow. [CI run 35616517354](https://github.com/Oruk-AI/FluidAudio/actions/runs/35616517354) passed all 41 tests in the two affected classes and all 240,662 native helper assertions on macOS 15/Xcode 16.4; see `buffer-checks-ci.json`. With Xcode installed, run the complete two affected test classes with the debug configuration required by upstream test hooks:

```sh
swift test -c debug --jobs 4 --filter 'TdtDecoderStateV3Tests|MLArrayCacheTests'
```

The September 20 review measured the same corrected helper against the PR's then-current 0.15.8 base: 400/400 exact transcript matches and 20–26% lower transcription-call latency on three recordings. Those are historical host results, not new measurements of this 0.15.5 fork or iPhone latency. Current package/engine timing and transcript checks are recorded separately. This runtime change does not require new model weights or a new model format.
