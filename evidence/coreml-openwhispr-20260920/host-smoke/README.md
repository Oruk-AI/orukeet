> The checked-in engine-smoke.json is historical, from unmodified FluidAudio 0.15.5. Running this updated reproducer uses the current package dependency.

# Reproduce the macOS engine lifecycle smoke

This standalone package retains the runner used for [`../engine-smoke.json`](../engine-smoke.json), with its machine-specific model/audio/output paths replaced by positional arguments and its package dependency made relative. The inference sequence and assertions are unchanged. It is outside the shipping Swift package and contains no models or audio.

Requirements: Apple Silicon Mac, macOS 14+, Swift 6+, an existing **compiled Orukeet bundle (greedy or INT8)** suitable for that Mac, and two existing 16 kHz mono WAV fixtures. The short fixture must decode to 0.3–15 seconds. The long fixture must contain speech beyond 15 seconds. This runner loads local models; it does not download weights. SwiftPM resolves the shipping package's pinned FluidAudio 0.15.5-orukeet.1 dependency.

From the repository root, set paths to existing files, then run:

```sh
export ORUKEET_TEST_MODELS='/path/to/existing/compiled/greedy'
export ORUKEET_TEST_AUDIO='/path/to/en_us-1042003289011443756.wav'
export ORUKEET_TEST_LONG_AUDIO='/path/to/yc-first-30s.wav'
export ORUKEET_SMOKE_OUTPUT='/tmp/orukeet-engine-smoke.json'
swift run --package-path evidence/coreml-openwhispr-20260920/host-smoke EngineSmoke \
  "$ORUKEET_TEST_MODELS" "$ORUKEET_TEST_AUDIO" "$ORUKEET_TEST_LONG_AUDIO" "$ORUKEET_SMOKE_OUTPUT"
```

The original run used the default **debug** build, encoder CPU+Neural Engine, and batch concurrency 1. The runner checks six calls: short audio; its repeat; natural long audio; short audio after long audio; the first 15 seconds of long audio; and short audio after unloading/reloading. It requires identical short transcripts across the lifecycle. To detect accidental truncation, the full long transcript must have more normalized words than the first-window transcript, share its first eight normalized words, and contain a final eight-word suffix absent from the first-window transcript. These fixture-specific smoke assertions are not an accuracy benchmark.

The runner emits its measured results to the requested JSON path and stdout. The committed evidence additionally records fixture/source hashes, revision, architecture, timestamp, inference sequence, and limitations. Reproduce that metadata from the repository root with:

```sh
python3 - "$ORUKEET_SMOKE_OUTPUT" "$ORUKEET_TEST_AUDIO" "$ORUKEET_TEST_LONG_AUDIO" <<'PY'
import datetime, hashlib, json, pathlib, platform, subprocess, sys
output, short_audio, long_audio = map(pathlib.Path, sys.argv[1:])
engine = pathlib.Path('export/coreml/benchmark/Sources/OrukeetCoreML/OrukeetEngine.swift')
report = json.loads(output.read_text())
report.update({
    'verified_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'build_configuration': 'debug',
    'architecture': platform.machine(),
    'fluid_audio_version': '0.15.5-orukeet.1',
    'encoder_compute_units': 'cpuAndNeuralEngine',
    'short_audio_sha256': hashlib.sha256(short_audio.read_bytes()).hexdigest(),
    'long_audio_sha256': hashlib.sha256(long_audio.read_bytes()).hexdigest(),
    'engine_source_sha256': hashlib.sha256(engine.read_bytes()).hexdigest(),
    'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'inference_sequence': ['short', 'short_repeat', 'natural_long', 'short_after_long',
                           'long_first_15s', 'short_after_unload_reload'],
    'all_assertions_passed': True,
    'limitations': [
        'One macOS host lifecycle smoke, no iPhone measurements or accuracy score.',
        'Reported processing time excludes model loading. Peak RSS covers the entire smoke process.',
        'Audio durations use decoded PCM frame counts, not WAV header estimates.',
    ],
})
output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
PY
```

Only enrich metadata after the runner exits successfully. Check the resulting audio/source hashes against the committed JSON to establish whether the inputs and engine match. Timings, resident memory, model load time, and transcripts can vary by host, OS, and model bundle; the run does not establish iPhone performance.

The retained run read **91,520 samples / 5.72 s** from the short fixture and **479,212 samples / 29.95075 s** from the long fixture. `afinfo` reported 480,000 samples / 30 s for the long WAV container, while `AVAudioFile.read` produced 479,212 PCM frames. The runner uses the actual buffer `frameLength` after decoding; it does not pad the missing tail or substitute a historical header estimate. No manual trimming step is performed in this harness. Model loading is timed separately from inference, and peak RSS covers the entire process.
