# Orukeet transcription demo

The [28-second video](../launch/assets/orukeet-proof.mp4) pairs an 11-second public JFK recording with the actual **r3 Q8** transcript, then shows a measured call through the release package. Audio plays at its original speed. The timing card records **77 ms on Apple M5 Max / Metal** after one warmup, with the model already loaded.

The measured interval includes file decoding, PCM preparation, worker IPC, native inference and result assembly. Model initialization and audio playback sit outside it. The complete [receipt](../launch/assets/orukeet-proof-receipt.json) records the model hash, runtime, transcript, word times, warmup and measured duration.

## Reproduce

Use the current Q8 file and installed native runtime. FFmpeg renders the video locally.

```sh
python -m pip install -e '.[dev]'
python demos/make_proof.py \
  --model /path/to/orukeet-v0.1.0rc1-q8.gguf \
  --runtime /path/to/installed-metal-sdk --device metal
python demos/verify_proof.py
```

The script verifies the current Q8 hash, runs one warmup and one measured call, then renders the actual returned transcript and timing. Each run produces its own measurement. Use `--render-existing` to render the saved result or `--output /path/to/folder` to keep a separate recording.

Outputs include the MP4, thumbnail, SRT/VTT captions, transcript, accessible description, captured result and verification receipts under `launch/assets/`. The verifier checks encoded audio alignment, captions, dimensions, duration and hashes. This is an edited presentation of file transcription; the OpenWhispr application demonstration is maintained with its separate private integration.

[Three-language listening kit](multilingual/README.md) · [Audio provenance and notices](fixtures/SOURCE.md) · [Earlier demo records](history/pre-gabor/README.md)
