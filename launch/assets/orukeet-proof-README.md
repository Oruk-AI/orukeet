# Orukeet r3 proof media

The 28-second video pairs the public 11-second JFK recording with the actual
r3 Q8 transcript, then shows **77 ms** for one warm file transcription on
**Apple M5 Max / Metal**. One warmup precedes the timed call. Exact elapsed time:
**0.07697083300445229 seconds**.

The measurement includes file decoding, native inference and worker IPC. Model
initialization and playback are outside the timed interval. The input plays at
its original speed from second 3 through second 14. The video presents a captured
file-transcription result; the separate OpenWhispr integration has its own tests.

Use [the MP4](orukeet-proof.mp4), [thumbnail](orukeet-proof-thumbnail.png),
[VTT](orukeet-proof.vtt), [SRT](orukeet-proof.srt) and
[accessible descriptions](orukeet-proof-alt-text.md) together. Video format is
1920×1080, 30 fps, H.264 with AAC audio and an embedded subtitle track.

The [captured receipt](orukeet-proof-receipt.json),
[render metadata](orukeet-proof-render.json) and
[verification](orukeet-proof-verification.json) identify the exact model,
recording, output, encoded audio alignment and captions. Introduction, audio
replay, timing and closing frames have been visually reviewed.

[Reproduce the video](../../demos/README.md) ·
[Audio provenance and notices](../../demos/fixtures/SOURCE.md) ·
[Presentation license: MIT](../../LICENSE)

Prepared privately for release review.
