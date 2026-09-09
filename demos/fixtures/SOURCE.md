# Public audio fixture

`jfk.wav` contains the same 176,000 PCM samples used by the saved Orukeet
benchmarks: 11 seconds, mono, 16 kHz, signed 16-bit PCM. The WAV header was
rewritten in the original benchmark preparation. No private speech is included.

- WAV SHA-256: `d7d4e74b8a333ed02186008bc109a1b1a19d16da668bd56e785d80d69a16a72f`
- PCM SHA-256: `a29462b8ebd467318000e683b9117ade46230d3255ed2024e7db894abd9b38c9`
- The original NVIDIA WAV has SHA-256
  `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`.
  Its PCM is identical; the container bytes differ.

The immediate source is NVIDIA's
[NeMo-Speech.cpp fixture at the pinned converter revision](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/4f9676226f667d14608487df744f375db87127f8/test_files/asr/wav/test/jfk.wav).
Its [third-party notice](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/4f9676226f667d14608487df744f375db87127f8/THIRD_PARTY_NOTICES.md#whispercpp-sample-audio)
traces the sample to `ggml-org/whisper.cpp` revision
`23ee03506a91ac3d3f0071b40e66a430eebdfa1d`, and supplies the MIT notice for
the fixture. That notice is retained in [LICENSE-MIT](LICENSE-MIT).

The underlying speech is John F. Kennedy's Inaugural Address, January 20, 1961.
The [JFK Library's official sound-recording record](https://www.jfklibrary.org/asset-viewer/archives/jfkwha-001)
identifies **JFKWHA-001**, White House Audio Collection, as **Public Domain**.
The [official historic-speech page](https://www.jfklibrary.org/learn/about-jfk/historic-speeches/inaugural-address)
also identifies its inaugural-address motion-picture record as public domain.
These pages establish the official speech recording's status; the exact
11-second sample's redistribution chain is recorded separately above.

Preferred attribution: **White House Audio Collection. Swearing-in Ceremony
and Inaugural Address, 20 January 1961. John F. Kennedy Presidential Library
and Museum.** Sample via whisper.cpp and NVIDIA NeMo-Speech.cpp.

The video plays the supplied PCM at its original speed, with no noise
reduction, voice synthesis or change in pitch. Its AAC audio track is a lossy
encoding of that playback. The model always receives the verified PCM WAV.
The resulting transcript is model output, not a manually corrected reference.

Source records and native attribution were checked September 5–6, 2026.
