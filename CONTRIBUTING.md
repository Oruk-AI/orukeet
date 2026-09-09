# Contributing

Start with a small reproducible case. For a transcription bug, include the model
SHA-256, OS/architecture, runtime/device, input format and expected behavior.
For speed claims, include the audio duration, cold versus warm state, every
timing sample and what the measurement includes. A median without its setup is
hard to use.

```sh
python -m pip install -e '.[dev,eval]'
python -m pytest
python training/selection/test_evaluation.py
```

Optional model tests use `ORUKEET_TEST_MODEL`, `ORUKEET_TEST_RUNTIME`,
`ORUKEET_TEST_AUDIO` (the pinned 11-second JFK clip) and `ORUKEET_TEST_DEVICE`.
They perform real offline inference; ordinary tests do not download weights.

[Package CI](evidence/package-ci.json) verifies tests, wheel/source builds and
CLI entry points on Windows, Linux and macOS. Local real-model checks cover
CPU and Metal; a separate [CUDA receipt](evidence/package-cuda.json) exercises
the wheel's native path on an A100. These are distinct from physical microphone
and application delivery checks.

Keep measured results immutable. Add a new receipt for a new experiment and
explain selection exposure. Changes to text normalization, membership or
clustering need a new evaluation identity, not an overwritten score.

Submit a reproducible issue or a focused pull request. Do not attach private
speech or credentials. Negative results are useful; keep data selection and
measurement scope explicit.

For independent accuracy or performance results, use the
[community evaluation format](evaluation/community/README.md). It includes
fixed-selection guidance, a result JSON template and an issue outline.
