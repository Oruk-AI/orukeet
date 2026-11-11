# Security policy

Orukeet is published by Oruk, Inc. Report security problems in this repository,
its released packages, the native runtime SDK archives or the published model
artifacts privately. Do not open a public issue for them.

- Email access@oruk.ai with the subject "Security report". Include the affected
  release tag or artifact SHA-256, the platform and reproduction steps.
- Oruk's disclosure contact and policy: https://oruk.ai/security and
  https://oruk.ai/.well-known/security.txt (RFC 9116).
- Responses are best effort. Oruk acknowledges reports and keeps the reporter
  informed until the issue is resolved or declined.

## Scope

- The `orukeet` Python package, its installer's download verification (pinned
  Hugging Face revisions and SHA-256 checks) and the native runtime SDK archives
  attached to GitHub releases.
- The model artifacts on https://huggingface.co/oruk/orukeet and their
  published hashes.

Transcription errors and accuracy regressions are not security issues; report
them through [GitHub issues](https://github.com/Oruk-AI/orukeet/issues/new/choose).
Questions about the hosted API (keys, data handling, infrastructure) are
answered at https://oruk.ai/security.
