# Orukeet Core ML

Use the [iOS batch integration guide](../../integrations/openwhispr/ios/README.md) to add Orukeet to a Swift app for English and all 25 supported languages. It covers the Swift package, pinned portable model, installation, and 16 kHz mono record-then-transcribe flow.

The [validation record](../../evidence/coreml-openwhispr-20260920/README.md) separates build, simulator, and inference evidence from physical-device measurements. The [bundle verifier](verify_bundle.py) authenticates the released archive against the [pinned manifest](huggingface/manifest.json); it does not convert or download a training checkpoint.

The conversion research and original parity experiments remain in [PR #6](https://github.com/Oruk-AI/orukeet/pull/6), with [source and instructions pinned to its final commit](https://github.com/Oruk-AI/orukeet/tree/852c3e355f20a111a2ee39f76677bc0ba65147bf/export/coreml). They are separate from this app integration.
