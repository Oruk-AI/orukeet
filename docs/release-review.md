# Orukeet release scope

Orukeet is checkpoint **r3**, with 12,288 fitted, frozen Gabor kernels. The NeMo source, native Q8 and native F16 all derive from this checkpoint. [The artifact catalog](../src/orukeet/artifacts.json) pins their hashes and immutable download revision. The model cards, inference package and technical report describe these weights and their 74 paired benchmark splits.

The release contains MIT inference and training code, CC BY-SA 4.0 weights and fitted kernels, and CC BY 4.0 metric records. Dataset audio comes from its original providers. Historical checkpoints, raw research archives and internal launch materials remain in the private research history.

[Model card](../MODEL_CARD.md) · [Report](technical-report.md) · [Kernel fitting](gabor-surgery.md) · [Source inference](gabor-source.md)

The separate OpenWhispr patch targets OpenWhispr/openwhispr v1.9.2 and uses the same r3 Q8 export. The model release serves general ASR independently of that integration.
