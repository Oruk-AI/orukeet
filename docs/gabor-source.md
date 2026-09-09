# Run the Orukeet source model

Orukeet is a 25-language speech recognizer built from NVIDIA Parakeet TDT 0.6B v3. It replaces half of the encoder's temporal depthwise filters with **12,288 fitted, frozen Gabor kernels** and trains the remaining parameters on multilingual and multi-accent data.

Orukeet outperforms Parakeet on **61 of 74 tested splits**, including LibriSpeech test-clean (**1.46% vs. 1.53% WER**), test-other (**2.86% vs. 3.14%**), and FLEURS English (**3.82% vs. 4.28%**). Across all 25 FLEURS languages, pooled WER is **9.85% vs. 11.01%**, a **10.6% relative reduction**. Final adaptation and checkpoint selection use LibriSpeech test-other.

Use Orukeet for recordings, media, batch transcription, server workers and interactive applications. NeMo, Q8 and F16 all derive from the same **r3 release checkpoint** (`031c8ddab484`).

## Run Orukeet with NeMo

Use a CUDA-enabled PyTorch environment with `nemo_toolkit[asr]==3.0.0` and `huggingface-hub`. The [recorded source environment](https://github.com/Oruk-AI/orukeet/blob/main/evidence/standard-asr-20260908/runtime.json) lists the exact package versions used for evaluation.

```python
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import ASRModel

checkpoint = hf_hub_download(
    "oruk/orukeet", "orukeet-v0.1.0.nemo",
    revision="555136b50265a132d4cea0d35560c26fc4f657ab",
)
asr = ASRModel.restore_from(checkpoint)
asr.eval()
print(asr.transcribe(["recording.wav"], return_hypotheses=True)[0].text)
```

`orukeet fetch source` retrieves the same hash-checked checkpoint. Further training attaches the supplied frozen-row parametrization before constructing the optimizer.

## Model files

| Format | File | Bytes |
|:--|:--|--:|
| NeMo source | `orukeet-v0.1.0.nemo` | 2,509,342,720 |
| Native Q8 | `orukeet-v0.1.0-q8.gguf` | 714,456,704 |
| Native F16 | `orukeet-v0.1.0-f16.gguf` | 1,296,681,088 |

All three files derive from **r3**. Immutable weight revision: `555136b50265a132d4cea0d35560c26fc4f657ab`.

- NeMo SHA-256: `031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`
- Q8 SHA-256: `93ce19c6d8244acbfea980eeaf970531d4f216171578ef8e041dcc2d070a45bd`
- F16 SHA-256: `de53fb8ec251fb07ade15baabe17b00774ae3f1112f8618b062337f90fb49194`

Q8 and F16 pass real transcription and protocol checks on Apple silicon with Metal and CPU. Conversion audits verify all 12,288 fitted kernels after F16 rounding. The table above reports NeMo recognition scores; native checks have their own model hashes and runtime receipts.

[Artifact catalog](../src/orukeet/artifacts.json) · [Native conversion and validation](../evidence/r3-promotion-20260908/)

## Further training

Attach [FrozenGaborRows](../training/gabor_half/README.md) before constructing the optimizer. The fitted rows stay fixed while the remaining parameters train. [The r3 recipe](../training/librispeech_ft/README.md) records the final continuation.
