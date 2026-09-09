# Training and lineage

Orukeet **r3** is the source of the current NeMo, Q8 and F16 downloads.
It continues the FT-4035 parent for 168 low-learning-rate updates while keeping
all 12,288 fitted Gabor kernels fixed. Start with the
[final continuation recipe](librispeech_ft/README.md) and
[source inference guide](../docs/gabor-source.md). The
[artifact catalog](../src/orukeet/artifacts.json) pins every current download.

## Reproduce the reported scores

The current comparison covers 74 splits: 27 complete LibriSpeech/FLEURS splits
and 47 accent/domain splits. Integer error counts, model hashes and recording
identifiers accompany the results. From the repository root, run:

```sh
python evaluation/standard_asr/build_current_report.py --verify-pdf
```

This recomputes WER/CER, pooled summaries and all 74 paired table rows without
audio or a GPU. The Hub's `metric-evidence.tar.gz` contains the same r3 count
records and verifier under `metric-r3/`. Extract it and run the command above
from that directory. [All scores and methods](../docs/current-checkpoint-benchmarks.md)
identify the exact model and scoring configuration.

## Model ancestry

| Stage | Operation | Record |
| --- | --- | --- |
| Initial adaptation | 20,000-step Common Voice/FLEURS adaptation with EMA | `historical/train.sh` and the historical recipe archive |
| Cooldown | 4,000-step continuation with EMA | `configs/stage2_input_cfg.yaml` and the stage-2 artifact |
| Anchored continuation | 800 updates to the top six encoder layers | `finetune_anchored.py` and `evidence/training_argv.json` |
| Adaptation baseline | Blend floating parameters: 0.75 continuation + 0.25 Parakeet | `selection/blend_models.py` and the saved lineage |
| Gabor fitting and recovery | Fit every temporal depthwise row, replace the closest half and freeze them; recover the remaining parameters to R15-0100 | [Fitting recipe and experiment record](gabor_half/README.md) |
| Parent continuation | 4,035 updates at peak learning rate `1e-6`, producing FT-4035 | [Parent recipe](regression_ft/README.md) and [export audit](../evidence/regression-ft-20260907/export-audit.json) |

| Final r3 continuation | 168 updates at peak learning rate `5e-6`; LibriSpeech test-other supplies adaptation and selection | [Final recipe](librispeech_ft/README.md) and [export audit](../evidence/librispeech-ft-20260908/r3/export-audit.json) |

The final export audit checks the tokenizer, tensor shapes, normalization
buffers and every fitted coefficient. All 651 trainable parameter tensors
changed in the final continuation; the 110,592 Gabor coefficients remained
bit-exact. Inference exports store those coefficients as ordinary convolution
weights. The dense model has 627,008,134 parameters; 626,897,542 are trainable
when the frozen-row parametrization is installed.

For a new adaptation, restore the NeMo source and attach
`training/gabor_half/frozen.py`'s `install` function before creating the optimizer.
Use the committed fit sidecar, keep BatchNorm running statistics fixed, and
call its `verify` function while training. The final continuation trainer shows
the full setup, optimizer coverage checks, resume handling and export audit.
New data and updates produce a new checkpoint identity.

## Training environment and inputs

The recorded continuation environment uses one A100 40 GB with
`nvcr.io/nvidia/nemo-speech:26.07.00` and NVIDIA
[Speech](https://github.com/NVIDIA-NeMo/Speech) commit
`fd6a877539710e2b98f28c43272ff81312f83417`. Exact run arguments and package
versions accompany the experiment records. NeMo training uses this separate
environment; the small native inference package does not install it.

The parent run uses the 24-split plan in [regression_ft/README.md](regression_ft/README.md).
The final r3 pass uses 2,939 LibriSpeech test-other recordings, with native casing
and punctuation retained in its corrected targets. [The final recipe](librispeech_ft/README.md)
records the input identities, target construction, optimizer and frozen-row checks. Dataset
audio comes from the original providers under their
[source terms](../docs/data-and-licenses.md). The private research archive
retains exact manifests, predictions and optimizer state for review.

## Recover the earlier adaptation

The [historical artifact catalog](gabor_half/results/pre-gabor-artifacts.json)
records the earlier checkpoints and private input archives. The current
`orukeet fetch` command accepts only `source`, `q8` and `f16`; those names
always fetch r3. An authorized reviewer can recover historical inputs
with their separate repository and revision pins:

```python
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

catalog = json.loads(Path("training/gabor_half/results/pre-gabor-artifacts.json").read_text())
for name in ("stage2", "training-inputs", "historical-recipe"):
    item = catalog["files"][name]
    path = Path(hf_hub_download(
        item.get("repo_id", catalog["repo_id"]), item["path"],
        revision=item.get("revision", catalog["revision"]),
        local_dir="artifacts/historical",
    ))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert path.stat().st_size == item["size"] and digest == item["sha256"]
    print(name, path)
```

Extract verified archives with `tarfile.extractall(..., filter="data")`.
The recovered stage-3 recipe expects a working root mounted at
`/home/nathanroll/parakeet-ft`, the stage-2 file at
`models/ft/parakeet-tdt-0.6b-v3-ft-cv-fleurs.nemo`, and the pinned NVIDIA
checkout at `Speech`. Run `python training/launch.py` to print that historical
command; `--run` executes it. Relocated input paths need a derived configuration
with new hashes.

`audit_prepare.py`, `build_sampling.py` and `prepare_accent_extension.py`
record the earlier formatting, duplicate filtering and accent sampling.
`selection/blend_models.py` reproduces the adaptation baseline from its
recorded inputs. `evaluation/export_oruk_native.py` belongs to that baseline;
the current r3 conversion lineage is recorded separately in
`evidence/r3-promotion-20260908/`. Historical selection decisions and
measurements retain their original checkpoint identities.
