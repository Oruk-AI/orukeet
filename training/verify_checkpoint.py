#!/usr/bin/env python3
"""Verify that exported adaptation changed only the intended encoder layers."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

import torch


def read(path):
    with tarfile.open(path) as archive:
        weight_files = [m for m in archive.getmembers()
                        if m.isfile() and m.name.endswith("model_weights.ckpt")]
        if len(weight_files) != 1:
            raise ValueError("Ambiguous checkpoint weights")
        weights = torch.load(io.BytesIO(archive.extractfile(weight_files[0]).read()),
                             map_location="cpu", weights_only=True)
        tokens = {re.sub(r"^[0-9a-f]{32}_", "", Path(m.name).name):
                  hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                  for m in archive.getmembers() if m.isfile() and
                  Path(m.name).suffix in (".model", ".vocab", ".txt")}
    return weights, tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--top-layers", type=int, default=6)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    a, ta = read(args.reference)
    b, tb = read(args.candidate)
    if a.keys() != b.keys():
        raise ValueError("Exported state keys changed, or teacher leaked into model")
    if not ta or ta != tb:
        raise ValueError("Tokenizer artifacts changed")
    layer = re.compile(r"^encoder\.layers\.(\d+)\.")
    count = 1 + max(int(layer.match(k).group(1)) for k in a if layer.match(k))
    changed = []
    for key in a:
        if not torch.isfinite(b[key]).all():
            raise ValueError("Nonfinite tensor: " + key)
        if not torch.equal(a[key], b[key]):
            match = layer.match(key)
            if not match or int(match.group(1)) < count - args.top_layers:
                raise ValueError("Unexpected updated tensor: " + key)
            if key.endswith(("running_mean", "running_var", "num_batches_tracked")):
                raise ValueError("BatchNorm statistics changed: " + key)
            changed.append(key)
    if not changed:
        raise ValueError("No weight updates in exported candidate")
    result = {"status": "pass", "encoder_layers": count,
              "adapted_top_layers": args.top_layers, "changed_tensors": len(changed),
              "changed_parameter_elements": sum(a[k].numel() for k in changed),
              "state_keys_unchanged": True, "frozen_weights_bitwise_unchanged": True,
              "batchnorm_statistics_unchanged": True, "all_tensors_finite": True,
              "tokenizer_artifact_hashes": ta, "changed_keys": changed}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "changed_keys"}, indent=2))


if __name__ == "__main__":
    main()
