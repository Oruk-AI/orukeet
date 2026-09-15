#!/usr/bin/env python3
"""Keep positional projections and folded temporal filters in FP16.

These tensors account for a small fraction of encoder storage but affect every
attention/temporal stage. The large pointwise and feed-forward weights retain
the baseline's 6-bit palettes. This is an accuracy/latency experiment.
"""
import argparse
import json
import shutil
import tarfile
from pathlib import Path

import coremltools as ct
from coremltools.libmilstoragepython import _BlobStorageWriter
import numpy as np
import torch

from convert import SOURCE_SHA256, Weights, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or sha256(args.source) != SOURCE_SHA256:
        raise ValueError("Choose a new output directory and the verified source")
    torch.set_num_threads(4)
    with tarfile.open(args.source) as archive:
        state = torch.load(archive.extractfile("./model_weights.ckpt"), map_location="cpu", weights_only=True)
    weights = Weights(state)
    args.output.mkdir(parents=True)
    for path in args.models.iterdir():
        if path.name not in {"Encoder.mlpackage", "Encoder.mlmodelc", "precision.json"}:
            (args.output / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    destination = args.output / "Encoder.mlpackage"
    shutil.copytree(args.models / "Encoder.mlpackage", destination)
    graph = destination / "Data/com.apple.CoreML/model.mlmodel"
    spec = ct.utils.load_spec(str(graph))
    ops = next(iter(spec.mlProgram.functions["main"].block_specializations.values())).operations
    extra = destination / "Data/com.apple.CoreML/weights/precision.bin"
    writer = _BlobStorageWriter(str(extra))
    changed = []
    for op in ops:
        name = op.outputs[0].name
        if op.type != "constexpr_lut_to_dense" or not name.startswith(("op_", "const_")):
            continue
        tensor = weights.tensor("Encoder", name).numpy().astype(np.float16)
        offset = writer.write_fp16_data(np.ascontiguousarray(tensor.reshape(-1)).view(np.uint16))
        op.type = "const"
        op.attributes.clear()
        value = op.attributes["val"]
        value.type.CopyFrom(op.outputs[0].type)
        value.blobFileValue.fileName = "@model_path/weights/precision.bin"
        value.blobFileValue.offset = offset
        changed.append({"name": name, "elements": tensor.size})
    del writer
    if len(changed) != 48:
        raise ValueError(f"Expected 24 positional projections and 24 temporal filters, got {len(changed)}")
    ct.utils.save_spec(spec, str(graph))
    ct.models.utils.compile_model(str(destination), destination_path=str(args.output / "Encoder.mlmodelc"))
    receipt = {"profile": "fp16-position-and-temporal", "source_sha256": SOURCE_SHA256,
               "graph_sha256": sha256(graph), "changed": changed,
               "additional_blob_bytes": extra.stat().st_size,
               "status": "candidate; quality and latency validation required"}
    (args.output / "precision.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
