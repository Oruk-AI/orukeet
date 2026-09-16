#!/usr/bin/env python3
"""Remove unused top-K outputs for greedy decoding; preserve decision operations.

This optional profile is for calls without language hints or vocabulary biasing.
Keep the baseline package when a caller needs the top-64 candidate outputs.
"""
import argparse
import json
import shutil
from pathlib import Path

import coremltools as ct

from convert import sha256


def prune_outputs(spec, keep):
    if set(keep) - {o.name for o in spec.description.output}:
        raise ValueError("Requested output is absent")
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            needed = set(keep)
            retained = []
            for op in reversed(block.operations):
                if op.blocks:
                    raise ValueError("This optimizer requires a straight-line joint graph")
                if needed.intersection(o.name for o in op.outputs):
                    retained.append(op)
                    for binding in op.inputs.values():
                        needed.update(a.name for a in binding.arguments if a.name)
            retained.reverse()
            del block.operations[:]
            block.operations.extend(retained)
            del block.outputs[:]
            block.outputs.extend(keep)
    outputs = [o for o in spec.description.output if o.name in keep]
    del spec.description.output[:]
    spec.description.output.extend(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists; choose a new directory")
    package = args.models / "JointDecisionv3.mlpackage"
    graph_relative = "Data/com.apple.CoreML/model.mlmodel"
    original = ct.utils.load_spec(str(package / graph_relative))
    before = next(iter(original.mlProgram.functions["main"].block_specializations.values()))
    before_ops = len(before.operations)
    prune_outputs(original, ["token_id", "token_prob", "duration"])
    after = next(iter(original.mlProgram.functions["main"].block_specializations.values()))
    if any(o.type == "topk" for o in after.operations):
        raise ValueError("Top-K still feeds a retained output")
    args.output.mkdir(parents=True)
    # Symlink large unchanged artifacts locally. The packaging script resolves
    # these links when preparing a self-contained distribution.
    for path in args.models.iterdir():
        if path.name not in {"JointDecisionv3.mlpackage", "JointDecisionv3.mlmodelc", "optimization.json"}:
            (args.output / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    destination = args.output / "JointDecisionv3.mlpackage"
    shutil.copytree(package, destination)
    ct.utils.save_spec(original, str(destination / graph_relative))
    ct.models.utils.compile_model(str(destination), destination_path=str(args.output / "JointDecisionv3.mlmodelc"))
    receipt = {"profile": "greedy-no-topk", "source_graph_sha256": sha256(package / graph_relative),
               "graph_sha256": sha256(destination / graph_relative), "operations_before": before_ops,
               "operations_after": len(after.operations), "outputs": list(after.outputs),
               "requires": "No language hints or top-K vocabulary reranking",
               "status": "candidate; exact-output and latency validation required"}
    (args.output / "optimization.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
