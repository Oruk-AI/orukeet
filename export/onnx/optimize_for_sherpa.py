#!/usr/bin/env python3
"""Optimize the pinned Orukeet INT8 encoder without changing quantized values.

Nine-tap depthwise ConvInteger nodes become centered FP32 Conv nodes followed
by an INT32 cast. Each centered product is an integer of magnitude <= 65,025;
every partial sum has magnitude <= 585,225, below FP32's exact-integer limit.
The surrounding dynamic quantization, zero points and scales remain intact.

The original encoder and validated optimized encoder are both hash-pinned.
No custom operator, calibration data, training, or runtime dependency is added.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from export_orukeet import SOURCE_SHA256, graph_info, sha256, write_json

ORIGINAL_ENCODER_SHA256 = "d10711f1b8f3a516e2d7a93adb219caf8aba2b55295305db92f3e80e78c1499a"
OPTIMIZED_ENCODER_SHA256 = "7b55f2a504a20a8e462899f5befd45f4a1784948d76ed0127902d9cf39405487"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    args = parser.parse_args()
    source = args.model_dir / "encoder.int8.onnx"
    assert sha256(source) == ORIGINAL_ENCODER_SHA256
    assert not args.output.exists(), "Choose a fresh output directory"
    parent = json.loads(args.export_receipt.read_text())
    assert parent["status"] == "pass" and parent["source_sha256"] == SOURCE_SHA256
    parent_graphs = {g["file"]: g for g in parent["graphs"]}
    assert parent_graphs[source.name]["sha256"] == ORIGINAL_ENCODER_SHA256
    validation = json.loads(args.validation_receipt.read_text())
    assert validation["status"] == "completed" and validation["variant"] == "depthwise-fp32"
    assert validation["source_encoder_sha256"] == ORIGINAL_ENCODER_SHA256
    assert validation["sha256"] == OPTIMIZED_ENCODER_SHA256
    assert len(validation["operator_tests"]) == 24
    assert all(len(case["checks"]) == 3 and all(c["exact"] for c in case["checks"]) for case in validation["operator_tests"])
    assert {(c["batch"], c["frames"]) for c in validation["comparison"]["cases"]} == {(1, 257), (1, 481), (1, 1501), (2, 257)}
    assert all(c["equality"]["encoder_bit_exact"] and c["equality"]["lengths_exact"] for c in validation["comparison"]["cases"])

    model = onnx.load(str(source))
    tensors = {t.name: t for t in model.graph.initializer}
    nodes, constants, replaced = [], [], []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "ConvInteger":
            nodes.append(node)
            continue
        attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        weight = numpy_helper.to_array(tensors[node.input[1]])
        if not (weight.shape == (1024, 1, 9) and attributes.get("group") == 1024):
            nodes.append(node)
            continue
        assert attributes.get("kernel_shape") == [9]
        assert attributes.get("pads") == [0, 0]
        assert attributes.get("strides") == [1]
        assert attributes.get("dilations") == [1]
        assert weight.dtype == np.uint8 and len(node.input) == 4
        zero_point = numpy_helper.to_array(tensors[node.input[3]])
        assert zero_point.shape == () and zero_point.dtype == np.uint8
        prefix = f"orukeet_speed_{index}_"
        centered_name = prefix + "weight_centered_float"
        constants.append(numpy_helper.from_array(weight.astype(np.float32) - zero_point.astype(np.float32), centered_name))

        def add(kind, inputs, output, **kwargs):
            nodes.append(helper.make_node(kind, inputs, [output], name=output + "_node", **kwargs))
            return output

        activation = add("Cast", [node.input[0]], prefix + "activation_float", to=TensorProto.FLOAT)
        activation_zero = add("Cast", [node.input[2]], prefix + "activation_zero_point_float", to=TensorProto.FLOAT)
        centered_activation = add("Sub", [activation, activation_zero], prefix + "activation_centered_float")
        result = add("Conv", [centered_activation, centered_name], prefix + "sum_float", **attributes)
        add("Cast", [result], node.output[0], to=TensorProto.INT32)
        replaced.append(node.name)
    assert len(replaced) == 24
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(constants)
    args.output.mkdir(parents=True)
    target = args.output / source.name
    onnx.save(model, str(target))
    assert sha256(target) == OPTIMIZED_ENCODER_SHA256, "Output differs from the validated deployment graph"
    del model, tensors, nodes, constants
    gc.collect()
    onnx.checker.check_model(str(target))

    for name in ["decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt", "bpe.vocab", "LICENSE-WEIGHTS", "NOTICE.md"]:
        path = args.model_dir / name
        if name in parent_graphs:
            assert sha256(path) == parent_graphs[name]["sha256"]
        if name == "tokens.txt":
            assert sha256(path) == parent["tokens"]["sha256"]
        if path.exists():
            shutil.copy2(path, args.output / name)

    optimization = {
        "status": "pass", "source_sha256": SOURCE_SHA256,
        "original_encoder_sha256": ORIGINAL_ENCODER_SHA256,
        "optimized_encoder_sha256": OPTIMIZED_ENCODER_SHA256,
        "parent_export_receipt_sha256": sha256(args.export_receipt),
        "validation_receipt_sha256": sha256(args.validation_receipt),
        "optimizer_sha256": sha256(Path(__file__)),
        "onnx": onnx.__version__, "numpy": np.__version__,
        "method": "24 centered-integer FP32 depthwise Conv substitutions with INT32 outputs",
        "replaced_nodes": replaced, "taps_per_kernel": 9,
        "integer_product_magnitude_bound": 65025,
        "integer_partial_sum_magnitude_bound": 585225,
        "fp32_exact_integer_limit": 16777216,
        "original_quantized_initializers_retained": True,
        "dynamic_quantization_and_scales_unchanged": True,
        "model_metadata_io_and_opsets_unchanged": True,
        "new_runtime_dependencies": [],
        "validation": {"per_operator_cases": 72, "full_encoder_cases": len(validation["comparison"]["cases"]), "all_bit_exact": True},
    }
    optimization["optimized_graph"] = graph_info(target)
    optimization_path = args.output / "optimization-receipt.json"
    write_json(optimization_path, optimization)
    print(json.dumps({"encoder": str(target), "bytes": target.stat().st_size, "sha256": OPTIMIZED_ENCODER_SHA256,
                      "optimization_receipt": str(optimization_path), "original_export_receipt": str(args.export_receipt)}, indent=2))


if __name__ == "__main__":
    main()
