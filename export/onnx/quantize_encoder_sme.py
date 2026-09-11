#!/usr/bin/env python3
"""Quantize the float FastConformer encoder graph to int8 in the form ONNX Runtime runs fastest.

Replaces quantize_dynamic(weight_type=QUInt8) for the encoder. Three changes, same trained weights:

1. Every MatMul with a constant 2-D weight becomes a com.microsoft.DynamicQuantizeMatMul node with
   symmetric int8 per-output-channel weights (zero point 0) and the following bias Add folded in.
   ONNX Runtime routes only this form to its KleidiAI SME2 kernels (Apple M4/M5) and I8MM kernels
   (M2/M3); asymmetric uint8 weights always fall back to the older NEON path.
2. The 1x1 pointwise Conv nodes of the conv modules are the same op with a transpose on each side
   (disable with --no-pointwise to keep them in float: slightly more accurate, ~20% slower).
3. Each layer's Q/K/V projections are merged into one N=3072 GEMM followed by a Split; the result is
   bit-identical and one wide GEMM is far more efficient on SME than three narrow ones.

Depthwise 9-tap convs and the 2-D subsampling convs stay in float.

Usage: quantize_encoder_sme.py encoder.onnx encoder.int8.onnx [--no-pointwise]
"""
import argparse
from collections import Counter, defaultdict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MS_DOMAIN = "com.microsoft"
QKV = ("linear_q", "linear_k", "linear_v")


def quantize_symmetric_per_channel(weight):
    """weight [K, N] float -> (int8 [K, N], scale [N])."""
    amax = np.abs(weight).max(axis=0)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    return np.clip(np.rint(weight / scale), -127, 127).astype(np.int8), scale


class Graph:
    def __init__(self, model):
        self.graph = model.graph
        self.init = {t.name: t for t in self.graph.initializer}
        self.consumers = defaultdict(list)
        for node in self.graph.node:
            for name in node.input:
                self.consumers[name].append(node)

    def constant(self, name):
        return numpy_helper.to_array(self.init[name]) if name in self.init else None

    def add_constant(self, name, array):
        tensor = numpy_helper.from_array(np.ascontiguousarray(array), name)
        self.graph.initializer.append(tensor)
        self.init[name] = tensor
        return name

    def bias_add_after(self, output, n):
        """The single Add consuming `output` with a constant bias of n elements, or None."""
        users = self.consumers.get(output, [])
        if len(users) != 1 or users[0].op_type != "Add":
            return None
        add = users[0]
        other = [name for name in add.input if name != output]
        bias = self.constant(other[0]) if len(other) == 1 else None
        if bias is None or bias.size != n:
            return None
        return add, bias.astype(np.float32).reshape(n)

    def quant_matmul_node(self, name, a, weight, bias, output):
        """DynamicQuantizeMatMul(a, weight_q8, scale, zero_point[, bias]) -> output."""
        q8, scale = quantize_symmetric_per_channel(weight)
        inputs = [a, self.add_constant(name + "_q8", q8), self.add_constant(name + "_scale", scale),
                  self.add_constant(name + "_zp", np.array(0, dtype=np.int8))]
        if bias is not None:
            inputs.append(self.add_constant(name + "_bias", bias.astype(np.float32)))
        return helper.make_node("DynamicQuantizeMatMul", inputs, [output], name=name, domain=MS_DOMAIN)


def convert_matmul(g, node, removed, stats):
    weight = g.constant(node.input[1])
    if weight is None or weight.ndim != 2 or weight.dtype != np.float32:
        return [node]
    output, bias = node.output[0], None
    folded = g.bias_add_after(output, weight.shape[1])
    if folded:
        add, bias = folded
        output = add.output[0]
        removed.add(id(add))
        stats["bias folded"] += 1
    stats["MatMul -> DynamicQuantizeMatMul"] += 1
    return [g.quant_matmul_node(node.name + "_dqmm", node.input[0], weight, bias, output)]


def convert_pointwise_conv(g, node, stats):
    weight = g.constant(node.input[1])
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    if weight is None or weight.ndim != 3 or weight.shape[2] != 1 or attrs.get("group", 1) != 1:
        return [node]
    c_out, c_in = weight.shape[:2]
    bias = g.constant(node.input[2]) if len(node.input) > 2 else None
    x_t, y_t = node.output[0] + "_xT", node.output[0] + "_yT"
    stats["pointwise Conv -> DynamicQuantizeMatMul"] += 1
    return [
        helper.make_node("Transpose", [node.input[0]], [x_t], perm=[0, 2, 1], name=node.name + "_xT"),
        g.quant_matmul_node(node.name + "_dqmm", x_t, weight.reshape(c_out, c_in).T, bias, y_t),
        helper.make_node("Transpose", [y_t], node.output, perm=[0, 2, 1], name=node.name + "_yT"),
    ]


def merge_qkv(g, nodes, stats):
    """Replace each layer's three Q/K/V DynamicQuantizeMatMul nodes (same input) by one GEMM + Split."""
    groups = defaultdict(list)
    for node in nodes:
        if node.op_type == "DynamicQuantizeMatMul" and any(k in node.name for k in QKV):
            groups[node.input[0]].append(node)
    for a, group in groups.items():
        if len(group) != 3:
            continue
        group.sort(key=lambda n: next(i for i, k in enumerate(QKV) if k in n.name))
        name = group[0].name.replace("linear_q", "linear_qkv")
        q8 = np.concatenate([g.constant(n.input[1]) for n in group], axis=1)
        scale = np.concatenate([g.constant(n.input[2]) for n in group])
        bias = np.concatenate([g.constant(n.input[4]) if len(n.input) > 4 else np.zeros(g.constant(n.input[1]).shape[1], np.float32) for n in group])
        merged = helper.make_node("DynamicQuantizeMatMul",
                                  [a, g.add_constant(name + "_q8", q8), g.add_constant(name + "_scale", scale),
                                   g.add_constant(name + "_zp", np.array(0, dtype=np.int8)), g.add_constant(name + "_bias", bias)],
                                  [name + "_Y"], name=name, domain=MS_DOMAIN)
        split = helper.make_node("Split", [name + "_Y"], [n.output[0] for n in group], axis=-1, name=name + "_split")
        position = min(nodes.index(n) for n in group)
        nodes = [n for n in nodes if n not in group]
        nodes[position:position] = [merged, split]
        stats["QKV merged"] += 1
    return nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--no-pointwise", action="store_true", help="keep the 1x1 pointwise convs in float")
    args = parser.parse_args()

    model = onnx.load(args.src, load_external_data=True)
    g = Graph(model)
    stats, removed, nodes = Counter(), set(), []
    for node in model.graph.node:
        if node.op_type == "MatMul":
            nodes += convert_matmul(g, node, removed, stats)
        elif node.op_type == "Conv" and not args.no_pointwise:
            nodes += convert_pointwise_conv(g, node, stats)
        else:
            nodes.append(node)
    nodes = [n for n in nodes if id(n) not in removed]
    nodes = merge_qkv(g, nodes, stats)

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {name for node in nodes for name in node.input} | {o.name for o in model.graph.output}
    for tensor in [t for t in model.graph.initializer if t.name not in used]:
        model.graph.initializer.remove(tensor)
    if not any(o.domain == MS_DOMAIN for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid(MS_DOMAIN, 1))
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"  nodes: {len(model.graph.node)}")
    onnx.save(model, args.dst)


if __name__ == "__main__":
    main()
