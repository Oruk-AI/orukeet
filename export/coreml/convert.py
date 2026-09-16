#!/usr/bin/env python3
"""Install Orukeet weights into the pinned FluidAudio v3 Core ML graphs.

The model.mlmodel files remain byte-for-byte identical. Only weight blob payloads
change. Handles NeMo's folded batch norms, precomputed relative positions, LSTM
gate order, and the shipped encoder's actual 6-bit (not int8) palettes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import tarfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import yaml

SOURCE_SHA256 = "031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56"
REFERENCE_REPO = "FluidInference/parakeet-tdt-0.6b-v3-coreml"
REFERENCE_REVISION = "7dd20fe6b1797d35f5e3307e8b1732d9a178edfe"
COMPONENTS = {
    "Encoder": "mlpackages/Encoder.mlpackage",
    "Decoder": "mlpackages/Decoder.mlpackage",
    "JointDecisionv3": "JointDecisionv3.mlpackage",
}
GRAPH_SHA256 = {
    "Encoder": "cfec08b0fb04d25f7f61203e02571091e53d22fc85c47c5cc9699019d2ab0d2a",
    "Decoder": "8bfdcef9fb1b46a1ec2f3c1895c74ed90644a69105efda370e412a522bd5b353",
    "JointDecisionv3": "c31044bcbcc2959ced2a59e8a0177b617aa359fa1ad613b57de5b63a430f9e31",
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def operations(package: Path):
    spec = ct.utils.load_spec(str(package / "Data/com.apple.CoreML/model.mlmodel"))
    blocks = spec.mlProgram.functions["main"].block_specializations
    if len(blocks) != 1:
        raise ValueError("Expected a single pinned Core ML function block")
    return next(iter(blocks.values())).operations


def blob_info(handle, value) -> tuple[int, int]:
    if value.WhichOneof("value") != "blobFileValue":
        raise ValueError("Learned weights must use external blobs")
    if value.blobFileValue.fileName != "@model_path/weights/weight.bin":
        raise ValueError("Unexpected weight file")
    handle.seek(value.blobFileValue.offset)
    magic, _, size, offset = struct.unpack("<IIQQ", handle.read(24))
    if magic != 0xDEADBEEF or offset < value.blobFileValue.offset + 24:
        raise ValueError("Invalid Core ML blob header")
    return offset, size


def read_blob(handle, value, dtype) -> np.ndarray:
    offset, size = blob_info(handle, value)
    handle.seek(offset)
    return np.frombuffer(handle.read(size), dtype=dtype).copy()


def write_blob(handle, value, array: np.ndarray):
    offset, size = blob_info(handle, value)
    data = np.ascontiguousarray(array).tobytes()
    if len(data) != size:
        raise ValueError(f"Blob size mismatch: {len(data)} != {size}")
    handle.seek(offset)
    handle.write(data)


def pack_six_bit(indices: np.ndarray) -> np.ndarray:
    """MIL iOS17 LUT indices use little-endian bit packing, including tail bits."""
    if np.any(indices > 63):
        raise ValueError("6-bit palette index exceeds 63")
    bits = np.unpackbits(indices.astype(np.uint8).reshape(-1, 1), axis=1, bitorder="little")[:, :6]
    return np.packbits(bits.reshape(-1), bitorder="little")


def fit_palette(values: np.ndarray, initial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic weighted 1D Lloyd fit on the complete FP16 value histogram.

    FP16 has only 65,536 bit patterns. Clustering this histogram is equivalent
    to clustering all values with multiplicity, avoiding repeated passes over
    millions of weights. Start from the upstream palette to retain its scale.
    """
    half = np.ascontiguousarray(values, dtype=np.float16)
    if not np.isfinite(half).all():
        raise ValueError("Weights are not finite in FP16")
    histogram = np.bincount(half.view(np.uint16).ravel(), minlength=65536)
    occupied = np.flatnonzero(histogram)
    levels = occupied.astype(np.uint16).view(np.float16).astype(np.float64)
    order = np.argsort(levels, kind="stable")
    levels, counts = levels[order], histogram[occupied][order]
    centers = np.sort(initial.astype(np.float64))
    if len(centers) != 64 or not np.isfinite(centers).all():
        raise ValueError("Expected 64 finite palette centers")
    for _ in range(256):
        assignments = np.searchsorted((centers[:-1] + centers[1:]) / 2, levels)
        masses = np.bincount(assignments, weights=counts, minlength=64)
        sums = np.bincount(assignments, weights=counts * levels, minlength=64)
        updated = np.divide(sums, masses, out=centers.copy(), where=masses > 0)
        updated.sort()
        if np.array_equal(updated.astype(np.float16), centers.astype(np.float16)):
            centers = updated
            break
        centers = updated
    lut = centers.astype(np.float16)
    # Assign against rounded centers, exactly as the Core ML model sees them.
    boundaries = (lut[:-1].astype(np.float64) + lut[1:].astype(np.float64)) / 2
    codes = np.zeros(65536, dtype=np.uint8)
    codes[occupied[order]] = np.searchsorted(boundaries, levels).astype(np.uint8)
    return lut, codes[half.view(np.uint16)]


def reorder_lstm(tensor: torch.Tensor) -> torch.Tensor:
    # PyTorch: input, forget, cell, output. Core ML: input, forget, output, cell.
    i, f, g, o = tensor.chunk(4, dim=0)
    return torch.cat((i, f, o, g), dim=0)


class Weights:
    def __init__(self, state):
        self.state = state
        self.used: set[str] = set()
        self.names = {key.replace(".", "_"): key for key in state}
        positions = torch.arange(187, -188, -1, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, 1024, 2, dtype=torch.float32) * -(math.log(10000.0) / 1024))
        self.pe = torch.zeros(1, 375, 1024)
        self.pe[0, :, 0::2] = torch.sin(positions * div)
        self.pe[0, :, 1::2] = torch.cos(positions * div)

    def get(self, key):
        self.used.add(key)
        return self.state[key]

    def tensor(self, component: str, name: str) -> torch.Tensor | None:
        name = name.removesuffix("_palettized").removesuffix("_to_fp16")
        prefix = "joint_module_" if component == "JointDecisionv3" else "module_"
        source_prefix = {"Encoder": "encoder_", "Decoder": "decoder_", "JointDecisionv3": "joint_"}[component]
        if name.startswith(prefix):
            key = self.names[source_prefix + name.removeprefix(prefix)]
            return self.get(key)
        if component == "Encoder":
            if name in {"linear_1_bias_0", "linear_2_bias_0"}:
                return None  # Bias-free NeMo linears acquire shared zero biases.
            if name.startswith("op_"):
                number = int(name.removeprefix("op_"))
                layer, remainder = divmod(number - 273, 163)
                if remainder or not 0 <= layer < 24:
                    raise ValueError(f"Unexpected positional constant: {name}")
                matrix = self.get(f"encoder.layers.{layer}.self_attn.linear_pos.weight")
                projected = torch.nn.functional.linear(self.pe, matrix)
                return projected.reshape(1, 375, 8, 128).permute(0, 2, 3, 1).contiguous()
            if name.startswith("const_"):
                layer, is_bias = divmod(int(name.removeprefix("const_")) - 248, 2)
                if not 0 <= layer < 24:
                    raise ValueError(f"Unexpected folded convolution: {name}")
                prefix = f"encoder.layers.{layer}.conv."
                gamma = self.get(prefix + "batch_norm.weight")
                var = self.get(prefix + "batch_norm.running_var")
                scale = gamma / torch.sqrt(var + 1e-5)
                if is_bias:
                    return self.get(prefix + "batch_norm.bias") - self.get(prefix + "batch_norm.running_mean") * scale
                return self.get(prefix + "depthwise_conv.weight") * scale[:, None, None]
        if component == "Decoder" and name.startswith("concat_"):
            layer, which = divmod(int(name.removeprefix("concat_")), 3)
            if layer not in (0, 1):
                raise ValueError(f"Unexpected LSTM constant: {name}")
            prefix = "decoder.prediction.dec_rnn.lstm."
            if which == 0:
                tensor = self.get(prefix + f"bias_ih_l{layer}") + self.get(prefix + f"bias_hh_l{layer}")
            else:
                tensor = self.get(prefix + f"weight_{'ih' if which == 1 else 'hh'}_l{layer}")
            return reorder_lstm(tensor)
        raise ValueError(f"Unmapped learned constant: {component}/{name}")


def convert_component(source: Path, destination: Path, component: str, weights: Weights):
    if sha256(source / "Data/com.apple.CoreML/model.mlmodel") != GRAPH_SHA256[component]:
        raise ValueError(f"Unrecognized reference graph for {component}; use the pinned revision")
    shutil.copytree(source, destination)
    blob = destination / "Data/com.apple.CoreML/weights/weight.bin"
    receipts = []
    with blob.open("r+b") as handle:
        for op in operations(destination):
            if op.type == "constexpr_lut_to_dense":
                name = op.outputs[0].name
                tensor = weights.tensor(component, name)
                data = tensor.numpy()
                shape = tuple(d.constant.size for d in op.outputs[0].type.tensorType.dimensions)
                if data.shape != shape:
                    raise ValueError(f"Shape mismatch {name}: {data.shape} != {shape}")
                lut, indices = fit_palette(data, read_blob(handle, op.attributes["lut"], np.float16))
                reconstructed = lut[indices].astype(np.float32)
                write_blob(handle, op.attributes["lut"], lut)
                write_blob(handle, op.attributes["indices"], pack_six_bit(indices))
                delta = data - reconstructed
                receipts.append({"name": name, "encoding": "lut6", "elements": data.size,
                                 "max_abs_error": float(np.max(np.abs(delta))),
                                 "rmse": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2)))})
                print(component, name, flush=True)
            elif op.type == "const" and op.attributes["val"].WhichOneof("value") == "blobFileValue":
                name = op.outputs[0].name
                tensor = weights.tensor(component, name)
                if tensor is None:
                    if np.any(read_blob(handle, op.attributes["val"], np.float16)):
                        raise ValueError(f"Expected a zero bias: {name}")
                    continue
                data = tensor.numpy().astype(np.float16)
                shape = tuple(d.constant.size for d in op.outputs[0].type.tensorType.dimensions)
                if data.shape != shape or not np.isfinite(data).all():
                    raise ValueError(f"Invalid FP16 tensor {name}: {data.shape}, expected {shape}")
                write_blob(handle, op.attributes["val"], data)
                receipts.append({"name": name, "encoding": "fp16", "elements": data.size})
    graph = "Data/com.apple.CoreML/model.mlmodel"
    if (source / graph).read_bytes() != (destination / graph).read_bytes():
        raise ValueError(f"Graph changed for {component}")
    return {"graph_sha256": sha256(destination / graph), "graph_byte_identical": True,
            "weights_sha256": sha256(blob), "tensors": receipts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists; choose a new directory")
    if sha256(args.source) != SOURCE_SHA256:
        raise ValueError("Source is not the verified Orukeet r3 release")
    torch.set_num_threads(4)
    with tarfile.open(args.source) as archive:
        with archive.extractfile("./model_config.yaml") as handle:
            config = yaml.safe_load(handle)
        vocabulary = json.loads((args.reference / "parakeet_vocab.json").read_text())
        expected_vocabulary = {str(i): piece.replace("▁", " ") for i, piece in enumerate(config["joint"]["vocabulary"])}
        if vocabulary != expected_vocabulary:
            raise ValueError("The reference tokenizer differs from the Orukeet source")
        with archive.extractfile("./model_weights.ckpt") as handle:
            state = torch.load(handle, map_location="cpu", weights_only=True)
    weights = Weights(state)
    args.output.mkdir(parents=True)
    receipt = {"source_sha256": SOURCE_SHA256, "reference_repo": REFERENCE_REPO,
               "reference_revision": REFERENCE_REVISION, "torch": torch.__version__,
               "coremltools": ct.__version__, "components": {}}
    for component, relative in COMPONENTS.items():
        destination = args.output / f"{component}.mlpackage"
        receipt["components"][component] = convert_component(args.reference / relative, destination, component, weights)
        if not args.no_compile:
            ct.models.utils.compile_model(str(destination), destination_path=str(args.output / f"{component}.mlmodelc"))
    ignored = {key for key in state if key.endswith("num_batches_tracked") or key.startswith("preprocessor.")}
    missing = set(state) - weights.used - ignored
    if missing:
        raise ValueError(f"Source weights were not transferred: {sorted(missing)}")
    # Both models use the same front end and tokenizer. Copy these exact artifacts.
    shutil.copytree(args.reference / "Preprocessor.mlmodelc", args.output / "Preprocessor.mlmodelc")
    shutil.copytree(args.reference / "mlpackages/Preprocessor.mlpackage", args.output / "Preprocessor.mlpackage")
    shutil.copy2(args.reference / "parakeet_vocab.json", args.output / "parakeet_vocab.json")
    receipt["source_tensors_used"] = len(weights.used)
    receipt["vocabulary_matches_source"] = True
    receipt["source_tensors_ignored"] = sorted(ignored)
    receipt["status"] = "converted; accuracy and latency validation required"
    (args.output / "conversion.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
