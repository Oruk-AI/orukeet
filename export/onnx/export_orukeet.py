#!/usr/bin/env python3
"""Export the released Orukeet checkpoint in sherpa-onnx's Parakeet TDT layout.

The conversion sequence follows k2-fsa/sherpa-onnx's Parakeet TDT v3 exporter
at 11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf (Apache-2.0; see the bundled
LICENSE.sherpa-onnx). This wrapper pins source identity and records graph,
quantization, kernel, and numerical checks. It never changes a checkpoint.
"""

import argparse
import subprocess
import sys
import gc
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

SOURCE_SHA256 = "031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56"
UPSTREAM_COMMIT = "11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def graph_info(path):
    import onnx
    m = onnx.load(str(path), load_external_data=False)
    return {
        "file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
        "ir_version": m.ir_version,
        "opset_imports": {v.domain or "ai.onnx": v.version for v in m.opset_import},
        "inputs": [{"name": v.name, "type": onnx.helper.printable_type(v.type)} for v in m.graph.input],
        "outputs": [{"name": v.name, "type": onnx.helper.printable_type(v.type)} for v in m.graph.output],
        "metadata": {v.key: v.value for v in m.metadata_props},
        "operators": sorted({(v.domain or "ai.onnx") + "::" + v.op_type for v in m.graph.node}),
        "external_files": sorted({v.value for t in m.graph.initializer for v in t.external_data if v.key == "location"}),
    }


def set_metadata(path, metadata):
    import onnx
    m = onnx.load(str(path), load_external_data=False)
    onnx.helper.set_model_props(m, metadata)
    # Preserve the old sherpa/ONNX Runtime ABI. No post-IR-9 graph feature is
    # emitted by the legacy opset-17 exporter, including the INT8 quantizer.
    m.ir_version = min(m.ir_version, 9)
    onnx.save(m, str(path))


def audit_gabor(model, fits):
    import numpy as np
    text = gzip.open(fits, "rt").read() if fits.suffix == ".gz" else fits.read_text()
    records = json.loads(text)["records"]
    state = model.state_dict()
    selected = [r for r in records if r["selected"]]
    assert len(selected) == 12288
    t = np.arange(-4, 5, dtype=np.float64)
    for r in selected:
        A, mu, sigma, freq, phase = r["params"]
        shifted = t - mu
        expected = (A * np.exp(-0.5 * (shifted / sigma) ** 2) *
                    np.cos(2 * np.pi * freq * shifted + phase)).astype(np.float32)
        actual = state[r["name"]][r["channel"], 0].detach().cpu().numpy()
        assert np.array_equal(actual, expected), (r["name"], r["channel"])
    return {"fitted_rows_exact": len(selected), "coefficients_exact": len(selected) * 9,
            "fits_sha256": sha256(fits), "stored_shape": [1024, 1, 9],
            "storage": "ordinary torch Conv1d weight tensors; no custom Gabor operator"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--fits", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--skip-export", action="store_true", help="Resume quantization/validation of already exported FP32 graphs")
    a = p.parse_args()
    a.source = a.source.resolve()
    a.fits = a.fits.resolve()
    a.output = a.output.resolve()
    assert sha256(a.source) == SOURCE_SHA256, "This exporter is pinned to the released r3 checkpoint"
    a.output.mkdir(parents=True, exist_ok=True)
    os.chdir(a.output)

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260910)
    receipt = {"source_sha256": SOURCE_SHA256, "source_bytes": a.source.stat().st_size,
               "upstream_exporter_commit": UPSTREAM_COMMIT,
               "exporter_sha256": sha256(Path(__file__)),
               "python": platform.python_version(), "platform": platform.platform(),
               "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                            if d.metadata["Name"].lower().replace("-", "_") in
                            {"nemo_toolkit", "torch", "onnx", "onnxruntime", "numpy", "onnxscript"}},
               "threads": a.threads, "opset": 17, "exporter": "legacy TorchScript",
               "quantization": {"encoder": "dynamic QInt8 symmetric per-channel, DynamicQuantizeMatMul (quantize_encoder_sme.py)",
                                "decoder": "dynamic QInt8", "joiner": "dynamic QInt8"}}
    if not a.skip_export:
        import nemo.collections.asr as nemo_asr
        print("Loading released r3 checkpoint on CPU", flush=True)
        model = nemo_asr.models.ASRModel.restore_from(str(a.source), map_location="cpu")
        model = model.cpu().float().eval()
        assert sum(p.numel() for p in model.parameters()) == 627008134
        receipt["gabor_audit"] = audit_gabor(model, a.fits)
        durations = list(model.cfg.model_defaults.tdt_durations)
        assert durations == [0, 1, 2, 3, 4]
        receipt["tdt_durations"] = durations
        vocabulary = model.joint.vocabulary
        assert len(vocabulary) == 8192
        Path("tokens.txt").write_text("".join(f"{piece} {i}\n" for i, piece in enumerate(vocabulary)) + "<blk> 8192\n")
        sp = model.tokenizer.tokenizer
        Path("bpe.vocab").write_text("".join(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n" for i in range(sp.get_piece_size())))
        metadata = {"vocab_size": "8192", "normalize_type": str(model.cfg.preprocessor.normalize),
                    "pred_rnn_layers": str(model.decoder.pred_rnn_layers), "pred_hidden": str(model.decoder.pred_hidden),
                    "subsampling_factor": "8", "model_type": "EncDecRNNTBPEModel", "version": "2",
                    # sherpa 1.13.4 identifies TDT by the literal "tdt" in this
                    # metadata URL; the model's own URL must retain that marker.
                    "model_author": "NeMo", "url": "https://huggingface.co/oruk/orukeet#parakeet-tdt-v3",
                    "comment": "Orukeet r3; standard Parakeet TDT v3 transducer graph",
                    "feat_dim": "128", "orukeet_source_sha256": SOURCE_SHA256}
        write_json("metadata.json", metadata)
        with torch.inference_mode():
            # Different dynamic lengths catch time-axis hard-coding in the export.
            for frames in [257, 481]:
                features = torch.randn(1, 128, frames)
                lengths = torch.tensor([frames], dtype=torch.int64)
                encoded, encoded_lengths = model.encoder(audio_signal=features, length=lengths)
                np.savez(f"encoder-reference-{frames}.npz", features=features.numpy(), lengths=lengths.numpy(),
                         encoded=encoded.numpy(), encoded_lengths=encoded_lengths.numpy())
            for name, module in [("encoder", model.encoder), ("decoder", model.decoder), ("joiner", model.joint)]:
                started = time.monotonic()
                print(f"Exporting {name}", flush=True)
                module.export(f"{name}.onnx", onnx_opset_version=17, use_dynamo=False)
                print(f"Exported {name} in {time.monotonic()-started:.1f}s", flush=True)
        write_json("source-audit.json", receipt)
        del model
        gc.collect()
    else:
        receipt = json.loads(Path("source-audit.json").read_text())
    metadata = json.loads(Path("metadata.json").read_text())
    for name in ["encoder", "decoder", "joiner"]:
        path = Path(f"{name}.int8.onnx")
        print(f"Quantizing {name}", flush=True)
        if name == "encoder":
            # Symmetric int8 per-channel weights emitted directly as
            # com.microsoft.DynamicQuantizeMatMul (see quantize_encoder_sme.py).
            # Same 8-bit dynamic quantization, but in the form ONNX Runtime's CPU
            # provider routes to its SME2 (Apple M4/M5) and I8MM (M2/M3) GEMM
            # kernels; asymmetric uint8 weights are never eligible for those.
            subprocess.run([sys.executable, str(Path(__file__).with_name("quantize_encoder_sme.py")),
                            "encoder.onnx", str(path)], check=True)
        else:
            quantize_dynamic(f"{name}.onnx", str(path), weight_type=QuantType.QInt8)
        if name == "encoder":
            set_metadata(path, metadata)
            set_metadata(Path("encoder.onnx"), metadata)
        else:
            set_metadata(path, {"orukeet_source_sha256": SOURCE_SHA256})
        onnx.checker.check_model(str(path))
        gc.collect()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = a.threads
    opts.inter_op_num_threads = 1
    receipt["numerical_checks"] = []
    for precision in ["fp32", "int8"]:
        print(f"Checking {precision} graph inference", flush=True)
        name = "encoder.onnx" if precision == "fp32" else "encoder.int8.onnx"
        session = ort.InferenceSession(name, sess_options=opts, providers=["CPUExecutionProvider"])
        for frames in [257, 481]:
            ref = np.load(f"encoder-reference-{frames}.npz")
            feed = {session.get_inputs()[0].name: ref["features"], session.get_inputs()[1].name: ref["lengths"]}
            out = session.run(None, feed)
            assert np.array_equal(out[1], ref["encoded_lengths"])
            diff = out[0] - ref["encoded"]
            check = {"precision": precision, "feature_frames": frames,
                     "shape": list(out[0].shape), "finite": bool(np.isfinite(out[0]).all()),
                     "max_abs_error": float(np.abs(diff).max()), "mean_abs_error": float(np.abs(diff).mean()),
                     "relative_rms_error": float(np.linalg.norm(diff) / np.linalg.norm(ref["encoded"]))}
            assert check["finite"]
            if precision == "fp32":
                assert np.allclose(out[0], ref["encoded"], rtol=1e-3, atol=1e-3), check
            receipt["numerical_checks"].append(check)
        del session
        gc.collect()
    receipt["graphs"] = [graph_info(Path(f"{name}{suffix}.onnx"))
                         for name in ["encoder", "decoder", "joiner"] for suffix in ["", ".int8"]]
    receipt["tokens"] = {"count": 8193, "blank_id": 8192, "sha256": sha256("tokens.txt")}
    receipt["status"] = "pass"
    write_json("export-receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
