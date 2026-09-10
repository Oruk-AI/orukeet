#!/usr/bin/env python3
"""Finalize TDT detection metadata without changing any ONNX tensor or node."""
import argparse
import json
from pathlib import Path

import onnx
from export_orukeet import graph_info, sha256, write_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--directory", type=Path, required=True)
a = p.parse_args()
metadata = json.loads((a.directory / "metadata.json").read_text())
metadata["url"] = "https://huggingface.co/oruk/orukeet#parakeet-tdt-v3"
write_json(a.directory / "metadata.json", metadata)
receipt = json.loads((a.directory / "export-receipt.json").read_text())
for name in ["encoder.onnx", "encoder.int8.onnx"]:
    path = a.directory / name
    if not path.exists():
        continue
    m = onnx.load(str(path), load_external_data=False)
    graph_before = m.graph.SerializeToString()
    onnx.helper.set_model_props(m, metadata)
    assert m.graph.SerializeToString() == graph_before
    onnx.save(m, str(path))
    receipt["graphs"] = [graph_info(path) if g["file"] == name else g for g in receipt["graphs"]]
receipt["metadata_finalization"] = {
    "reason": "sherpa-onnx 1.13.4 identifies TDT by 'tdt' in encoder metadata URL",
    "graph_nodes_and_tensors_unchanged": True,
    "script_sha256": sha256(Path(__file__)),
    "original_exporter_sha256": receipt["exporter_sha256"],
}
receipt["exporter_sha256"] = sha256(Path(__file__).with_name("export_orukeet.py"))
write_json(a.directory / "export-receipt.json", receipt)
