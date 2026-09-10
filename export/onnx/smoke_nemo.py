#!/usr/bin/env python3
"""Record released-checkpoint NeMo transcripts for the same ONNX smoke audio."""
import argparse
import json
from pathlib import Path
import time

from export_orukeet import SOURCE_SHA256, sha256

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--wav", type=Path, action="append", required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
assert sha256(a.source) == SOURCE_SHA256
import nemo.collections.asr as nemo_asr
import torch
from omegaconf import OmegaConf

torch.set_num_threads(4)
model = nemo_asr.models.ASRModel.restore_from(str(a.source), map_location="cpu").cpu().float().eval()
model.preprocessor.featurizer.dither = 0.0
model.change_decoding_strategy(OmegaConf.create({"strategy": "greedy_batch", "greedy": {"max_symbols": 10}}))
result = {"source_sha256": SOURCE_SHA256, "runtime": "NeMo", "precision": "fp32", "provider": "cpu", "results": []}
with torch.inference_mode():
    for wav in a.wav:
        started = time.monotonic()
        out = model.transcribe(audio=[str(wav)], batch_size=1, return_hypotheses=True)
        if isinstance(out, tuple):
            out = out[0]
        row = {"file": wav.name, "sha256": sha256(wav), "text": out[0].text,
               "decode_seconds": time.monotonic()-started}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        result["results"].append(row)
a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
