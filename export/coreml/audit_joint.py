#!/usr/bin/env python3
"""Require byte-identical greedy outputs on evolving states from real recordings."""
import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf

from convert import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--audio", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = []
    for units in [ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_NE]:
        def load(directory, name):
            return ct.models.CompiledMLModel(str(directory / f"{name}.mlmodelc"), compute_units=units)
        pre = load(args.baseline, "Preprocessor")
        enc = load(args.baseline, "Encoder")
        dec = load(args.baseline, "Decoder")
        base = load(args.baseline, "JointDecisionv3")
        candidate = load(args.candidate, "JointDecisionv3")
        for path in args.audio:
            audio, rate = sf.read(path, dtype="float32")
            if audio.ndim != 1 or rate != 16000 or not 0 < len(audio) <= 240000:
                raise ValueError("Expected 16 kHz mono audio, 0 < duration <= 15s")
            mel = pre.predict({"audio_signal": np.pad(audio, (0, 240000 - len(audio)))[None], "audio_length": np.array([len(audio)], np.int32)})
            encoded = enc.predict(mel)
            h = np.zeros((2, 1, 640), np.float32)
            c = h.copy()
            target = 8192
            comparisons = 0
            for frame in range(int(encoded["encoder_length"].item())):
                prediction = dec.predict({"targets": np.array([[target]], np.int32), "target_length": np.array([1], np.int32), "h_in": h, "c_in": c})
                inputs = {"encoder_step": encoded["encoder"][:, :, frame:frame + 1].copy(), "decoder_step": prediction["decoder"]}
                expected, actual = base.predict(inputs), candidate.predict(inputs)
                for name in ["token_id", "token_prob", "duration"]:
                    a, b = expected[name], actual[name]
                    if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                        raise ValueError(f"Byte mismatch: {units.name}/{path.name}/{frame}/{name}")
                    comparisons += 1
                token = int(expected["token_id"].item())
                if token != 8192:
                    target, h, c = token, prediction["h_out"], prediction["c_out"]
            reports.append({"units": units.name, "audio_sha256": sha256(path), "audio": path.name,
                            "scalar_output_comparisons": comparisons, "byte_identical": True})
    args.output.write_text(json.dumps({"passed": True, "fixtures": reports}, indent=2) + "\n")


if __name__ == "__main__":
    main()
