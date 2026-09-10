#!/usr/bin/env python3
"""Decode real audio with FP32 and INT8 Orukeet using sherpa's existing loader."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import wave

import numpy as np
import sherpa_onnx


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_audio(path):
    with wave.open(str(path)) as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2
        return np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768, f.getframerate()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--wav", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    audio = [(path.name, *read_audio(path), sha256(path)) for path in a.wav]
    audio.append(("silence-3s", np.zeros(48000, np.float32), 16000, None))
    result = {"runtime": "sherpa-onnx", "version": importlib.metadata.version("sherpa-onnx"),
              "provider": "cpu", "threads": a.threads, "results": [], "batch_checks": []}
    for precision, suffix in [("fp32", ""), ("int8", ".int8")]:
        started = time.monotonic()
        r = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(a.model_dir / f"encoder{suffix}.onnx"),
            decoder=str(a.model_dir / f"decoder{suffix}.onnx"),
            joiner=str(a.model_dir / f"joiner{suffix}.onnx"),
            tokens=str(a.model_dir / "tokens.txt"),
            num_threads=a.threads, sample_rate=16000, feature_dim=128,
            decoding_method="greedy_search", provider="cpu", model_type="nemo_transducer")
        print(f"Loaded {precision} in {time.monotonic()-started:.3f}s", flush=True)
        for name, samples, sample_rate, sha in audio:
            started = time.monotonic()
            stream = r.create_stream()
            stream.accept_waveform(sample_rate, samples)
            r.decode_stream(stream)
            text = stream.result.text
            entry = {"precision": precision, "file": name, "sha256": sha,
                     "seconds": len(samples) / sample_rate, "text": text,
                     "decode_seconds": time.monotonic() - started}
            assert text.strip() if name != "silence-3s" else text.strip() == "", entry
            result["results"].append(entry)
            print(json.dumps(entry, ensure_ascii=False), flush=True)
        # The exported decoder has the same recurrent-state layout as stock.
        # Exercise sherpa's multi-stream entry point as well as serial calls.
        streams = []
        for name, samples, sample_rate, _ in audio[:2]:
            stream = r.create_stream()
            stream.accept_waveform(sample_rate, samples)
            streams.append(stream)
        started = time.monotonic()
        r.decode_streams(streams)
        texts = [s.result.text for s in streams]
        assert all(text.strip() for text in texts), texts
        result["batch_checks"].append({"precision": precision, "streams": 2,
                                       "texts": texts, "decode_seconds": time.monotonic()-started})
        del r
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
