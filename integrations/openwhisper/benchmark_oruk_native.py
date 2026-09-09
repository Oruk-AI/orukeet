"""Benchmark an explicit GGUF through the app's persistent worker and audio path.

No downloads. The model path is overridden only inside this benchmark process.
"""
import argparse
import hashlib
import json
import platform
import statistics
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--whisper", choices=["base", "turbo"], help="Benchmark the existing Whisper baseline instead")
    parser.add_argument("--device", choices=["cpu", "cuda", "metal", "vulkan"], required=True)
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if not args.whisper and (args.model is None or args.runtime is None):
        parser.error("Native benchmarks require --model and --runtime")
    import numpy as np
    import psutil
    from faster_whisper.audio import decode_audio
    from transcriber.optional_backend import LocalSpeechBackend
    if args.whisper:
        from huggingface_hub import hf_hub_download
        from services.hf_access import resolve_model_repo
        args.model = Path(hf_hub_download(resolve_model_repo(args.whisper), "model.bin", local_files_only=True))
    with args.model.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    report = dict(platform=platform.platform(), machine=platform.machine(),
                  processor=platform.processor(), python=sys.version, device=args.device,
                  model=args.whisper or args.model.name, model_sha256=checksum, model_bytes=args.model.stat().st_size,
                  runtime="faster-whisper" if args.whisper else "NVIDIA/NeMo-Speech.cpp v0.1.0", repeats=args.repeats, results=[])
    peak = [0]
    stop = threading.Event()
    def monitor():
        process = psutil.Process()
        while not stop.wait(.02):
            try:
                peak[0] = max(peak[0], sum(p.memory_info().rss for p in [process, *process.children(recursive=True)]))
            except psutil.Error:
                pass
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    backend = LocalSpeechBackend("parakeet", "parakeet-v3", args.device)
    try:
        with patch("services.components.is_installed", return_value=True), \
             patch("services.components.component_dir", return_value=str(args.runtime.resolve()) if args.runtime else ""), \
             patch("services.local_asr.cache.is_cached", return_value=True), \
             patch("services.local_asr.cache.load_path", return_value=str(args.model.resolve())), \
             patch.object(backend, "_settings", return_value={}):
            started = time.perf_counter()
            if args.whisper:
                from transcriber.local_backend import LocalWhisperBackend
                backend = LocalWhisperBackend(args.whisper, device=args.device,
                                              compute_type="int8" if args.device == "cpu" else "float16")
            else:
                backend.reload_model()
            report["load_s"] = time.perf_counter() - started
            assert backend.is_available() and backend.device == args.device
            for path in args.audio:
                audio = decode_audio(str(path), sampling_rate=16000)
                duration = len(audio) / 16000
                timings, texts = [], []
                for _ in range(args.repeats + 1):
                    started = time.perf_counter()
                    texts.append(backend.transcribe(str(path)))
                    timings.append(time.perf_counter() - started)
                median = statistics.median(timings[1:])
                row = dict(audio=path.name, audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           duration_s=duration, first_s=timings[0], warm_s=timings[1:],
                           warm_median_s=median, warm_p95_s=float(np.percentile(timings[1:], 95)),
                           rtfx=duration/median, transcript=texts[-1], deterministic=len(set(texts)) == 1)
                report["results"].append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "transcript"}), flush=True)
    finally:
        backend.cleanup()
        stop.set()
        watcher.join()
        report["peak_process_tree_rss_bytes"] = peak[0]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
