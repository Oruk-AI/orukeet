"""Record a real local Orukeet call, then render a labeled fixture replay.

The video is a presentation of captured output, not a microphone or GUI recording.
No downloads are performed. Run with --help for the required verified local paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import re
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import textwrap
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = json.loads((ROOT / "src/orukeet/artifacts.json").read_text())["files"]["q8"]["sha256"]
AUDIO_SHA = "d7d4e74b8a333ed02186008bc109a1b1a19d16da668bd56e785d80d69a16a72f"
PCM_SHA = "a29462b8ebd467318000e683b9117ade46230d3255ed2024e7db894abd9b38c9"
SAMPLE_SOURCE = "https://github.com/NVIDIA/NeMo-Speech.cpp/blob/4f9676226f667d14608487df744f375db87127f8/test_files/asr/wav/test/jfk.wav"
SOURCE_NOTICE = "https://github.com/NVIDIA/NeMo-Speech.cpp/blob/4f9676226f667d14608487df744f375db87127f8/THIRD_PARTY_NOTICES.md#whispercpp-sample-audio"
ARCHIVE_SOURCE = "https://www.jfklibrary.org/asset-viewer/archives/jfkwha-001"
WIDTH, HEIGHT, FPS, DURATION, AUDIO_START = 1920, 1080, 30, 28, 3.0


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def stamp(seconds):
    ms = round(seconds * 1000)
    return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02}.{ms%1000:03}"


def read_audio(path):
    if sha(path) != AUDIO_SHA:
        raise ValueError("Use the exact documented JFK WAV fixture")
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
            raise ValueError("Expected mono 16 kHz PCM16")
        pcm = source.readframes(source.getnframes())
    if hashlib.sha256(pcm).hexdigest() != PCM_SHA:
        raise ValueError("Fixture PCM differs")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768


def measure(args, output):
    from orukeet import Orukeet

    samples = read_audio(args.audio)
    if sha(args.model) != MODEL_SHA:
        raise ValueError("This release demo requires the exact verified Q8 artifact")
    before = time.perf_counter()
    with Orukeet(args.model, args.runtime, device=args.device) as engine:
        construction_s = time.perf_counter() - before
        warm_start = time.perf_counter()
        warmup = engine.transcribe(args.audio)
        warmup_s = time.perf_counter() - warm_start
        started = datetime.now(timezone.utc).isoformat()
        begin = time.perf_counter()
        result = engine.transcribe(args.audio)
        elapsed = time.perf_counter() - begin
        ended = datetime.now(timezone.utc).isoformat()
    if not result["text"].strip() or result != warmup:
        raise RuntimeError("Warmup and measured output differ; inspect before making this demo")
    chip = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    import orukeet
    code_root = Path(orukeet.__file__).resolve().parent
    receipt = {
        "status": "passed",
        "kind": "real_native_file_transcription",
        "private_review": True,
        "started_utc": started, "ended_utc": ended,
        "model": {"name": "Orukeet r3", "sha256": MODEL_SHA, "bytes": args.model.stat().st_size, "format": "Q8_0 GGUF"},
        "audio": {"file": "demos/fixtures/jfk.wav", "container_sha256": AUDIO_SHA,
                  "pcm_sha256": PCM_SHA, "duration_s": len(samples)/16000,
                  "sample_rate_hz": 16000, "channels": 1, "encoding": "PCM16",
                  "native_fixture_source": SAMPLE_SOURCE, "upstream_notice": SOURCE_NOTICE,
                  "archive_source": ARCHIVE_SOURCE,
                  "attribution": "John F. Kennedy, Inaugural Address, January 20, 1961. White House Audio Collection, JFK Library; sample via whisper.cpp and NVIDIA NeMo-Speech.cpp.",
                  "rights": "JFK Library identifies the official inaugural recording as Public Domain. The redistributed fixture is retained with upstream MIT notice; see demos/fixtures/SOURCE.md."},
        "environment": {"platform": platform.platform(), "chip": chip, "device": args.device,
                        "python": platform.python_version(), "orukeet_version": importlib.metadata.version("orukeet"),
                        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "av")}},
        "runtime": {"name": "NVIDIA NeMo-Speech.cpp", "version": "v0.1.0",
                    "files_sha256": {str(p.relative_to(args.runtime)): sha(p) for p in sorted(args.runtime.rglob("*")) if p.is_file() and p.suffix in (".dylib", ".so", ".dll")}},
        "package_files_sha256": {p.name: sha(p) for p in sorted(code_root.glob("*.py"))},
        "timing": {"construction_including_hash_and_model_load_s": construction_s,
                   "warmup_calls": 1, "warmup_s": warmup_s,
                   "measured_calls": 1, "measured_call_s": elapsed,
                   "included": ["local file decoding", "resampling/windowing", "temporary PCM write", "worker IPC", "native inference", "result assembly"],
                   "excluded": ["model hash/loading", "fixture playback", "video rendering", "microphone capture", "GUI/hotkey/paste"],
                   "interpretation": "One measured warm file transcription after one warmup; wall-clock call duration on the recorded device."},
        "result": result,
    }
    write_json(output/"orukeet-proof-receipt.json", receipt)
    return receipt, samples


def fonts():
    # The repository's existing dev environment supplies these portable fonts.
    import matplotlib
    folder = Path(matplotlib.get_data_path()) / "fonts/ttf"
    return folder/"DejaVuSans.ttf", folder/"DejaVuSans-Bold.ttf", folder/"DejaVuSansMono.ttf"


def render(receipt, samples, args, output):
    from PIL import Image, ImageDraw, ImageFont

    regular, bold, mono = fonts()
    font = lambda size, heavy=False: ImageFont.truetype(str(bold if heavy else regular), size)
    fs = {size: font(size) for size in (24, 26, 28, 30, 34, 38, 40, 44, 48, 56, 68)}
    fb = {size: font(size, True) for size in (26, 30, 34, 42, 54, 68, 96, 150)}
    fm = ImageFont.truetype(str(mono), 26)
    bg, ink, muted, teal, line, card = "#F3F1EB", "#172A32", "#4D626A", "#006F67", "#CBD5D2", "#FFFFFF"
    measured_ms = receipt["timing"]["measured_call_s"] * 1000
    elapsed_text = f"{measured_ms:.0f} ms" if measured_ms < 1000 else f"{measured_ms/1000:.2f} s"
    chip = receipt["environment"]["chip"]
    device = receipt["environment"]["device"].upper() if args.device == "cpu" else receipt["environment"]["device"].title()

    def wrapped(draw, value, x, y, width, face, fill=ink, gap=14):
        words, lines, current = value.split(), [], ""
        for word in words:
            candidate = (current + " " + word).strip()
            if current and draw.textlength(candidate, font=face) > width:
                lines.append(current); current = word
            else: current = candidate
        if current: lines.append(current)
        for value in lines:
            draw.text((x, y), value, font=face, fill=fill)
            y += face.size + gap
        return y

    def base(kicker):
        im = Image.new("RGB", (WIDTH, HEIGHT), bg); d = ImageDraw.Draw(im)
        d.text((88, 60), "orukeet", font=fb[54], fill=ink)
        d.text((1832, 78), "r3 Q8  /  FIXTURE REPLAY", font=fb[26], anchor="ra", fill=muted)
        d.line((88, 156, 1832, 156), fill=line, width=2)
        d.text((88, 196), kicker, font=fb[26], fill=teal)
        d.line((88, 982, 1832, 982), fill=line, width=2)
        d.text((88, 1010), "Actual r3 output and timing, presented with the original audio.", font=fs[24], fill=muted)
        d.text((1832, 1010), "ORUK AI", font=fb[26], anchor="ra", fill=muted)
        return im

    intro = base("A LOCAL TRANSCRIPTION, WITH THE RECEIPT")
    d = ImageDraw.Draw(intro)
    d.text((88, 284), "Hear the clip.", font=fb[96], fill=ink)
    d.text((88, 410), "Read the result.", font=fb[96], fill=ink)
    d.text((92, 594), "11 seconds of public speech. One verified Q8 model.", font=fs[48], fill=muted)
    d.text((92, 696), f"Orukeet r3  ·  {chip} / {device}", font=fs[34], fill=ink)
    d.rounded_rectangle((88, 816, 1170, 912), radius=18, fill=teal)
    d.text((120, 842), "The next 11 seconds play the actual input audio.", font=fb[30], fill="white")

    listening = base("01 / THE INPUT PLAYS AT ITS ORIGINAL SPEED")
    d = ImageDraw.Draw(listening)
    d.text((88, 264), "JFK · January 20, 1961", font=fb[54], fill=ink)
    d.text((88, 344), "Public inaugural-address fixture · 16 kHz mono", font=fs[34], fill=muted)
    d.rounded_rectangle((88, 434, 880, 862), radius=22, fill=card)
    d.text((122, 470), "ACTUAL AUDIO WAVEFORM", font=fb[26], fill=muted)
    # Envelope is derived from the exact PCM played in the video.
    bins = 330
    envelope = np.array([np.max(np.abs(x)) for x in np.array_split(samples, bins)])
    envelope /= max(float(envelope.max()), 1e-6)
    for i, value in enumerate(envelope):
        x = 126 + i * 2.16
        length = max(3, float(value) * 115)
        d.line((x, 657-length, x, 657+length), fill=line, width=2)
    d.text((124, 800), "00:00", font=fm, fill=muted)
    d.text((844, 800), "00:11", font=fm, anchor="ra", fill=muted)
    d.text((966, 444), "SAVED MODEL OUTPUT", font=fb[26], fill=teal)
    bottom = wrapped(d, receipt["result"]["text"], 966, 510, 858, fs[48], gap=20)
    if bottom > 900: raise ValueError("Captured transcript does not fit; adapt the layout instead of clipping")
    d.text((88, 906), "Audio: White House Audio Collection / JFK Library. Sample via whisper.cpp / NVIDIA.", font=fs[26], fill=muted)

    result = base("02 / A FRESH, TIMED CALL THROUGH THE RELEASE PACKAGE")
    d = ImageDraw.Draw(result)
    d.text((88, 276), elapsed_text, font=fb[150], fill=teal)
    d.text((94, 466), "One warm file transcription", font=fb[54], fill=ink)
    d.text((94, 556), f"{chip}  /  {device}  /  11.0-second input", font=fs[40], fill=muted)
    wrapped(d, "The model stays loaded between requests. This timing records one file transcription after one warmup call.", 94, 660, 1660, fs[38], gap=16)
    d.rounded_rectangle((88, 830, 1832, 928), radius=18, fill=card)
    d.text((118, 858), "Includes decoding + native inference + worker IPC. Excludes loading and playback.", font=fs[30], fill=ink)

    ending = base("03 / THE RESULT IS INSPECTABLE")
    d = ImageDraw.Draw(ending)
    d.text((88, 268), "Try the model. Check the work.", font=fb[68], fill=ink)
    rows = [
        ("MODEL", "Orukeet r3 Q8 · 714 MB · 12,288 frozen Gabor kernels"),
        ("OUTPUT", "Exact transcript, timestamps and measured-call receipt included"),
        ("CODE", "Local inference package, model card and technical report"),
    ]
    for index, (label, value) in enumerate(rows):
        y = 426 + 130 * index
        d.text((94, y), label, font=fb[26], fill=teal)
        d.text((360, y-4), value, font=fs[34], fill=ink)
        d.line((94, y+74, 1826, y+74), fill=line, width=2)
    d.text((94, 858), "Source weights, Q8 and F16. Code MIT · weights CC BY-SA 4.0.", font=fs[30], fill=muted)

    # Native segment boundaries locate the captured transcript in the audio replay.
    captions = ["WEBVTT", "", "NOTE Captions reproduce captured Orukeet output. Audio begins at 00:03.000.", ""]
    for i, segment in enumerate(receipt["result"]["segments"], 1):
        start = max(0., min(11., segment["start"])) + AUDIO_START
        end = max(0., min(11., segment["end"])) + AUDIO_START
        if end <= start: continue
        captions.extend([str(i), f"{stamp(start)} --> {stamp(end)}", textwrap.fill(segment["text"], width=56), ""])
    if len(captions) == 4: raise ValueError("No usable captions")
    vtt = output/"orukeet-proof.vtt"; vtt.write_text("\n".join(captions)+"\n")
    srt = "\n".join(captions[4:]) + "\n"
    srt = re.sub(r"(\d{2}:\d{2}:\d{2})\.(\d{3})", r"\1,\2", srt)
    (output/"orukeet-proof.srt").write_text(srt, encoding="utf-8")
    (output/"orukeet-proof-transcript.txt").write_text(receipt["result"]["text"]+"\n")
    poster = base("A LOCAL TRANSCRIPTION, WITH THE RECEIPT")
    d = ImageDraw.Draw(poster)
    d.text((88, 286), "11 seconds of speech.", font=fb[96], fill=ink)
    d.text((88, 430), f"{elapsed_text} to transcribe.", font=fb[96], fill=teal)
    d.text((92, 636), f"{chip} / {device} · one measured warm call", font=fs[40], fill=ink)
    d.text((92, 724), "Public JFK fixture. Model loading and playback excluded.", font=fs[34], fill=muted)
    d.text((92, 842), "Actual output + reproducible timing receipt included", font=fb[34], fill=ink)
    poster.save(output/"orukeet-proof-thumbnail.png")

    # Keep diagnostic renders in this owned directory, out of the public media list.
    qa = Path(__file__).parent/".tmp"; qa.mkdir(exist_ok=True)
    logfile = qa/"ffmpeg.log"
    command = [args.ffmpeg, "-hide_banner", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
               "-r", str(FPS), "-i", "pipe:0", "-i", str(args.audio), "-i", str(vtt),
               "-map", "0:v", "-map", "1:a", "-map", "2:s", "-af", f"adelay={round(AUDIO_START*1000)},apad",
               "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac",
               "-b:a", "128k", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng", "-movflags", "+faststart",
               "-t", str(DURATION), str(output/"orukeet-proof.mp4")]
    with logfile.open("wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for i in range(FPS*DURATION):
                t = i/FPS
                if t < 3: im = intro
                elif t < 14:
                    im = listening.copy(); d = ImageDraw.Draw(im)
                    fraction = (t-AUDIO_START)/11
                    x = 126 + 711*fraction
                    d.line((x, 532, x, 780), fill=teal, width=4)
                    d.ellipse((x-7, 777, x+7, 791), fill=teal)
                elif t < 21: im = result
                else: im = ending
                if i in (0, FPS*8, FPS*17, FPS*25): im.save(qa/f"frame-{int(t):02}.png")
                process.stdin.write(im.tobytes())
            process.stdin.close()
            if process.wait() != 0: raise RuntimeError(f"FFmpeg failed; see {logfile}")
        finally:
            if process.poll() is None: process.kill(); process.wait()
    alt = (
        f"A 28-second Orukeet r3 Q8 proof video. It plays an 11-second public John F. Kennedy inaugural-address fixture "
        f"beside its real waveform and saved model transcript. A following card reports {elapsed_text} for one "
        f"warm local file transcription on {chip}/{device}, after one warmup. Model loading and playback are excluded. "
        "This is a labeled fixture replay and edited presentation, not a microphone or application recording. "
        "The final card links the result to the verified Q8 model and reproducible receipt. All frames identify r3 Q8 and fixture replay."
    )
    (output/"orukeet-proof-alt-text.md").write_text("# Video description\n\n"+alt+"\n\n# Thumbnail alt text\n\n"+
        f"Orukeet: 11 seconds of speech, {elapsed_text} to transcribe. {chip}/{device}; one measured warm call. "
        "Public JFK fixture; model loading and playback excluded. r3 Q8 release candidate.\n")
    write_json(output/"orukeet-proof-render.json", {
        "video": "orukeet-proof.mp4", "video_sha256": sha(output/"orukeet-proof.mp4"),
        "receipt_sha256": sha(output/"orukeet-proof-receipt.json"), "script_sha256": sha(__file__),
        "dimensions": [WIDTH, HEIGHT], "fps": FPS, "duration_s": DURATION,
        "audio_offset_s": AUDIO_START, "audio_duration_s": 11,
        "fonts_sha256": {p.name: sha(p) for p in (regular,bold,mono)},
        "ffmpeg_version": subprocess.check_output([args.ffmpeg,"-version"],text=True).splitlines()[0],
        "rendering": "Static explanatory cards and exact-PCM waveform. Timings and transcript from saved real native call. No simulated GUI/microphone or invented text."
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Exact verified Q8 GGUF; required unless --render-existing")
    parser.add_argument("--runtime", type=Path, help="Already installed pinned native SDK")
    parser.add_argument("--device", choices=["cpu", "metal", "cuda", "vulkan"], default="metal")
    parser.add_argument("--audio", type=Path, default=ROOT/"demos/fixtures/jfk.wav")
    parser.add_argument("--output", type=Path, default=ROOT/"launch/assets")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--render-existing", action="store_true", help="Render the saved receipt without rerunning inference")
    args = parser.parse_args()
    if not shutil.which(args.ffmpeg): parser.error("FFmpeg is required for local video rendering")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.render_existing:
        receipt = json.loads((args.output/"orukeet-proof-receipt.json").read_text())
        samples = read_audio(args.audio)
        if receipt["audio"]["container_sha256"] != AUDIO_SHA or receipt["model"]["sha256"] != MODEL_SHA:
            parser.error("Saved receipt is not the documented release fixture/model")
        args.device = receipt["environment"]["device"]
    else:
        if not args.model or not args.runtime: parser.error("Pass --model and --runtime, or --render-existing")
        receipt, samples = measure(args, args.output)
    print(json.dumps({"measured_call_s": receipt["timing"]["measured_call_s"], "text": receipt["result"]["text"]}), flush=True)
    render(receipt, samples, args, args.output)
    print(f"Rendered {args.output/'orukeet-proof.mp4'}", flush=True)


if __name__ == "__main__": main()
