"""Run the fixed three-clip private listening kit; keep all outputs unchanged.

Requires the Orukeet 0.1.0rc1 environment, released Q8 model and native runtime.
This is a qualitative inspection, not an accuracy or speed benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parent
LANGUAGES = {"fr_fr": ("French", "fr"), "es_419": ("Spanish · Latin America", "es"), "lv_lv": ("Latvian", "lv")}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def playback_copy(source, target):
    # Use the package's actual decoding path, including its 16 kHz PCM16
    # conversion. This provides browser-compatible playback; source WAV remains.
    import numpy as np
    from orukeet.audio import windows
    samples = np.concatenate([samples for _, samples in windows(str(source))])
    pcm = (samples * 32768).astype("<i2")
    with wave.open(str(target), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm.tobytes())
    return {"local_file": f"audio/{target.name}", "sha256": sha(target),
            "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
            "duration_s": len(samples)/16000, "sample_rate_hz": 16000,
            "channels": 1, "encoding": "PCM16 WAV",
            "changes": "Decoded with Orukeet's audio path to mono 16 kHz PCM16 for browser playback. No trimming, denoising, speed change or generated speech."}


def render(receipt, destination):
    esc = html.escape
    cards = []
    for item in receipt["clips"]:
        label, language = LANGUAGES[item["config"]]
        source = item["source"]
        result = item["result"]
        actual = result["text"] if result is not None else f"INFERENCE ERROR: {item['error']}"
        cards.append(f'''<article aria-labelledby="{item['config']}">
<h2 id="{item['config']}">{esc(label)}</h2>
<p class="meta">FLEURS test · original TSV row 0 · ID {source['id']} · {item['playback']['duration_s']:.2f} seconds</p>
<audio controls preload="metadata" aria-label="Listen to the {esc(label)} FLEURS recording" src="{esc(item['playback']['local_file'])}">Audio playback is unavailable. <a href="{esc(item['playback']['local_file'])}">Download WAV</a>.</audio>
<div class="comparison"><section><h3>Provider reference</h3><p lang="{language}" class="transcript">{esc(source['reference_raw'])}</p></section>
<section><h3>Actual Orukeet output</h3><p class="transcript">{esc(actual)}</p></section></div>
<details><summary>Source and reproduction details</summary>
<p>Source file: <code>{esc(source['source_filename'])}</code>. Reference and prediction are shown without editorial corrections. The reference is supplied by the dataset, not newly checked by a native speaker.</p>
<p>Provider-normalized reference: <span lang="{language}">{esc(source['reference_normalized_by_provider'])}</span></p>
<p>Model call {item['call_index_one_based']} in this process: {item['transcribe_call_s']:.6f} seconds after initialization. No warm-up or repeated best-of selection. This number is a run receipt, not a benchmark.</p>
<p>Original WAV SHA-256: <code>{esc(source['audio']['sha256'])}</code></p>
<p><a href="{esc(source['audio']['local_file'])}">Original source WAV</a> · <a href="provenance/{item['config']}.json">Clip provenance</a> · <a href="{esc(source['tsv']['url'])}">Pinned provider TSV</a></p>
</details></article>''')
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Orukeet — private multilingual listening kit</title>
<style>
:root {{ color-scheme: light; font-family: system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#17232b; background:#f4f5f2; }}
body {{ max-width:1040px; margin:0 auto; padding:36px 24px 60px; line-height:1.55; }}
h1 {{ font-size:clamp(1.8rem,4vw,2.6rem); line-height:1.15; margin:12px 0; }}
h2 {{ margin:0 0 8px; font-size:1.45rem; }} h3 {{ font-size:1rem; margin:0 0 8px; }}
.status {{ color:#60430c; font-weight:750; letter-spacing:.04em; font-size:.8rem; }}
.scope {{ max-width:88ch; }} .meta {{ color:#465663; font-size:.9rem; }}
article {{ margin-top:26px; padding:26px; background:white; border:1px solid #cbd2d6; border-radius:10px; }}
audio {{ width:100%; margin:12px 0 22px; }}
.comparison {{ display:grid; grid-template-columns:1fr 1fr; gap:26px; }}
.transcript {{ margin:0; font-size:1.13rem; line-height:1.65; white-space:pre-wrap; overflow-wrap:anywhere; }}
details {{ margin-top:25px; border-top:1px solid #dce0e2; padding-top:14px; font-size:.9rem; }}
summary {{ cursor:pointer; font-weight:650; }} code {{ overflow-wrap:anywhere; }}
a {{ color:#005f73; text-underline-offset:3px; }} a:focus-visible,summary:focus-visible,audio:focus-visible {{ outline:3px solid #ae5a00; outline-offset:4px; }}
footer {{ margin-top:32px; font-size:.9rem; }}
@media(max-width:650px) {{ body {{ padding:22px 14px 40px; }} article {{ padding:20px; }} .comparison {{ grid-template-columns:1fr; gap:22px; }} }}
</style></head><body>
<div class="status">PRIVATE REVIEW · THREE FIXED EXAMPLES</div>
<h1>Listen to Orukeet in three languages.</h1>
<p class="scope">French, Spanish and Latvian: the first physical test-TSV row in each selected FLEURS configuration, fixed before inference. Play the recording and compare the provider's reference with the actual model output. Orukeet r3 supplies every transcript shown here.</p>
<p class="scope">These three read-speech examples use FLEURS, a public evaluation source used in this project. Clip membership is fixed and every output is shown as returned. Full quantitative comparisons and their protocols accompany the model release.</p>
<p class="meta">Orukeet r3 · Q8 · {esc(receipt['environment']['chip'])} · {esc(receipt['environment']['device'])} · run {esc(receipt['started_utc'])}</p>
{''.join(cards)}
<footer><p><strong>Audio and references:</strong> FLEURS (2022), Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera and Ankur Bapna; distributed by Google. <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Original files are preserved. Playback copies use mono 16 kHz PCM16; no content was trimmed or synthesized. No endorsement is implied.</p>
<p>Source revision: <code>{esc(receipt['source_revision'])}</code>. <a href="README.md">Scope and reproduction</a> · <a href="receipt.json">Complete run receipt</a> · <a href="provenance/selection.json">Selection made before inference</a> · <a href="provenance/source-card.json">Verified provider card record</a></p>
<p>This static page uses local audio and inline styles. It has no external assets, uploads, analytics or network requests unless a visitor follows an external source/license link.</p></footer>
</body></html>'''
    destination.write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "metal", "cuda", "vulkan"), default="metal")
    parser.add_argument("--output", type=Path, default=ROOT / "receipt.json")
    args = parser.parse_args()
    if importlib.metadata.version("orukeet") != "0.1.0rc1":
        raise ValueError("This receipt requires Orukeet 0.1.0rc1")
    import orukeet
    from orukeet import Orukeet
    catalog = json.loads((ROOT.parents[1] / "src/orukeet/artifacts.json").read_text())
    assert sha(args.model) == catalog["files"]["q8"]["sha256"], "Use the current r3 Q8 model"
    selection = json.loads((ROOT / "provenance/selection.json").read_text())
    sources = [json.loads((ROOT / "provenance" / f"{config}.json").read_text()) for config in selection["configs"]]
    for source in sources:
        if sha(ROOT / source["audio"]["local_file"]) != source["audio"]["sha256"]:
            raise ValueError("Original source WAV differs from provenance")
        if sha(ROOT / source["tsv"]["selected_row_file"]) != source["tsv"]["selected_row_sha256"]:
            raise ValueError("Selected TSV row differs from provenance")
    code_root = Path(orukeet.__file__).resolve().parent
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip() if sys.platform == "darwin" else platform.processor()
    receipt = {
        "private_review": True, "kind": "qualitative_reused_source_multilingual_listening",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": selection["revision"], "selection_sha256": sha(ROOT / "provenance/selection.json"),
        "command": ".venv/bin/python demos/multilingual/reproduce.py --model /path/to/orukeet-v0.1.0rc1-q8.gguf --runtime /path/to/native-runtime --device " + args.device,
        "command_note": "Portable invocation; exact local paths are kept only in the private .tmp run cache. Artifact, code and runtime hashes identify this run.",
        "environment": {"platform": platform.platform(), "architecture": platform.machine(), "chip": chip,
                        "device": args.device, "python": platform.python_version(), "orukeet_version": importlib.metadata.version("orukeet"),
                        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "av")}},
        "model": {"name": "Orukeet r3", "sha256": sha(args.model), "bytes": args.model.stat().st_size, "format": "Q8_0 GGUF", "release_artifact": "orukeet-v0.1.0rc1-q8.gguf"},
        "runtime": {"name": "NVIDIA NeMo-Speech.cpp",
                    "files_sha256": {str(path.relative_to(args.runtime)): sha(path) for path in sorted(args.runtime.rglob("*")) if path.is_file() and path.suffix in (".dylib", ".so", ".dll", ".metallib")}},
        "package_files_sha256": {path.name: sha(path) for path in sorted(code_root.iterdir()) if path.is_file() and path.suffix in (".py", ".json")},
        "script_sha256": sha(Path(__file__)),
        "timing_scope": "One sequential call per fixed recording after one model initialization, no warmup and no selection across repeated calls. Includes file decoding, PCM preparation, IPC and inference. Excludes initialization, browser playback and app interaction. Not a benchmark.",
        "limitations": ["Only three fixed examples from a reused public evaluation source", "Prior training or evaluation exposure of these particular recordings is not established here", "Not fresh held-out confirmation", "No baseline comparison or aggregate accuracy claim", "No native-speaker assessment", "Read-speech examples do not cover general accents or real-world conditions", "No microphone or OpenWhispr integration demonstrated"],
        "clips": [],
    }
    cache = ROOT.parents[1] / ".tmp" / "multilingual-sources"
    cache.mkdir(parents=True, exist_ok=True)
    write_json(cache / "last-invocation.json", {"command": shlex.join([sys.executable, *sys.argv]), "started_utc": receipt["started_utc"]})
    begin = time.perf_counter()
    with Orukeet(args.model, args.runtime, device=args.device) as model:
        receipt["model_initialization_s"] = time.perf_counter() - begin
        for index, source in enumerate(sources, 1):
            config = source["config"]
            source_path = ROOT / source["audio"]["local_file"]
            playback = playback_copy(source_path, ROOT / "audio" / f"{config}.playback.wav")
            if round(playback["duration_s"] * 16000) != source["source_num_samples"]:
                raise ValueError("Decoded audio length differs from provider metadata")
            begin = time.perf_counter()
            try:
                result = model.transcribe(source_path)
                error = None
            except Exception as exc:
                result = None
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - begin
            receipt["clips"].append({"config": config, "source": source, "playback": playback,
                                      "call_index_one_based": index, "transcribe_call_s": elapsed,
                                      "result": result, "error": error})
            write_json(args.output, receipt)
            print(json.dumps({"config": config, "result": result, "error": error, "call_s": elapsed}, ensure_ascii=False), flush=True)
    receipt["ended_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["status"] = "completed_with_all_outputs_retained"
    write_json(args.output, receipt)
    render(receipt, args.output.with_suffix(".html") if args.output.name != "receipt.json" else ROOT / "index.html")


if __name__ == "__main__":
    main()
