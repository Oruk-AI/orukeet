"""Verify the prepared media against its captured native result and audio fixture."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from make_proof import AUDIO_SHA, AUDIO_START, DURATION, HEIGHT, MODEL_SHA, ROOT, WIDTH, read_audio, sha


def verify(output, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    video = output/"orukeet-proof.mp4"
    receipt = json.loads((output/"orukeet-proof-receipt.json").read_text())
    render = json.loads((output/"orukeet-proof-render.json").read_text())
    assert render["video_sha256"] == sha(video)
    assert render["receipt_sha256"] == sha(output/"orukeet-proof-receipt.json")
    assert render["script_sha256"] == sha(ROOT/"demos/make_proof.py")
    assert receipt["model"]["sha256"] == MODEL_SHA
    assert receipt["audio"]["container_sha256"] == AUDIO_SHA
    assert receipt["timing"]["measured_calls"] == receipt["timing"]["warmup_calls"] == 1
    assert receipt["timing"]["measured_call_s"] > 0
    assert (output/"orukeet-proof-transcript.txt").read_text().strip() == receipt["result"]["text"]
    metadata = json.loads(subprocess.check_output([ffprobe,"-v","error","-show_streams","-show_format","-of","json",str(video)],text=True))
    v = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    a = next(s for s in metadata["streams"] if s["codec_type"] == "audio")
    sub = next(s for s in metadata["streams"] if s["codec_type"] == "subtitle")
    assert (v["width"],v["height"],v["codec_name"],v["pix_fmt"]) == (WIDTH,HEIGHT,"h264","yuv420p")
    assert abs(float(metadata["format"]["duration"])-DURATION) < .05
    assert video.stat().st_size < 100_000_000 and DURATION <= 60
    assert a["codec_name"] == "aac" and sub["codec_name"] == "mov_text"
    # Decode the MP4 track, not the original WAV. AAC is lossy, so check alignment,
    # correlation and silence instead of expecting byte equality with the PCM.
    decoded = np.frombuffer(subprocess.check_output([ffmpeg,"-v","error","-i",str(video),"-map","0:a:0","-ac","1","-ar","16000","-f","f32le","pipe:1"]),dtype="<f4")
    source = read_audio(ROOT/"demos/fixtures/jfk.wav")
    start = round(AUDIO_START*16000)
    region = decoded[start:start+len(source)]
    correlation = float(np.corrcoef(source,region)[0,1])
    assert correlation > .99, correlation
    prefix_rms = float(np.sqrt(np.mean(decoded[:start-100]**2)))
    suffix_rms = float(np.sqrt(np.mean(decoded[start+len(source)+100:]**2)))
    assert prefix_rms < .001 and suffix_rms < .001
    assert np.max(np.abs(decoded)) <= 1.01
    vtt = (output/"orukeet-proof.vtt").read_text()
    caption_lines = [s for s in vtt.splitlines() if s and s!="WEBVTT" and not s.startswith("NOTE") and "-->" not in s and not s.isdigit()]
    assert " ".join(caption_lines) == " ".join(x["text"] for x in receipt["result"]["segments"])
    srt = (output/"orukeet-proof.srt").read_text(encoding="utf-8")
    srt_lines = [x for x in srt.splitlines() if x and "-->" not in x and not x.isdigit()]
    assert srt_lines == caption_lines
    assert srt.count(" --> ") == vtt.count(" --> ")
    for segment in receipt["result"]["segments"]:
        assert 0 <= segment["start"] < segment["end"] <= 11
    output_receipt = {
        "status": "passed", "video_sha256": sha(video), "receipt_sha256": sha(output/"orukeet-proof-receipt.json"),
        "dimensions": [v["width"],v["height"]], "duration_s": float(metadata["format"]["duration"]),
        "bytes": video.stat().st_size, "codecs": [v["codec_name"],a["codec_name"],sub["codec_name"]],
        "decoded_audio_correlation_at_expected_3s_offset": correlation,
        "before_audio_rms": prefix_rms, "after_audio_rms": suffix_rms,
        "transcript_matches_receipt_and_captions": True,
        "scope": "Metadata, exact identity linkage, decoded AAC alignment/content, caption text and local asset size. Visual layout reviewed separately in extracted video frames."
    }
    (output/"orukeet-proof-verification.json").write_text(json.dumps(output_receipt,indent=2)+"\n")
    print(json.dumps(output_receipt,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"launch/assets")
    parser.add_argument("--ffmpeg",default="ffmpeg")
    parser.add_argument("--ffprobe",default="ffprobe")
    args=parser.parse_args()
    verify(args.output,args.ffmpeg,args.ffprobe)
