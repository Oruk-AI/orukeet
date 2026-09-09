import glob, json, os, soundfile as sf
root = "/work/users/nathanroll/parakeet-ft/data/librispeech/LibriSpeech/test-other"
out = "/work/users/nathanroll/parakeet-ft/manifests/final/libri_en_test_other.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
n = 0; hours = 0.0
with open(out, "w") as f:
    for trans in sorted(glob.glob(f"{root}/*/*/*.trans.txt")):
        d = os.path.dirname(trans)
        for line in open(trans):
            uid, text = line.strip().split(" ", 1)
            p = f"{d}/{uid}.flac"; dur = sf.info(p).duration
            f.write(json.dumps({"audio_filepath": p, "duration": round(dur, 3), "text": text.lower(),
                                "lang": "en", "src": "librispeech"}) + "\n")
            n += 1; hours += dur
print("libri test-other", n, round(hours / 3600, 2), "h")
