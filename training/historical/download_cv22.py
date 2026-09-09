#!/usr/bin/env python3
"""Download Common Voice 22.0 (fsicoli mirror, CC0) train/dev/test shards + transcripts
for the 25 Parakeet-v3 languages, capping the high-resource languages."""
import os, sys
from huggingface_hub import snapshot_download

# CV locale -> ISO used by parakeet v3
LOCALES = {
    "bg": "bg", "cs": "cs", "da": "da", "nl": "nl", "en": "en", "et": "et", "fi": "fi",
    "fr": "fr", "de": "de", "el": "el", "hu": "hu", "it": "it", "lv": "lv", "lt": "lt",
    "mt": "mt", "pl": "pl", "pt": "pt", "ro": "ro", "ru": "ru", "sk": "sk", "sl": "sl",
    "es": "es", "sv-SE": "sv", "uk": "uk",
    # 'hr' (Croatian) is absent from CV 22.0
}
# train shard caps (each en shard ~57 h); None = all shards
TRAIN_CAP = {"en": 10, "fr": 8, "de": 8, "es": 6}

out = sys.argv[1] if len(sys.argv) > 1 else "cv_raw"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
patterns = []
for loc in LOCALES:
    patterns.append(f"transcript/{loc}/*")
    patterns.append(f"audio/{loc}/dev/*")
    patterns.append(f"audio/{loc}/test/*")
    cap = TRAIN_CAP.get(loc)
    if cap is None:
        patterns.append(f"audio/{loc}/train/*")
    else:
        patterns += [f"audio/{loc}/train/{loc}_train_{i}.tar" for i in range(cap)]
snapshot_download(
    "fsicoli/common_voice_22_0", repo_type="dataset", local_dir=out,
    allow_patterns=patterns, max_workers=6,
)
print("CV22_DOWNLOAD_COMPLETE", flush=True)
