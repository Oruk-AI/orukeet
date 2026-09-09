#!/usr/bin/env python3
"""Download FLEURS (google/fleurs) audio tarballs + TSVs for the 25 Parakeet-v3 languages."""
import os, sys
from huggingface_hub import snapshot_download

# FLEURS code -> ISO 639-1 used by parakeet-tdt-0.6b-v3
LANGS = {
    "bg_bg": "bg", "hr_hr": "hr", "cs_cz": "cs", "da_dk": "da", "nl_nl": "nl",
    "en_us": "en", "et_ee": "et", "fi_fi": "fi", "fr_fr": "fr", "de_de": "de",
    "el_gr": "el", "hu_hu": "hu", "it_it": "it", "lv_lv": "lv", "lt_lt": "lt",
    "mt_mt": "mt", "pl_pl": "pl", "pt_br": "pt", "ro_ro": "ro", "ru_ru": "ru",
    "sk_sk": "sk", "sl_si": "sl", "es_419": "es", "sv_se": "sv", "uk_ua": "uk",
}

out = sys.argv[1] if len(sys.argv) > 1 else "fleurs_raw"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
patterns = [f"data/{k}/*" for k in LANGS]
path = snapshot_download(
    "google/fleurs", repo_type="dataset", local_dir=out,
    allow_patterns=patterns, max_workers=8,
)
print("FLEURS_DOWNLOAD_COMPLETE", path, flush=True)
