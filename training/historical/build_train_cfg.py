#!/usr/bin/env python3
"""Build the Lhotse input_cfg (two-tier temperature sampling) and validation subsets.

Language weight  ∝ hours^ALPHA  (ALPHA=0.5 = NVIDIA's Parakeet/Canary recipe), with English
capped at EN_MAX_SHARE of the mix (it is 23% of hours; replay only needs to prevent forgetting).
Within a language, CV vs FLEURS weight ∝ hours^0.5 (up-weights FLEURS' clean read speech).
"""
import json, os, random, sys
import yaml

ROOT = "/work/users/nathanroll/parakeet-ft"
FIN = f"{ROOT}/manifests/final"
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
EN_MAX_SHARE = 0.10
VAL_PER_LANG = {"fleurs": 150, "cv": 100}
random.seed(0)

stats = json.load(open(f"{FIN}/stats.json"))
langs = sorted({k.split("_")[1] for k in stats})

hours = {l: {s: stats.get(f"{s}_{l}_train", {}).get("hours", 0.0) for s in ("cv", "fleurs")} for l in langs}
lang_w = {l: (hours[l]["cv"] + hours[l]["fleurs"]) ** ALPHA for l in langs}
tot = sum(lang_w.values())
if lang_w["en"] / tot > EN_MAX_SHARE:
    others = tot - lang_w["en"]
    lang_w["en"] = EN_MAX_SHARE * others / (1 - EN_MAX_SHARE)
tot = sum(lang_w.values())

groups = []
print(f"{'lang':4} {'hours':>7} {'share%':>7} {'cv_w':>6} {'fl_w':>6}")
for l in langs:
    inner = []
    hcv, hfl = hours[l]["cv"], hours[l]["fleurs"]
    wcv, wfl = hcv ** 0.5, hfl ** 0.5
    for src, h, w in (("cv", hcv, wcv), ("fleurs", hfl, wfl)):
        if h > 0:
            inner.append({"type": "nemo", "manifest_filepath": f"{FIN}/{src}_{l}_train.json",
                          "weight": round(w / (wcv + wfl), 4), "tags": {"lang": l, "src": src}})
    groups.append({"type": "group", "weight": round(lang_w[l] / tot, 5), "tags": {"lang": l}, "input_cfg": inner})
    print(f"{l:4} {hcv+hfl:7.1f} {100*lang_w[l]/tot:7.2f} {wcv/(wcv+wfl):6.2f} {wfl/(wcv+wfl):6.2f}")

with open(f"{FIN}/train_input_cfg.yaml", "w") as f:
    yaml.safe_dump(groups, f, sort_keys=False)

# validation subsets (fixed random subsample per language, deterministic)
for src, n in VAL_PER_LANG.items():
    rows = []
    for l in langs:
        p = f"{FIN}/{src}_{l}_dev.json"
        if not os.path.exists(p):
            continue
        lines = open(p, encoding="utf-8").read().splitlines()
        random.shuffle(lines)
        rows += lines[:n]
    with open(f"{FIN}/val_{src}_dev_sub.json", "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"val_{src}_dev_sub: {len(rows)} utts")

# pilot config: Greek + English only
pilot = [g for g in groups if g["tags"]["lang"] in ("el", "en")]
pilot[0]["weight"], pilot[1]["weight"] = 0.5, 0.5
with open(f"{FIN}/pilot_input_cfg.yaml", "w") as f:
    yaml.safe_dump(pilot, f, sort_keys=False)
print("wrote", f"{FIN}/train_input_cfg.yaml", "and pilot_input_cfg.yaml")
