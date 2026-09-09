"""Language-confusion audit: Latvian-diacritic leakage and script mismatches, base vs finetuned.
usage: python leak.py eval/base_full eval/ft_ema20k"""
import glob, json, re, sys, yaml

BASE, FT = sys.argv[1], sys.argv[2]
LV = re.compile(r"[āēīūļņķģĀĒĪŪĻŅĶĢ]")
CYR = re.compile(r"[\u0400-\u04FF]"); GRK = re.compile(r"[\u0370-\u03FF]")


def script(t):
    return "cyr" if CYR.search(t) else "grk" if GRK.search(t) else "lat"


print(f"{'set':22} {'lv-leak base':>14} {'lv-leak ft':>14} {'script-mism base':>18} {'ft':>14}")
tot = [0, 0, 0, 0]
for p in sorted(glob.glob(f"{FT}/results_*_hyps.jsonl")):
    name = p.split("results_")[1].replace("_hyps.jsonl", "")
    if "_lv_" in name:
        continue
    f = [json.loads(l) for l in open(p)]
    b = [json.loads(l) for l in open(p.replace(FT, BASE))]
    lvf = sum(bool(LV.search(r["pred_text"])) for r in f); lvb = sum(bool(LV.search(r["pred_text"])) for r in b)
    smf = sum(script(r["pred_text"]) != script(r["text"]) for r in f if r["pred_text"].strip())
    smb = sum(script(r["pred_text"]) != script(r["text"]) for r in b if r["pred_text"].strip())
    for i, v in enumerate((lvb, lvf, smb, smf)):
        tot[i] += v
    if lvf + lvb + smf + smb > 0:
        print(f"{name:22} {lvb:6d} ({100*lvb/len(b):4.1f}%) {lvf:6d} ({100*lvf/len(f):4.1f}%) "
              f"{smb:8d} ({100*smb/len(b):4.1f}%) {smf:6d} ({100*smf/len(f):4.1f}%)")
print("TOTAL lv-leak base/ft:", tot[0], tot[1], "| script-mismatch base/ft:", tot[2], tot[3])

print()
cfg = yaml.safe_load(open("manifests/final/train_input_cfg.yaml"))
st = json.load(open("manifests/final/stats.json"))
for g in sorted(cfg, key=lambda g: -g["weight"]):
    l = g["tags"]["lang"]
    h = sum(st.get(f"{s}_{l}_train", {}).get("hours", 0) for s in ("cv", "fleurs"))
    inner = " ".join(f"{i['tags']['src']}={i['weight']}" for i in g["input_cfg"])
    print(f"{l:3} share {100*g['weight']:5.2f}%  hours {h:7.1f}  {inner}")
