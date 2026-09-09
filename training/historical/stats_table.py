import json, sys
s = json.load(open(sys.argv[1]))
langs = sorted({k.split("_")[1] for k in s})
print(f"{'lang':4} {'cv_train_h':>10} {'fl_train_h':>10} {'total_h':>8} {'cv_test':>7} {'fl_test':>7} {'unk_drop':>8} {'dup_drop':>8}")
tot = 0
for l in langs:
    cv = s.get(f"cv_{l}_train", {}).get("hours", 0); fl = s.get(f"fleurs_{l}_train", {}).get("hours", 0)
    ud = sum(v["dropped_unk"] for k, v in s.items() if k.split("_")[1] == l and k.endswith("train"))
    dd = sum(v["dropped_dup_vs_test"] for k, v in s.items() if k.split("_")[1] == l and k.endswith("train"))
    tot += cv + fl
    print(f"{l:4} {cv:10.1f} {fl:10.1f} {cv+fl:8.1f} {s.get(f'cv_{l}_test', {}).get('utts', 0):7d} "
          f"{s.get(f'fleurs_{l}_test', {}).get('utts', 0):7d} {ud:8d} {dd:8d}")
print("TOTAL train hours", round(tot, 1))
