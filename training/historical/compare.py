"""Side-by-side WER table: python compare.py eval/base/results.json eval/ft/results.json"""
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
rows = []
for k in sorted(a):
    if k in b:
        rows.append((k, a[k]["wer"], b[k]["wer"], a[k]["cer"], b[k]["cer"], a[k]["hours"]))
print(f"{'set':26} {'base':>7} {'ft':>7} {'rel%':>7} | {'cer_b':>6} {'cer_ft':>6} {'hours':>6}")
for k, wa, wb, ca, cb, h in rows:
    print(f"{k:26} {wa:7.2f} {wb:7.2f} {100*(wb-wa)/wa:+7.1f} | {ca:6.2f} {cb:6.2f} {h:6.2f}")
for src in ("fleurs", "cv"):
    sel = [(wa, wb) for k, wa, wb, *_ in rows if k.startswith(src)]
    if sel:
        ma, mb = sum(x for x, _ in sel) / len(sel), sum(y for _, y in sel) / len(sel)
        print(f"{src} macro-avg WER ({len(sel)} langs): {ma:.2f} -> {mb:.2f} ({100*(mb-ma)/ma:+.1f}%)")
