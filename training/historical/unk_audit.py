#!/usr/bin/env python3
"""Per-language <unk> audit of manifests against the parakeet-tdt-0.6b-v3 SentencePiece model."""
import collections, glob, json, sys
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file=sys.argv[1])
unk = sp.unk_id()
stats = collections.defaultdict(lambda: [0, 0, 0, collections.Counter()])  # utts, pieces, unk pieces, unk chars
for path in sorted(glob.glob(sys.argv[2])):
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids = sp.encode(r["text"], out_type=int)
            s = stats[(r["lang"], r["src"])]
            s[0] += 1; s[1] += len(ids)
            n_unk = sum(1 for i in ids if i == unk)
            if n_unk:
                s[2] += n_unk
                for piece, i in zip(sp.encode(r["text"], out_type=str), ids):
                    if i == unk:
                        s[3][piece] += 1
print(f"{'lang':5} {'src':7} {'utts':>8} {'pieces':>10} {'unk':>7} {'unk%':>7}  top-unk")
for (lang, src), (u, p, k, ctr) in sorted(stats.items()):
    print(f"{lang:5} {src:7} {u:8d} {p:10d} {k:7d} {100*k/max(p,1):7.3f}  {dict(ctr.most_common(6))}")
