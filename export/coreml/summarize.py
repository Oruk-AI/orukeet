#!/usr/bin/env python3
"""Summarize paired Swift measurements; optionally require exact text/token parity."""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summarize(report, baseline, require_parity=()):
    groups = defaultdict(list)
    paired = defaultdict(dict)
    for row in report["measurements"]:
        if row["repetition"] < 0:
            raise ValueError("Warmup must not be included in measurements")
        key = (row["model"], row["audioSHA256"])
        groups[key].append(row)
        pair_key = (row["audioSHA256"], row["repetition"])
        if row["model"] in paired[pair_key]:
            raise ValueError("Duplicate measurement")
        paired[pair_key][row["model"]] = row
    labels = {m["label"] for m in report["models"]}
    if baseline not in labels or set(require_parity) - labels:
        raise ValueError("Requested model label is missing")
    for pair in paired.values():
        if set(pair) != labels:
            raise ValueError("Unpaired measurements; every model must run each fixture/repetition")
    result = {"mode": report["mode"], "paced": report["paced"], "encoder_units": report["encoderUnits"],
              "baseline": baseline, "fixtures": [], "comparisons": {}, "failures": []}
    for (label, digest), rows in sorted(groups.items()):
        if len(rows) != report["repetitions"]:
            raise ValueError("Incomplete repetitions for a model/fixture")
        latency = [r["wallMs"] for r in rows]
        result["fixtures"].append({"model": label, "audio": Path(rows[0]["audio"]).name,
                                   "audio_sha256": digest, "runs": len(rows),
                                   "audio_seconds": rows[0]["audioSeconds"],
                                   "p50_ms": statistics.median(latency), "p95_ms": percentile(latency, .95),
                                   "rtfx": None if report["paced"] else rows[0]["audioSeconds"] * 1000 / statistics.median(latency),
                                   "text_variants": len({r["text"] for r in rows}),
                                   "transcript": rows[0]["text"],
                                   "p50_first_text_ms": statistics.median([r["firstTextMs"] for r in rows if r.get("firstTextMs") is not None]) if any(r.get("firstTextMs") is not None for r in rows) else None,
                                   "p50_finalization_ms": statistics.median([r["finalizationMs"] for r in rows if r.get("finalizationMs") is not None]) if any(r.get("finalizationMs") is not None for r in rows) else None})
    for label in sorted(labels - {baseline}):
        ratios, text_mismatches, token_mismatches = [], [], []
        for key, pair in sorted(paired.items()):
            base, candidate = pair[baseline], pair[label]
            ratios.append(candidate["wallMs"] / base["wallMs"])
            if candidate["text"] != base["text"]:
                text_mismatches.append(list(key))
            if candidate.get("tokenIDs") != base.get("tokenIDs"):
                token_mismatches.append(list(key))
        rng = random.Random(0)
        bootstrap = [statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000)]
        result["comparisons"][label] = {"paired_runs": len(ratios), "median_latency_ratio": statistics.median(ratios),
                                         "bootstrap_95_interval": [percentile(bootstrap, .025), percentile(bootstrap, .975)],
                                         "text_mismatches": text_mismatches, "token_mismatches": token_mismatches}
        if label in require_parity and (text_mismatches or token_mismatches):
            result["failures"].append(f"{label}: exact output parity failed")
    result["thermal_states"] = sorted({r["thermalState"] for r in report["measurements"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--require-parity", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.report.read_text()), args.baseline, args.require_parity)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["fixtures"]:
        print(f"{row['model']:20} {row['audio']:24} p50={row['p50_ms']:.2f} ms p95={row['p95_ms']:.2f} ms")
    for label, row in result["comparisons"].items():
        print(label, f"paired ratio={row['median_latency_ratio']:.4f}", row["bootstrap_95_interval"])
    if result["failures"]:
        raise SystemExit("; ".join(result["failures"]))


if __name__ == "__main__":
    main()
