#!/usr/bin/env python3
"""Summarize all raw engine calls, requiring complete repeat coverage."""
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def main():
    reports = [json.loads(p.read_text()) for p in sorted((HERE / "results").glob("*.json"))]
    groups = {}
    indexed = {}
    for report in reports:
        key = (report["label"], report["round"])
        if key in indexed:
            raise ValueError(f"Duplicate result: {key}")
        indexed[key] = report
        seen = set()
        for row in report["records"]:
            row_key = (row["fixture"], row["repetition"])
            if row_key in seen:
                raise ValueError(f"Duplicate call: {key} {row_key}")
            seen.add(row_key)
            if not all(math.isfinite(row[k]) and row[k] > 0 for k in ["wall_ms", "processing_ms"]):
                raise ValueError("Invalid timing")
            groups.setdefault((report["label"], row["fixture"]), []).append((report["round"], row))
        for fixture in {row["fixture"] for row in report["records"]}:
            expected = set(range(-3, 15))
            actual = {row["repetition"] for row in report["records"] if row["fixture"] == fixture}
            if actual != expected:
                raise ValueError(f"Incomplete fixture: {fixture}")
    summaries = []
    for (label, fixture), rows in sorted(groups.items()):
        if {rnd for rnd, _ in rows} != {0, 1}:
            raise ValueError(f"Incomplete rounds: {label} {fixture}")
        timed = [(rnd, r) for rnd, r in rows if not r["warmup"]]
        medians = [statistics.median(r["wall_ms"] for rnd, r in timed if rnd == i) for i in [0, 1]]
        processing = [statistics.median(r["processing_ms"] for rnd, r in timed if rnd == i) for i in [0, 1]]
        unique_text = sorted({r["text"] for _, r in rows})
        summaries.append({
            "label": label, "fixture": fixture, "audio_seconds": rows[0][1]["audio_seconds"],
            "timed_calls": len(timed), "warmup_calls": len(rows) - len(timed),
            "wall_round_median_ms": medians, "wall_mean_of_round_medians_ms": statistics.mean(medians),
            "processing_round_median_ms": processing,
            "processing_mean_of_round_medians_ms": statistics.mean(processing),
            "pooled_wall_median_ms": statistics.median(r["wall_ms"] for _, r in timed),
            "pooled_wall_p95_ms": percentile([r["wall_ms"] for _, r in timed], .95),
            "unique_transcripts_including_warmups": len(unique_text), "transcripts": unique_text,
            "thermal_states": sorted({r[k] for _, r in rows for k in ["thermal_before", "thermal_after"]}),
        })
    comparisons = []
    def compare(first, second, kind):
        available = all((label, i) in indexed for label in [first, second] for i in [0, 1])
        if not available:
            return
        first_rows = { (rnd, r["fixture"], r["repetition"]): r
                      for rnd in [0, 1] for r in indexed[first, rnd]["records"] }
        second_rows = { (rnd, r["fixture"], r["repetition"]): r
                       for rnd in [0, 1] for r in indexed[second, rnd]["records"] }
        if first_rows.keys() != second_rows.keys():
            raise ValueError("Comparison membership mismatch")
        changes = [{"round": k[0], "fixture": k[1], "repetition": k[2],
                    "baseline": first_rows[k]["text"], "candidate": second_rows[k]["text"]}
                   for k in sorted(first_rows) if first_rows[k]["text"] != second_rows[k]["text"]]
        deltas = []
        for fixture in sorted({k[1] for k in first_rows}):
            a = next(x for x in summaries if x["label"] == first and x["fixture"] == fixture)
            b = next(x for x in summaries if x["label"] == second and x["fixture"] == fixture)
            baseline = a["wall_mean_of_round_medians_ms"]
            candidate = b["wall_mean_of_round_medians_ms"]
            deltas.append({"fixture": fixture, "baseline_ms": baseline, "candidate_ms": candidate,
                           "reduction_percent": 100 * (1 - candidate / baseline)})
        comparisons.append({"kind": kind, "baseline": first, "candidate": second,
                            "compared_calls_including_warmups": len(first_rows),
                            "transcript_mismatch_count": len(changes), "mismatches": changes,
                            "wall_comparison": deltas})
    for profile in ["published", "int8"]:
        compare(f"main-original-{profile}-c1-ane", f"main-optimized-{profile}-c1-ane", "runtime")
    for concurrency in [2, 4]:
        compare("concurrency-optimized-int8-c1-ane", f"concurrency-optimized-int8-c{concurrency}-ane", "chunk_concurrency")
    compare("gpu-optimized-int8-c1-ane", "gpu-optimized-int8-c1-gpu", "encoder_compute_policy")
    report = {
        "complete_process_runs": len(reports), "total_calls_including_warmups": sum(len(r["records"]) for r in reports),
        "summaries": summaries, "comparisons": comparisons,
        "process_memory": [{"label": r["label"], "round": r["round"],
                            "peak_rss_bytes": r["peak_rss_bytes_final"], "load_ms": r["load_ms"]}
                           for r in reports],
        "statistics": "Performance comparisons use the mean of two round medians; each round contains 15 timed repetitions after three warmups per fixture. Separate pooled median/p95 are descriptive. No confidence-interval or iPhone claim."
    }
    (HERE / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"runs": len(reports), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
