#!/usr/bin/env python3
"""Apply the saved development-only selection rule before opening confirmation."""
import argparse
import json
import math
from pathlib import Path


def select(reference, reference_accents, candidate, protocol):
    for other in (reference_accents, candidate):
        for key in ("normalizer", "decoding"):
            if reference[key] != other[key]:
                raise ValueError("Unmatched " + key)
    if reference["model_sha256"] != reference_accents["model_sha256"]:
        raise ValueError("Reference model differs across evaluations")
    if reference["sets"].keys() & reference_accents["sets"].keys():
        raise ValueError("Duplicate reference sets")
    if set(reference_accents["sets"]) != {"english_dialects_dev", "speechocean762_dev"}:
        raise ValueError("Incomplete accent development coverage")
    baseline = reference["sets"] | reference_accents["sets"]
    if baseline.keys() != candidate["sets"].keys():
        raise ValueError("Candidate sets differ from the complete development suite")
    rows = {}
    for name, a in baseline.items():
        b = candidate["sets"][name]
        if a["manifest_sha256"] != b["manifest_sha256"]:
            raise ValueError("Manifest changed: " + name)
        av, bv = a["slices"]["all"], b["slices"]["all"]
        if not all(isinstance(v["wer"], (int, float)) and math.isfinite(v["wer"])
                   for v in (av, bv)):
            raise ValueError("Invalid WER: " + name)
        if (av["words"], av["rows"]) != (bv["words"], bv["rows"]):
            raise ValueError("Reference counts changed: " + name)
        rows[name] = {"reference_wer": av["wer"], "candidate_wer": bv["wer"],
                      "delta_pp": bv["wer"] - av["wer"]}
    families = {}
    for prefix, count in (("cv_", 24), ("fleurs_", 25)):
        values = [v for k, v in rows.items() if k.startswith(prefix)]
        if len(values) != count:
            raise ValueError("Incomplete language coverage: " + prefix)
        families[prefix.rstrip("_")] = {
            key: sum(v[key] for v in values) / len(values)
            for key in ("reference_wer", "candidate_wer", "delta_pp")}
    primary = {key: sum(f[key] for f in families.values()) / 2
               for key in ("reference_wer", "candidate_wer", "delta_pp")}
    maximum = protocol["maximum_individual_language_source_regression_percentage_points"]
    failures = [name for name, value in rows.items() if value["delta_pp"] > maximum]
    minimum = protocol["minimum_absolute_improvement_percentage_points"]
    if primary["delta_pp"] > -minimum:
        failures.append("primary_improvement_below_" + str(minimum) + "_pp")
    accepted = not failures
    chosen = candidate if accepted else reference
    return {"candidate_accepted": accepted, "failed_criteria": failures,
            "selected_model_path": chosen["model_path"],
            "selected_model_sha256": chosen["model_sha256"],
            "reference_model_sha256": reference["model_sha256"],
            "candidate_model_sha256": candidate["model_sha256"],
            "primary": primary, "families": families, "sets": rows,
            "protocol": protocol,
            "confirmation_used_for_selection": False}


def main():
    p = argparse.ArgumentParser()
    for arg in ("reference", "reference_accents", "candidate", "protocol", "output"):
        p.add_argument("--" + arg.replace("_", "-"), type=Path, required=True)
    args = p.parse_args()
    read = lambda path: json.loads(path.read_text())
    result = select(read(args.reference), read(args.reference_accents),
                    read(args.candidate), read(args.protocol))
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("candidate_accepted", "failed_criteria", "primary")}, indent=2))


if __name__ == "__main__":
    main()
