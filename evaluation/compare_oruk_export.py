"""Compare native deployment with existing evaluations on identical recordings.

Preserves the sealed language/corpus membership and shared speaker bootstrap.
Runtime and precision intentionally differ; this is an export regression check.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def cluster(row):
    if row["src"] == "fleurs":
        if not row.get("fleurs_id"):
            raise ValueError("Missing FLEURS sentence identity")
        return "fleurs:parallel_sentence:" + str(row["fleurs_id"])
    speaker = row.get("speaker_id") or row.get("client_id")
    if not speaker or str(speaker).lower() in ("none", "unknown"):
        raise ValueError("Missing speaker identity")
    return row["src"] + ":speaker:" + str(speaker)


def compare(reference, candidate, registry, replicates=5000):
    metadata = [json.loads((p / "results.json").read_text()) for p in (reference, candidate)]
    sets, groups = {}, set()
    for meta in metadata:
        if meta["normalizer"] != "legacy-compatible-nfc-v1" or set(meta["sets"]) != set(registry["sets"]):
            raise ValueError("Incomplete or incompatible evaluation")
    for name, spec in registry["sets"].items():
        for meta in metadata:
            if meta["sets"][name]["manifest_sha256"] != spec["manifest_sha256"]:
                raise ValueError("Manifest changed: " + name)
        def load(root):
            rows = [json.loads(line) for line in (root / (name + "_hypotheses.jsonl")).read_text().splitlines()]
            result = {r["audio_filepath"]: r for r in rows}
            if len(result) != len(rows) or len(rows) != spec["rows"]:
                raise ValueError("Duplicate or missing recordings")
            return result
        a, b = load(reference), load(candidate)
        if a.keys() != b.keys():
            raise ValueError("Recordings differ")
        sets[name] = []
        for key, row in a.items():
            other = b[key]
            if row["text"] != other["text"] or row["words"] != other["words"] or cluster(row) != cluster(other):
                raise ValueError("References or groups differ")
            group = cluster(row)
            groups.add(group)
            sets[name].append((group, row["errors"], other["errors"], row["words"]))
    metrics = {"set:" + k: v for k, v in sets.items()}
    for lang, spec in registry["primary_languages"].items():
        metrics["language:" + lang] = [r for name in spec["sets"] for r in sets[name]]
    for corpus, names in registry["english_corpora"].items():
        metrics["english:" + corpus] = [r for name in names for r in sets[name]]
    order = {g: i for i, g in enumerate(sorted(groups))}
    names = list(metrics)
    matrix = np.zeros((len(order), len(names), 3))
    for j, name in enumerate(names):
        for group, before, after, words in metrics[name]:
            matrix[order[group], j] += [before, after, words]
    totals = matrix.sum(axis=0)
    rng = np.random.default_rng(20260905)
    differences = np.full((replicates, len(names)), np.nan)
    for start in range(0, replicates, 100):
        count = min(100, replicates-start)
        weights = rng.multinomial(len(order), np.full(len(order), 1/len(order)), size=count)
        draws = (weights @ matrix.reshape(len(order), -1)).reshape(count, len(names), 3)
        with np.errstate(invalid="ignore", divide="ignore"):
            differences[start:start+count] = 100 * (draws[:, :, 1] - draws[:, :, 0]) / draws[:, :, 2]
    def interval(values):
        values = values[np.isfinite(values)]
        return np.quantile(values, [.025, .975]).tolist() if len(values) >= replicates * .8 else None
    metrics_out = {}
    for j, name in enumerate(names):
        before, after = 100 * totals[j, :2] / totals[j, 2]
        metrics_out[name] = dict(reference_wer=before, candidate_wer=after, delta_pp=after-before,
                                 paired_delta_95ci=interval(differences[:, j]), rows=len(metrics[name]))
    def macro(members):
        before = float(np.mean([metrics_out[n]["reference_wer"] for n in members]))
        after = float(np.mean([metrics_out[n]["candidate_wer"] for n in members]))
        return dict(reference_wer=before, candidate_wer=after, delta_pp=after-before,
                    relative_improvement=1-after/before,
                    paired_delta_95ci=interval(differences[:, [names.index(n) for n in members]].mean(axis=1)))
    return dict(primary=macro(["language:" + k for k in registry["primary_languages"]]),
                english=macro(["english:" + k for k in registry["english_corpora"]]),
                posthoc_sensitivity={
                    'interpretation': 'Descriptive after evaluation; not a new selection gate.',
                    'leave_one_language_out': {
                        excluded: macro(['language:' + k for k in registry['primary_languages'] if k != excluded])
                        for excluded in registry['primary_languages']},
                    'language_point_estimates_improved': int(sum(metrics_out['language:' + k]['delta_pp'] < 0 for k in registry['primary_languages'])),
                    'median_language_delta_pp': float(np.median([metrics_out['language:' + k]['delta_pp'] for k in registry['primary_languages']]))},
                families={k: macro(["set:" + n for n in v]) for k, v in registry["families"].items()},
                metrics=metrics_out, reference_model_sha256=metadata[0]["model_sha256"],
                candidate_model_sha256=metadata[1]["model_sha256"],
                rows=registry["total_rows"], hours=registry["total_hours"],
                bootstrap=dict(replicates=replicates, seed=20260905, global_clusters=len(order),
                               unit="Source-qualified speakers; shared parallel sentence IDs for FLEURS"),
                interpretation="Deployment regression; runtime and precision differ. Evaluation recordings were used previously.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "registry", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    result = compare(args.reference, args.candidate, registry)
    result["registry_sha256"] = hashlib.sha256(args.registry.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("primary", "english", "families")}, indent=2))
