#!/usr/bin/env python3
"""Run pinned FluidAudio and NeMo on a sealed corpus, using repository WER scoring."""
import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

import torch

from convert import SOURCE_SHA256, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--models", action="append", required=True)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    fixtures = manifest["fixtures"]
    paths = [args.corpus / row["path"] for row in fixtures]
    for row, path in zip(fixtures, paths):
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Fixture changed: {path}")
    (args.output / "corpus.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    command = [str(args.benchmark), "--repeat", "1", "--warmup", "0", "--output", str(args.output / "coreml.json")]
    for entry in args.models:
        command += ["--models", entry]
    for path in paths:
        command += ["--audio", str(path)]
    subprocess.run(command, check=True)
    if sha256(args.source) != SOURCE_SHA256:
        raise ValueError("Unexpected source checkpoint")
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    torch.set_num_threads(4)
    model = EncDecRNNTBPEModel.restore_from(str(args.source), map_location="cpu").eval()
    model.preprocessor.featurizer.dither = 0
    source = []
    for row, path in zip(fixtures, paths):
        hypotheses = model.transcribe(audio=[str(path)], batch_size=1, return_hypotheses=True, verbose=False)
        model.eval()
        source.append({"sha256": row["sha256"], "text": hypotheses[0].text})
        print("NeMo", row["path"], hypotheses[0].text, flush=True)
    (args.output / "nemo.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    score_directory = Path(__file__).resolve().parents[2] / "evaluation/standard_asr"
    sys.path.insert(0, str(score_directory))
    from scoring import counts
    references = {row["sha256"]: row for row in fixtures}
    coreml = json.loads((args.output / "coreml.json").read_text())
    groups = collections.defaultdict(list)
    for row in coreml["measurements"]:
        groups[row["model"]].append((row["audioSHA256"], row["text"]))
    groups["orukeet-nemo"] = [(row["sha256"], row["text"]) for row in source]
    result = {"corpus_sha256": sha256(args.corpus / "manifest.json"), "source_sha256": SOURCE_SHA256,
              "sample_count": len(fixtures), "scope": "Small regression sample; not the full published evaluation",
              "scorer": "evaluation/standard_asr/scoring.py", "models": {}}
    for label, rows in groups.items():
        languages = collections.defaultdict(lambda: {"words": 0, "errors": 0, "utterances": 0})
        for digest, text in rows:
            ref = references[digest]
            score = counts(ref["reference"], text, ref["language"])
            totals = languages[ref["language"]]
            for key in ("words", "errors"):
                totals[key] += score[key]
            totals["utterances"] += 1
        words = sum(row["words"] for row in languages.values())
        errors = sum(row["errors"] for row in languages.values())
        result["models"][label] = {"words": words, "errors": errors, "wer": errors / words, "languages": dict(languages)}
    result["greedy_exact_text_match"] = (
        dict(groups["orukeet"]) == dict(groups["orukeet-greedy"])
        if "orukeet" in groups and "orukeet-greedy" in groups else None
    )
    (args.output / "accuracy.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
