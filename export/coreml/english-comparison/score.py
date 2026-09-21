#!/usr/bin/env python3
"""Score the complete paired English16 diagnostic using only Python's stdlib.

This tiny, reused FLEURS set is a diagnostic, not a release benchmark. It cannot
establish an English model preference or iPhone accuracy/performance.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata


MODELS = ("parakeet_v2", "orukeet")
FIXTURE_COUNT = 16
COUNT_KEYS = ("substitutions", "deletions", "insertions", "referenceWords")


def normalize(text):
    """NFKC, lowercase, unify curly apostrophes, separate punctuation, split.

    Straight apostrophes and hyphens are preserved to match the sealed reference
    policy, including after curly-quote conversion. Numbers are not expanded and
    spellings are not rewritten. Sealed references do not pass through this code.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = "".join(
        " " if char not in "'-" and unicodedata.category(char).startswith("P") else char
        for char in text
    )
    return " ".join(text.split())


def edit_counts(reference_words, hypothesis_words):
    """Minimum word edit counts; ties prefer substitution, deletion, insertion.

    Matching diagonals have priority over the three edit operations. Counts
    describe one deterministic optimal alignment; their sum is edit distance.
    """
    # Each cell is (distance, substitutions, deletions, insertions).
    previous = [(index, 0, 0, index) for index in range(len(hypothesis_words) + 1)]
    for ref_index, ref_word in enumerate(reference_words, start=1):
        current = [(ref_index, 0, ref_index, 0)]
        for hyp_index, hyp_word in enumerate(hypothesis_words, start=1):
            diagonal = previous[hyp_index - 1]
            if ref_word == hyp_word:
                diagonal_candidate = diagonal
            else:
                distance, substitutions, deletions, insertions = diagonal
                diagonal_candidate = (distance + 1, substitutions + 1, deletions, insertions)
            distance, substitutions, deletions, insertions = previous[hyp_index]
            deletion_candidate = (distance + 1, substitutions, deletions + 1, insertions)
            distance, substitutions, deletions, insertions = current[hyp_index - 1]
            insertion_candidate = (distance + 1, substitutions, deletions, insertions + 1)
            current.append(min(
                (diagonal_candidate, deletion_candidate, insertion_candidate),
                key=lambda candidate: candidate[0],
            ))
        previous = current
    errors, substitutions, deletions, insertions = previous[-1]
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "referenceWords": len(reference_words),
    }


def _require_string(record, key, context):
    value = record.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{context}: {key} must be a string")
    return value


def score(raw, fixtures):
    """Validate an exact 16 x 2 pairing, then score every recording or fail.

    Hashes here attest pairing with the sealed fixture manifest; the inference
    runner is responsible for hashing the actual audio bytes it consumes.
    """
    if not isinstance(raw, dict) or not isinstance(fixtures, dict):
        raise ValueError("raw and fixtures inputs must be JSON objects")
    fixture_list = fixtures.get("fixtures")
    if not isinstance(fixture_list, list) or len(fixture_list) != FIXTURE_COUNT:
        raise ValueError(f"fixtures must contain exactly {FIXTURE_COUNT} recordings")

    fixture_by_audio = {}
    for index, fixture in enumerate(fixture_list):
        context = f"fixture {index}"
        if not isinstance(fixture, dict):
            raise ValueError(f"{context} must be an object")
        path = _require_string(fixture, "path", context)
        audio = Path(path).name
        if not audio or "\\" in path or audio in fixture_by_audio:
            raise ValueError(f"{context}: invalid or duplicate recording basename {audio!r}")
        _require_string(fixture, "reference", context)
        normalized_reference = _require_string(fixture, "normalized_reference", context)
        if not normalized_reference.split():
            raise ValueError(f"{context}: sealed normalized reference is empty")
        digest = _require_string(fixture, "sha256", context)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{context}: sha256 must be a lowercase SHA-256 digest")
        fixture_by_audio[audio] = fixture

    measurements = raw.get("measurements")
    if not isinstance(measurements, list):
        raise ValueError("raw measurements must be an array")
    paired = {}
    for index, measurement in enumerate(measurements):
        context = f"measurement {index}"
        if not isinstance(measurement, dict):
            raise ValueError(f"{context} must be an object")
        model = _require_string(measurement, "model", context)
        audio = _require_string(measurement, "audio", context)
        if model not in MODELS or audio not in fixture_by_audio:
            raise ValueError(f"{context}: unexpected model/recording pair {model!r}, {audio!r}")
        key = (model, audio)
        if key in paired:
            raise ValueError(f"{context}: duplicate pair {model!r}, {audio!r}")
        if measurement.get("audioSHA256") != fixture_by_audio[audio]["sha256"]:
            raise ValueError(f"{context}: audio SHA-256 mismatch for {audio}")
        if measurement.get("error") is not None:
            raise ValueError(f"{context}: inference error for {model}/{audio}: {measurement['error']}")
        _require_string(measurement, "text", context)
        paired[key] = measurement
    expected = {(model, audio) for model in MODELS for audio in fixture_by_audio}
    missing = expected - paired.keys()
    if missing:
        raise ValueError(f"missing {len(missing)} model/recording pairs: {sorted(missing)!r}")

    totals = {model: {key: 0 for key in COUNT_KEYS + ("errors",)} for model in MODELS}
    clips = []
    for audio, fixture in fixture_by_audio.items():
        normalized_reference = fixture["normalized_reference"]
        reference_words = normalized_reference.split()
        model_scores = {}
        for model in MODELS:
            text = paired[(model, audio)]["text"]
            normalized_text = normalize(text)
            counts = edit_counts(reference_words, normalized_text.split())
            model_scores[model] = {
                "text": text,
                "normalizedText": normalized_text,
                **counts,
                "WER": counts["errors"] / counts["referenceWords"],
            }
            for key in totals[model]:
                totals[model][key] += counts[key]
        clips.append({
            "audio": audio,
            "audioSHA256": fixture["sha256"],
            "reference": fixture["reference"],
            "sealedNormalizedReference": fixture["normalized_reference"],
            "normalizedReference": normalized_reference,
            "referenceWords": len(reference_words),
            "models": model_scores,
            "WERDeltaOrukeetMinusParakeetV2": (
                model_scores["orukeet"]["WER"] - model_scores["parakeet_v2"]["WER"]
            ),
            "errorDeltaOrukeetMinusParakeetV2": (
                model_scores["orukeet"]["errors"] - model_scores["parakeet_v2"]["errors"]
            ),
        })
    for model in MODELS:
        totals[model]["clips"] = FIXTURE_COUNT
        totals[model]["microWER"] = totals[model]["errors"] / totals[model]["referenceWords"]

    return {
        "schemaVersion": 1,
        "scope": {
            "label": "Tiny reused English16 diagnostic; not a release benchmark",
            "limitations": [
                "These reused clips cannot establish general English accuracy or a model preference.",
                "Mac results do not establish physical iPhone accuracy, memory use, or latency.",
                "Sealed reference typography is preserved; spelling, number formatting, and hyphenation can affect WER.",
            ],
            "modelPreferenceChanged": False,
        },
        "normalization": {
            "hypothesisAlgorithm": "Unicode NFKC; lowercase; U+2018/U+2019 to straight apostrophe; "
                                   "Unicode punctuation to spaces except straight apostrophe and hyphen; collapse whitespace",
            "unicodeVersion": unicodedata.unidata_version,
            "referenceSource": "sealed normalized_reference.split(), unchanged for both models",
            "alignmentTieBreak": "diagonal (match or substitution), deletion, insertion",
            "WERUnits": "fraction; 1.0 means 100%; insertions may produce values above 1.0",
            "aggregate": "microWER = sum(substitutions + deletions + insertions) / sum(referenceWords)",
            "deltaDirection": "orukeet minus parakeet_v2; negative means fewer errors on this diagnostic",
        },
        "raw": raw,
        "fixtures": fixtures,
        "clips": clips,
        "aggregates": totals,
        "microWERDeltaOrukeetMinusParakeetV2": (
            totals["orukeet"]["microWER"] - totals["parakeet_v2"]["microWER"]
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() in (args.raw.resolve(), args.fixtures.resolve()):
            raise ValueError("output must not overwrite either input")
        raw_bytes = args.raw.read_bytes()
        fixture_bytes = args.fixtures.read_bytes()
        result = score(json.loads(raw_bytes), json.loads(fixture_bytes))
        result["inputSHA256"] = {
            "raw": hashlib.sha256(raw_bytes).hexdigest(),
            "fixtures": hashlib.sha256(fixture_bytes).hexdigest(),
        }
        # Publish a full validated result atomically. On failure, no output is written.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                         prefix=f".{args.output.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)
    except (OSError, ValueError, TypeError) as error:
        print(f"scoring failed: {error}", file=sys.stderr)
        return 1
    print(f"Scored {FIXTURE_COUNT} paired clips; diagnostic only: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
