"""Focused integrity and hand-computed WER tests for the English16 diagnostic."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("english_comparison_score", Path(__file__).with_name("score.py"))
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


def inputs():
    fixtures = {"dataset": "synthetic test", "fixtures": []}
    raw = {"runtime": {"revision": "sealed"}, "measurements": []}
    for index in range(16):
        audio = f"clip-{index}.wav"
        digest = hashlib.sha256(audio.encode()).hexdigest()
        reference = "A small cat." if index else "A cat."
        fixtures["fixtures"].append({
            "path": audio, "reference": reference,
            "normalized_reference": scorer.normalize(reference), "sha256": digest,
        })
        for model in scorer.MODELS:
            raw["measurements"].append({
                "model": model, "audio": audio, "audioSHA256": digest,
                "text": reference,
            })
    return raw, fixtures


class NormalizationTests(unittest.TestCase):
    def test_punctuation_apostrophe_whitespace_and_nfkc(self):
        self.assertEqual(
            scorer.normalize("  ＴＨＥ\t‘Cat’s’—film-editing, costs １２!\nDon't...  "),
            "the 'cat's' film-editing costs 12 don't",
        )

    def test_numbers_and_symbols_are_not_rewritten(self):
        self.assertEqual(scorer.normalize("1940 + 2 = 1942"), "1940 + 2 = 1942")


class AlignmentTests(unittest.TestCase):
    def test_hand_computed_substitution_deletion_and_insertion(self):
        counts = scorer.edit_counts("a b c d e f".split(), "a x c e f g".split())
        self.assertEqual(counts, {
            "substitutions": 1, "deletions": 1, "insertions": 1,
            "errors": 3, "referenceWords": 6,
        })

    def test_empty_hypothesis_counts_all_deletions(self):
        self.assertEqual(scorer.edit_counts(["a", "b"], []), {
            "substitutions": 0, "deletions": 2, "insertions": 0,
            "errors": 2, "referenceWords": 2,
        })

    def test_empty_reference_counts_all_insertions(self):
        self.assertEqual(scorer.edit_counts([], ["a", "b"]), {
            "substitutions": 0, "deletions": 0, "insertions": 2,
            "errors": 2, "referenceWords": 0,
        })

    def test_alignment_tie_prefers_substitutions(self):
        counts = scorer.edit_counts(["a", "b"], ["b", "a"])
        self.assertEqual((counts["substitutions"], counts["deletions"], counts["insertions"]), (2, 0, 0))


class ScoringTests(unittest.TestCase):
    def test_micro_wer_common_fixed_denominator_and_raw_preservation(self):
        raw, fixtures = inputs()
        raw["measurements"][1]["text"] = "A dog."  # One substitution on two words.
        raw["measurements"][3]["text"] = "A small cat extra."  # One insertion on three words.
        original = copy.deepcopy(raw)
        result = scorer.score(raw, fixtures)
        self.assertEqual(result["raw"], original)
        self.assertEqual(raw, original)
        self.assertEqual(result["aggregates"]["parakeet_v2"]["referenceWords"], 47)
        self.assertEqual(result["aggregates"]["orukeet"]["referenceWords"], 47)
        self.assertEqual(result["aggregates"]["orukeet"]["errors"], 2)
        self.assertAlmostEqual(result["aggregates"]["orukeet"]["microWER"], 2 / 47)
        self.assertAlmostEqual(result["microWERDeltaOrukeetMinusParakeetV2"], 2 / 47)
        self.assertEqual(result["clips"][0]["WERDeltaOrukeetMinusParakeetV2"], 0.5)
        self.assertFalse(result["scope"]["modelPreferenceChanged"])

    def test_sealed_reference_and_hyphen_denominator_are_unchanged(self):
        raw, fixtures = inputs()
        fixtures["fixtures"][0]["reference"] = "Film-editing."
        fixtures["fixtures"][0]["normalized_reference"] = "film-editing"
        for measurement in raw["measurements"][:2]:
            measurement["text"] = "Film-editing"
        clip = scorer.score(raw, fixtures)["clips"][0]
        self.assertEqual(clip["sealedNormalizedReference"], "film-editing")
        self.assertEqual(clip["normalizedReference"], "film-editing")
        self.assertEqual(clip["referenceWords"], 1)
        for model in scorer.MODELS:
            self.assertEqual(clip["models"][model]["WER"], 0)

    def test_scoring_never_recomputes_sealed_reference_from_raw_reference(self):
        raw, fixtures = inputs()
        fixtures["fixtures"][0]["reference"] = "A wholly different raw reference."
        fixtures["fixtures"][0]["normalized_reference"] = "a cat"
        clip = scorer.score(raw, fixtures)["clips"][0]
        self.assertEqual(clip["referenceWords"], 2)
        for model in scorer.MODELS:
            self.assertEqual(clip["models"][model]["WER"], 0)

    def test_empty_transcript_stays_in_denominator(self):
        raw, fixtures = inputs()
        raw["measurements"][0]["text"] = ""
        score = scorer.score(raw, fixtures)["aggregates"]["parakeet_v2"]
        self.assertEqual((score["deletions"], score["referenceWords"]), (2, 47))

    def test_complete_pairs_can_arrive_in_any_order(self):
        raw, fixtures = inputs()
        before = scorer.score(raw, fixtures)["aggregates"]
        raw["measurements"].reverse()
        self.assertEqual(scorer.score(raw, fixtures)["aggregates"], before)

    def test_rejects_incomplete_or_corrupt_inputs(self):
        cases = {
            "missing": lambda raw, fixture: raw["measurements"].pop(),
            "duplicate": lambda raw, fixture: raw["measurements"].append(copy.deepcopy(raw["measurements"][0])),
            "extra audio": lambda raw, fixture: raw["measurements"][0].update(audio="extra.wav"),
            "extra model": lambda raw, fixture: raw["measurements"][0].update(model="other"),
            "hash mismatch": lambda raw, fixture: raw["measurements"][0].update(audioSHA256="f" * 64),
            "inference error": lambda raw, fixture: raw["measurements"][0].update(error="failed to load"),
            "missing text": lambda raw, fixture: raw["measurements"][0].pop("text"),
            "null text": lambda raw, fixture: raw["measurements"][0].update(text=None),
            "too few fixtures": lambda raw, fixture: fixture["fixtures"].pop(),
            "duplicate fixture": lambda raw, fixture: fixture["fixtures"].__setitem__(1, fixture["fixtures"][0]),
            "empty reference": lambda raw, fixture: fixture["fixtures"][0].update(normalized_reference="  "),
            "malformed digest": lambda raw, fixture: fixture["fixtures"][0].update(sha256="not-a-digest"),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                raw, fixtures = inputs()
                mutate(raw, fixtures)
                with self.assertRaises(ValueError):
                    scorer.score(raw, fixtures)

    def test_cli_records_input_digests_and_preserves_output_on_invalid_input(self):
        raw, fixtures = inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path, fixtures_path, output_path = (root / name for name in ("raw.json", "fixtures.json", "scored.json"))
            raw_path.write_text(json.dumps(raw))
            fixtures_path.write_text(json.dumps(fixtures))
            arguments = ["--raw", str(raw_path), "--fixtures", str(fixtures_path), "--output", str(output_path)]
            self.assertEqual(scorer.main(arguments), 0)
            result = json.loads(output_path.read_text())
            self.assertEqual(result["inputSHA256"]["raw"], hashlib.sha256(raw_path.read_bytes()).hexdigest())
            valid_output = output_path.read_bytes()
            raw["measurements"].pop()
            raw_path.write_text(json.dumps(raw))
            self.assertEqual(scorer.main(arguments), 1)
            self.assertEqual(output_path.read_bytes(), valid_output)


if __name__ == "__main__":
    unittest.main()
