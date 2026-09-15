import copy
import unittest

from summarize import summarize


class SummaryTests(unittest.TestCase):
    def report(self):
        return {"mode": "batch", "paced": False, "encoderUnits": "ane", "repetitions": 2,
                "models": [{"label": "base"}, {"label": "candidate"}],
                "measurements": [
                    {"model": label, "audioSHA256": "fixture", "audio": "/real.wav", "repetition": rep,
                     "audioSeconds": 10, "wallMs": ms, "text": "hello", "tokenIDs": [42], "thermalState": 0}
                    for rep in range(2) for label, ms in [("base", 100), ("candidate", 80)]
                ]}

    def test_paired_ratio_and_token_parity(self):
        report = self.report()
        result = summarize(report, "base", ["candidate"])
        self.assertEqual(result["comparisons"]["candidate"]["median_latency_ratio"], .8)
        self.assertFalse(result["failures"])
        report["measurements"][1]["tokenIDs"] = [43]
        self.assertTrue(summarize(report, "base", ["candidate"])["failures"])

    def test_missing_pair_fails(self):
        report = self.report()
        report["measurements"].pop()
        with self.assertRaises(ValueError):
            summarize(report, "base")

    def test_duplicate_measurement_fails(self):
        report = self.report()
        report["measurements"].append(copy.deepcopy(report["measurements"][0]))
        with self.assertRaises(ValueError):
            summarize(report, "base")

    def test_missing_complete_repetition_fails(self):
        report = self.report()
        del report["measurements"][-2:]
        with self.assertRaises(ValueError):
            summarize(report, "base")

    def test_paced_wall_time_is_not_reported_as_rtfx(self):
        report = self.report()
        report["paced"] = True
        result = summarize(report, "base")
        self.assertTrue(all(row["rtfx"] is None for row in result["fixtures"]))


if __name__ == "__main__":
    unittest.main()
