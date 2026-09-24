import hashlib
import io
import json
import unittest
import zipfile

from verify_bundle import REQUIRED, SOURCE_SHA256, safe_name, verify_contents


class BundleVerificationTests(unittest.TestCase):
    def make_archive(self, mutation=None):
        payload = {name: b"fixture" for name in REQUIRED}
        payload["parakeet_vocab.json"] = json.dumps({str(i): str(i) for i in range(8192)}).encode()
        manifest = {"model": "Orukeet", "source": "r3", "source_sha256": SOURCE_SHA256,
                    "profile": "greedy", "sample_rate": 16000, "fluid_audio_version": "0.15.5",
                    "compiled_cache_included": False,
                    "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                              for name, data in payload.items()}}
        if mutation:
            mutation(payload, manifest)
        payload["bundle.json"] = json.dumps(manifest).encode()
        result = io.BytesIO()
        with zipfile.ZipFile(result, "w") as archive:
            for name, data in payload.items():
                archive.writestr("bundle/" + name, data)
        result.seek(0)
        return zipfile.ZipFile(result)

    def test_valid_portable_bundle(self):
        with self.make_archive() as archive:
            self.assertEqual(verify_contents(archive, "bundle/", "greedy"), len(REQUIRED))

    def test_changed_weight_rejected(self):
        with self.make_archive(lambda payload, manifest: payload.update({"Encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin": b"changed"})) as archive:
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_contents(archive, "bundle/", "greedy")

    def test_unlisted_payload_rejected(self):
        with self.make_archive(lambda payload, manifest: payload.update({"extra": b"x"})) as archive:
            with self.assertRaisesRegex(ValueError, "unlisted"):
                verify_contents(archive, "bundle/", "greedy")

    def test_missing_component_rejected(self):
        def remove(payload, manifest):
            payload.pop("LICENSE-WEIGHTS")
            manifest["files"].pop("LICENSE-WEIGHTS")
        with self.make_archive(remove) as archive:
            with self.assertRaisesRegex(ValueError, "Missing required"):
                verify_contents(archive, "bundle/", "greedy")

    def test_identity_mismatch_rejected(self):
        with self.make_archive(lambda payload, manifest: manifest.update({"source_sha256": "wrong"})) as archive:
            with self.assertRaisesRegex(ValueError, "identity"):
                verify_contents(archive, "bundle/", "greedy")

    def test_paths_cannot_escape_or_alias(self):
        for name in ("../x", "/x", "a/../x", "a\\x", "a//x", "a/./x", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_name(name)


if __name__ == "__main__":
    unittest.main()
