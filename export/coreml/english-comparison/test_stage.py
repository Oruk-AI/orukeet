import hashlib
import io
import os
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import stage


class StagingTests(unittest.TestCase):
    def test_staging_refuses_local_execution_before_downloading(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
            with patch("urllib.request.urlopen") as request:
                with self.assertRaises(RuntimeError):
                    stage.stage_v2(Path(temporary) / "models")
                with self.assertRaises(RuntimeError):
                    stage.stage_fixtures(Path(temporary) / "audio")
                request.assert_not_called()
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_asset_authentication_rejects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.mil"
            path.write_bytes(b"valid graph")
            entry = {"path": "model.mil", "size_bytes": 11,
                     "git_blob_sha1": hashlib.sha1(b"blob 11\0valid graph").hexdigest(), "sha256": None}
            self.assertEqual(stage.verify_asset(path, entry)["sha256"], hashlib.sha256(b"valid graph").hexdigest())
            path.write_bytes(b"wrong graph")
            with self.assertRaisesRegex(ValueError, "Git blob"):
                stage.verify_asset(path, entry)
            entry["sha256"] = hashlib.sha256(b"valid graph").hexdigest()
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                stage.verify_asset(path, entry)

    def test_streamed_archive_extracts_only_selected_exact_bytes(self):
        fmt = struct.pack("<HHIIHH", 3, 1, 16000, 64000, 4, 32)
        body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", 8) + struct.pack("<ff", 0.1, -0.2)
        data = b"RIFF" + struct.pack("<I", len(body)) + body
        fixture = {"path": "selected.wav", "archive_member": "test/selected.wav", "samples": 2,
                   "sha256": hashlib.sha256(data).hexdigest()}
        archive_bytes = io.BytesIO()
        with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
            for name, value in [("../../ignored.bin", b"not selected"), (fixture["archive_member"], data)]:
                item = tarfile.TarInfo(name)
                item.size = len(value)
                archive.addfile(item, io.BytesIO(value))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stage.extract_selected_audio(io.BytesIO(archive_bytes.getvalue()), directory, [fixture])
            self.assertEqual([p.name for p in directory.iterdir()], ["selected.wav"])
            self.assertEqual((directory / "selected.wav").read_bytes(), data)
            with self.assertRaisesRegex(ValueError, "missing"):
                stage.extract_selected_audio(io.BytesIO(archive_bytes.getvalue()), directory,
                                             [{**fixture, "archive_member": "test/absent.wav"}])

    def test_checked_in_manifests_keep_exact_sealed_selection(self):
        fixtures = stage.fixture_manifest()["fixtures"]
        self.assertEqual(len(fixtures), 16)
        self.assertEqual(sum(f["samples"] for f in fixtures), 2404480)
        self.assertEqual(len(stage.v2_manifest()["files"]), 21)


if __name__ == "__main__":
    unittest.main()
