import io
import struct
import unittest

import numpy as np
import torch

from convert import fit_palette, pack_six_bit, reorder_lstm, write_blob


class ConversionTests(unittest.TestCase):
    def test_six_bit_packing_matches_mil_little_endian_contract(self):
        for n in (1, 3, 4, 5, 63, 65):
            values = (np.arange(n) % 64).astype(np.uint8)
            packed = pack_six_bit(values)
            bits = np.unpackbits(packed, bitorder="little")[:n * 6].reshape(n, 6)
            decoded = (bits * (1 << np.arange(6))).sum(axis=1)
            np.testing.assert_array_equal(decoded, values)
            self.assertEqual(len(packed), (n * 6 + 7) // 8)

    def test_palette_fit_handles_repeated_values_and_signed_zero(self):
        values = np.array([-2, -2, -0.0, 0.0, 3, 3, 3], np.float32)
        initial = np.linspace(-3, 4, 64).astype(np.float16)
        lut, indices = fit_palette(values, initial)
        np.testing.assert_array_equal(lut[indices], values.astype(np.float16))
        again = fit_palette(values, initial)
        np.testing.assert_array_equal(lut, again[0])
        np.testing.assert_array_equal(indices, again[1])

    def test_palette_rejects_nonfinite_weights(self):
        with self.assertRaises(ValueError):
            fit_palette(np.array([np.nan]), np.zeros(64))

    def test_lstm_gate_order(self):
        tensor = torch.arange(8).reshape(8, 1)
        self.assertEqual(reorder_lstm(tensor).flatten().tolist(), [0, 1, 2, 3, 6, 7, 4, 5])

    def test_blob_write_preserves_header_and_neighbors(self):
        from coremltools.proto import MIL_pb2
        value = MIL_pb2.Value()
        value.blobFileValue.fileName = "@model_path/weights/weight.bin"
        value.blobFileValue.offset = 64
        original = b"X" * 64 + struct.pack("<IIQQ", 0xDEADBEEF, 1, 4, 128) + b"Y" * 40 + b"0000TAIL"
        handle = io.BytesIO(original)
        write_blob(handle, value, np.array([1, 2], np.float16))
        self.assertEqual(handle.getvalue()[:128], original[:128])
        self.assertEqual(handle.getvalue()[132:], b"TAIL")
        with self.assertRaises(ValueError):
            write_blob(handle, value, np.array([1], np.float16))


if __name__ == "__main__":
    unittest.main()
