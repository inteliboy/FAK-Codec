"""Unit tests for the verification path (tools/benchmark/wavio.py).

The verifier is the project's correctness oracle, so it gets its own tests:
round-trip fidelity over formats/extremes, and *negative* tests proving that any
single-bit change is detected.
"""
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "benchmark"))
from wavio import Pcm, pack_samples, read_wav, write_wav  # noqa: E402


def rand_pcm(rng, ch, bits, n, extremes=False):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if extremes:
        chans = [[rng.choice((lo, hi, 0, -1, 1)) for _ in range(n)] for _ in range(ch)]
    else:
        chans = [[rng.randint(lo, hi) for _ in range(n)] for _ in range(ch)]
    return Pcm(ch, 44100, bits, pack_samples(chans, bits)), chans


class WavRoundTrip(unittest.TestCase):
    def test_roundtrip_formats(self):
        rng = random.Random(1234)
        with tempfile.TemporaryDirectory() as td:
            for ch in (1, 2, 3, 6, 8):
                for bits in (8, 16, 24):
                    for extremes in (False, True):
                        for n in (0 + 1, 7, 1000):
                            pcm, _ = rand_pcm(rng, ch, bits, n, extremes)
                            p = Path(td) / "t.wav"
                            write_wav(p, pcm)
                            back = read_wav(p)
                            self.assertEqual(back.format_key(), pcm.format_key())
                            self.assertEqual(back.data, pcm.data)
                            self.assertEqual(back.sha256(), pcm.sha256())
                            self.assertEqual(back.frames, n)

    def test_pack_samples_known_values(self):
        self.assertEqual(pack_samples([[1, -1]], 16), b"\x01\x00\xff\xff")
        self.assertEqual(pack_samples([[-8388608, 8388607]], 24), b"\x00\x00\x80\xff\xff\x7f")
        self.assertEqual(pack_samples([[-128, 127]], 8), b"\x00\xff")
        # channel interleave order: L0 R0 L1 R1
        self.assertEqual(pack_samples([[1, 3], [2, 4]], 8), bytes([129, 130, 131, 132]))


class VerifierCatchesCorruption(unittest.TestCase):
    def test_every_single_bit_flip_changes_hash(self):
        rng = random.Random(99)
        pcm, _ = rand_pcm(rng, 2, 16, 8)
        base = pcm.sha256()
        for byte in range(len(pcm.data)):
            for bit in range(8):
                d = bytearray(pcm.data)
                d[byte] ^= 1 << bit
                self.assertNotEqual(Pcm(2, 44100, 16, bytes(d)).sha256(), base)

    def test_format_mismatch_is_distinguishable(self):
        a = Pcm(2, 44100, 16, bytes(8))
        self.assertNotEqual(a.format_key(), Pcm(1, 44100, 16, bytes(8)).format_key())
        self.assertNotEqual(a.format_key(), Pcm(2, 48000, 16, bytes(8)).format_key())
        self.assertNotEqual(a.format_key(), Pcm(2, 44100, 24, bytes(8)).format_key())

    def test_truncated_or_bad_files_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.wav"
            p.write_bytes(b"not a wav file at all")
            with self.assertRaises(ValueError):
                read_wav(p)
            pcm, _ = rand_pcm(random.Random(5), 2, 16, 10)
            write_wav(p, pcm)
            # cut mid-frame: data length no longer a multiple of the frame size
            raw = p.read_bytes()
            p.write_bytes(raw[:-1])
            with self.assertRaises(ValueError):
                read_wav(p)


if __name__ == "__main__":
    unittest.main()
