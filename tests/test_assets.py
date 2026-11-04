"""Pixel-art asset tests (stdlib only: PNG magic + IHDR dimensions)."""
import os
import struct
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "game", "assets")

EXPECTED = {
    "player_idle.png": (16, 24),
    "player_walk1.png": (16, 24),
    "player_walk2.png": (16, 24),
    "slime_0.png": (16, 16),
    "slime_1.png": (16, 16),
    "shade_0.png": (16, 32),
    "shade_1.png": (16, 32),
    "bat_0.png": (16, 16),
    "bat_1.png": (16, 16),
    "skeleton_0.png": (16, 24),
    "skeleton_1.png": (16, 24),
    "sword.png": (16, 16),
    "heart_full.png": (16, 16),
    "heart_empty.png": (16, 16),
}

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def png_size(path):
    with open(path, "rb") as fh:
        data = fh.read(33)
    assert data[:8] == PNG_MAGIC, f"{path}: bad PNG magic"
    (length,) = struct.unpack(">I", data[8:12])
    assert data[12:16] == b"IHDR", f"{path}: missing IHDR"
    assert length == 13, f"{path}: bad IHDR length"
    w, h = struct.unpack(">II", data[16:24])
    return w, h


class TestAssets(unittest.TestCase):
    def test_all_assets_exist_and_valid(self):
        self.assertTrue(os.path.isdir(ASSETS), "game/assets folder missing")
        for name, (ew, eh) in EXPECTED.items():
            path = os.path.join(ASSETS, name)
            self.assertTrue(os.path.isfile(path), f"{name} missing")
            self.assertGreater(os.path.getsize(path), 0, f"{name} empty")
            w, h = png_size(path)
            self.assertEqual((w, h), (ew, eh), f"{name} size")


if __name__ == "__main__":
    unittest.main()
