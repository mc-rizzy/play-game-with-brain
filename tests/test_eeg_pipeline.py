"""EEG pipeline tests: recordings, features, training, live bridge.

Needs numpy/scipy/torch (.venv). Skips cleanly when missing.
Run:  .venv/Scripts/python -m unittest discover -s tests   (Windows)
"""
import os
import unittest

try:
    import numpy as np
    import torch
    HAVE_ML = True
except ImportError:
    np = None
    torch = None
    HAVE_ML = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@unittest.skipUnless(HAVE_ML, "numpy/torch not installed")
class TestRecordings(unittest.TestCase):
    def test_example_left_trial(self):
        from gnn import recordings
        from gnn.model import CHANNELS
        trials = recordings.find_trials(os.path.join(ROOT, "EEG_recordings"))
        self.assertEqual(len(trials), 1)
        label, path = trials[0]
        self.assertEqual(label, "left")
        trial = recordings.load_trial(path, expected_channels=CHANNELS)
        n_ch, total = trial["data"].shape
        self.assertEqual(n_ch, 8)
        # ~12-13 s example; chunk widths vary (25/50), count varies
        # slightly with WAL recovery, so assert internal consistency
        self.assertGreater(total, 2500)
        self.assertEqual(total, sum(trial["chunk_samples"]))
        self.assertEqual(len(trial["time"]), total)
        self.assertEqual(trial["sf"], 250.0)
        self.assertEqual(trial["channels"], CHANNELS)

    def test_command_folders_exist(self):
        for cmd in ("up", "down", "left", "right", "interact"):
            d = os.path.join(ROOT, "EEG_recordings", cmd,
                             "sub-001", "ses-001")
            self.assertTrue(os.path.isdir(d), f"{cmd} skeleton missing")


@unittest.skipUnless(HAVE_ML, "numpy/torch not installed")
class TestDataset(unittest.TestCase):
    def test_windowing_and_split_hygiene(self):
        from gnn.dataset import (build_dataset, trial_split, sliding_windows,
                                 LABEL_TO_INDEX)
        rng = np.random.RandomState(0)
        trials = []
        for i in range(10):
            label = "left" if i < 4 else "right"
            trials.append({"label": label, "path": f"t{i}",
                           "trial_id": f"t{i}",
                           "data": rng.randn(8, 1000)})
        X, y, tids = build_dataset(trials)
        per = len(range(0, 1000 - 500 + 1, 62))
        self.assertEqual(X.shape, (10 * per, 8, 500))
        self.assertEqual(set(y.tolist()),
                         {LABEL_TO_INDEX["left"], LABEL_TO_INDEX["right"]})
        tr, va = trial_split(tids, y, 0.2, seed=1)
        self.assertTrue(len(tr) and len(va))
        self.assertEqual(set(tids[tr]) & set(tids[va]), set())
        # windows counts match the formula on the real example too
        self.assertEqual(sliding_windows(np.zeros((8, 3075))).shape[0],
                         len(range(0, 3075 - 500 + 1, 62)))


@unittest.skipUnless(HAVE_ML, "numpy/torch not installed")
class TestFeatures(unittest.TestCase):
    def test_preprocess_car_and_shapes(self):
        from gnn.model import preprocess, extract_node_features
        rng = np.random.RandomState(0)
        epoch = rng.randn(8, 500) * 20 + 128000  # huge DC like raw Mini data
        clean = preprocess(epoch)
        self.assertEqual(clean.shape, (8, 500))
        # CAR: mean across channels ~ 0 at every time point
        self.assertLess(np.abs(clean.mean(axis=0)).max(), 1e-6)
        feats = extract_node_features(epoch)
        self.assertEqual(feats.shape, (8, 12))
        self.assertTrue(np.isfinite(feats).all())
        # standardized across nodes
        self.assertLess(np.abs(feats.mean(axis=0)).max(), 1e-5)

    def test_adjacency(self):
        from gnn.model import build_static_adj, CHANNELS
        A = build_static_adj(CHANNELS).numpy()
        self.assertEqual(A.shape, (8, 8))
        self.assertTrue(np.allclose(np.diag(A), 1.0))
        self.assertTrue(np.allclose(A, A.T))
        self.assertGreaterEqual(A.min(), 0.0)
        self.assertLessEqual(A.max(), 1.0)


@unittest.skipUnless(HAVE_ML, "numpy/torch not installed")
class TestModelTrain(unittest.TestCase):
    def test_forward_and_tiny_train(self):
        import tempfile
        from gnn.model import EEGGraphNet, train, load_checkpoint
        model = EEGGraphNet()
        out = model(torch.randn(2, 8, 12))
        self.assertEqual(tuple(out.shape), (2, 5))
        # linearly separable synthetic features
        rng = np.random.RandomState(0)
        X0 = rng.randn(20, 8, 12).astype(np.float32) + 1.5
        X1 = rng.randn(20, 8, 12).astype(np.float32) - 1.5
        X = np.concatenate([X0, X1])
        y = np.array([0] * 20 + [1] * 20)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pt")
            _hist, best = train(EEGGraphNet(n_classes=2), X[:32], y[:32],
                                X[32:], y[32:], epochs=5, batch=8,
                                save_path=path, log_every=10)
            self.assertGreater(best, 0.5)
            self.assertTrue(os.path.exists(path))
            loaded, cfg = load_checkpoint(path)
            self.assertEqual(tuple(loaded(torch.randn(1, 8, 12)).shape),
                             (1, 2))
            self.assertEqual(len(cfg["channels"]), 8)


@unittest.skipUnless(HAVE_ML, "numpy/torch not installed")
class TestLiveBridge(unittest.TestCase):
    def test_gate_and_mapping(self):
        from gnn.live import BrainBridge
        from gnn.model import EEGGraphNet
        strict = BrainBridge(EEGGraphNet(), confidence=0.9999)
        rng = np.random.RandomState(0)
        cmd, probs = strict.step(rng.randn(8, 500))
        self.assertIsNone(cmd)  # impossible gate -> no command, game safe
        self.assertEqual(len(probs), 5)
        loose = BrainBridge(EEGGraphNet(), confidence=0.0, cooldown_s=0.0)
        cmd, _probs = loose.step(rng.randn(8, 500))
        self.assertIn(cmd, ("UP", "DOWN", "LEFT", "RIGHT", "INTERACT"))
        self.assertEqual(loose.command_to_input("LEFT")["move"], -1.0)
        self.assertEqual(loose.command_to_input("RIGHT")["move"], 1.0)
        self.assertTrue(loose.command_to_input("UP")["jump"])
        self.assertTrue(loose.command_to_input("DOWN")["place"])
        self.assertTrue(loose.command_to_input("INTERACT")["attack"])
        neutral = loose.command_to_input(None)
        self.assertEqual(neutral["move"], 0.0)
        self.assertFalse(neutral["jump"])

    def test_replay_source_streams(self):
        import time
        from gnn.source import ReplaySource
        from gnn.model import CHANNELS
        src = ReplaySource(os.path.join(ROOT, "EEG_recordings"),
                           realtime=False, expected_channels=CHANNELS)
        with src:
            window = None
            for _ in range(100):
                window = src.read_window(seconds=2.0, channels=CHANNELS)
                if window is not None:
                    break
                time.sleep(0.05)
        self.assertIsNotNone(window)
        self.assertEqual(window.shape, (8, 500))

    def test_game_input_roundtrip(self):
        from gnn.live import write_game_input, reset_game_input, GAME_INPUT
        from game import ai_control
        try:
            write_game_input(move=-1.0, jump=True)
            self.assertTrue(os.path.exists(GAME_INPUT))
            action = ai_control.load_file_action()
            self.assertEqual(action["move"], -1.0)
            self.assertTrue(action["jump"])
        finally:
            reset_game_input()
        self.assertFalse(os.path.exists(GAME_INPUT))


@unittest.skipUnless(HAVE_ML, "numpy/torch not installed")
class TestVizSmoke(unittest.TestCase):
    def test_dashboard_frames_headless(self):
        try:
            import pygame  # noqa: F401
        except ImportError:
            self.skipTest("pygame not installed")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        from gnn.source import ReplaySource
        from gnn.viz import Dashboard
        from gnn.model import CHANNELS
        src = ReplaySource(os.path.join(ROOT, "EEG_recordings"),
                           realtime=False, expected_channels=CHANNELS)
        with src:
            import time
            time.sleep(0.5)  # let the pump buffer windows
            Dashboard(src).run(frames=3)


if __name__ == "__main__":
    unittest.main()
