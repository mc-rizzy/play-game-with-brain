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

    def test_replay_aux_channels(self):
        import time
        from gnn.source import ReplaySource
        from gnn.model import CHANNELS
        src = ReplaySource(os.path.join(ROOT, "EEG_recordings"),
                           realtime=False, expected_channels=CHANNELS)
        with src:
            for _ in range(100):
                if src.buffered_seconds > 0.5:
                    break
                time.sleep(0.05)
            contacts = src.read_contacts(CHANNELS)
            self.assertIsNotNone(contacts)
            self.assertEqual(set(contacts), set(CHANNELS))
            self.assertTrue(all(isinstance(v, float)
                                for v in contacts.values()))
            accel = src.read_accel()
            self.assertIsNotNone(accel)
            self.assertEqual(len(accel), 3)
            battery = src.read_battery()
            self.assertTrue(battery is None or isinstance(battery, int))

    def test_full_trial_carries_aux(self):
        from gnn import recordings
        from gnn.model import CHANNELS
        _label, path = recordings.find_trials(
            os.path.join(ROOT, "EEG_recordings"))[0]
        trial = recordings.load_trial(path, expected_channels=CHANNELS,
                                      full=True)
        self.assertEqual(trial["full_data"].shape[1],
                         trial["data"].shape[1])
        self.assertIn("contact_F3", trial["full_names"])
        self.assertIn("Battery", trial["full_names"])
        self.assertIn("Accel_x", trial["full_names"])

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
class TestModalTrain(unittest.TestCase):
    def _import_with_fake_modal(self):
        import sys
        import types

        class _Chain:
            def pip_install(self, *a, **k):
                return self

            def add_local_dir(self, *a, **k):
                return self

        class _Image:
            @staticmethod
            def debian_slim(**k):
                return _Chain()

        class _App:
            def __init__(self, *a, **k):
                pass

            def function(self, *a, **k):
                return lambda fn: fn

            def local_entrypoint(self, *a, **k):
                return lambda fn: fn

        fake = types.ModuleType("modal")
        fake.App = _App
        fake.Image = _Image
        sys.modules["modal"] = fake
        try:
            import gnn.modal_train as mt
            return mt
        finally:
            del sys.modules["modal"]

    def test_prepare_payload_needs_two_classes(self):
        mt = self._import_with_fake_modal()
        trials = [{"label": "left", "path": "t0", "trial_id": "t0",
                   "data": np.random.RandomState(0).randn(8, 1000)}]
        with self.assertRaises(SystemExit):
            mt.prepare_payload(trials)

    def test_prepare_payload_shapes(self):
        mt = self._import_with_fake_modal()
        rng = np.random.RandomState(0)
        trials = [{"label": lab, "path": f"t{i}", "trial_id": f"t{i}",
                   "data": rng.randn(8, 1000)}
                  for i, lab in enumerate(["left"] * 2 + ["right"] * 2)]
        F_tr, y_tr, F_va, y_va, meta = mt.prepare_payload(trials)
        self.assertEqual(F_tr.shape[1:], (8, 12))
        self.assertEqual(F_va.shape[1:], (8, 12))
        self.assertEqual(meta["labels"], ["left", "right"])
        self.assertEqual(meta["n_trials"], 4)

    def test_dotenv_auth(self):
        import tempfile
        from unittest import mock

        mt = self._import_with_fake_modal()
        saved = {k: os.environ.get(k) for k in
                 ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_TOKEN")}
        for k in saved:
            os.environ.pop(k, None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                # missing file -> empty, no crash
                self.assertEqual(mt.load_dotenv(tmp), {})
                with open(os.path.join(tmp, ".env"), "w") as fh:
                    fh.write('# comment\n\nMODAL_TOKEN_ID="ak-test"\n'
                             "MODAL_TOKEN_SECRET='as-test'\n")
                self.assertEqual(mt.load_dotenv(tmp),
                                 {"MODAL_TOKEN_ID": "ak-test",
                                  "MODAL_TOKEN_SECRET": "as-test"})
                # both exported from .env
                self.assertEqual(mt.ensure_modal_auth(tmp, strict=False),
                                 ".env")
                self.assertEqual(os.environ["MODAL_TOKEN_ID"], "ak-test")
                self.assertEqual(os.environ["MODAL_TOKEN_SECRET"], "as-test")
                # existing env wins over .env
                os.environ["MODAL_TOKEN_ID"] = "ak-env"
                self.assertEqual(mt.ensure_modal_auth(tmp, strict=False),
                                 "env")
                self.assertEqual(os.environ["MODAL_TOKEN_ID"], "ak-env")
                # lone MODAL_TOKEN counts as the ID, but the missing
                # secret fails fast when no `modal setup` login exists
                os.environ.pop("MODAL_TOKEN_ID", None)
                os.environ.pop("MODAL_TOKEN_SECRET", None)
                with open(os.path.join(tmp, ".env"), "w") as fh:
                    fh.write("MODAL_TOKEN=ak-only\n")
                with tempfile.TemporaryDirectory() as home, \
                        mock.patch.dict(os.environ, {"HOME": home}):
                    with self.assertRaises(SystemExit) as ctx:
                        mt.ensure_modal_auth(tmp, strict=True)
                    self.assertIn("MODAL_TOKEN_SECRET", str(ctx.exception))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_fit_remote_body_runs_locally(self):
        mt = self._import_with_fake_modal()
        rng = np.random.RandomState(0)
        F_tr = rng.randn(16, 8, 12).astype(np.float32)
        y_tr = np.array([0, 1] * 8)
        result = mt.fit_remote(F_tr, y_tr, F_tr[:4], y_tr[:4],
                               hidden=8, epochs=1, batch=8)
        self.assertIn("state_dict", result)
        self.assertIn("best_val_acc", result)
        self.assertEqual(result["n_classes"], 5)
        self.assertEqual(len(result["history"]), 1)


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
            dash = Dashboard(src)
            dash.show_car = True  # also cover the CAR + PSD-select paths
            dash.selected = 3
            dash.run(frames=3)


if __name__ == "__main__":
    unittest.main()
