"""Dual-layer meta-pipeline tests: PSD features, spatial GNN, few-shot
adapter, Muse client (mocked), calibration, wizard, live bridge.

Needs numpy/scipy/torch (.venv). Skips cleanly when missing.
Run:  .venv/Scripts/python -m unittest discover -s tests   (Windows)
"""
import json
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

FAKE_MUSE_CALIB = {
    "coaching": ["Relax your jaw; keep eyes on the cue."],
    "thresholds": {"confidence": 0.65, "cooldown_s": 0.4, "ema": 0.75},
    "verdict": "Clean motor imagery; gates loosened slightly.",
}


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _mock_chat(content):
    def fake_urlopen(req, timeout=None, **_kw):
        return _FakeResp({"choices": [{"message": {"content": content}}]})
    return fake_urlopen


@unittest.skipUnless(HAVE_ML, "numpy/torch not installed")
class TestFastFeatures(unittest.TestCase):
    def test_chunk_features_shape(self):
        from gnn.features import N_FAST_FEATURES, extract_chunk_features
        rng = np.random.RandomState(0)
        feats = extract_chunk_features(rng.randn(8, 125) * 20)
        self.assertEqual(feats.shape, (8, N_FAST_FEATURES))
        self.assertEqual(N_FAST_FEATURES, 8)
        self.assertTrue(np.isfinite(feats).all())

    def test_alpha_peak_detected(self):
        from gnn.features import bandpowers
        t = np.arange(500) / 250.0
        alpha = np.sin(2 * np.pi * 10 * t) * 30  # strong 10 Hz alpha
        epoch = np.random.RandomState(0).randn(8, 500) * 5
        epoch[2] += alpha
        bp = bandpowers(epoch)
        self.assertEqual(bp.shape, (8, 4))  # theta/alpha/beta/gamma
        self.assertEqual(int(np.argmax(bp[2])), 1)  # alpha wins on ch 2

    def test_rolling_psd_stream(self):
        from gnn.features import RollingPSD
        roller = RollingPSD(n_ch=8)
        rng = np.random.RandomState(1)
        self.assertFalse(roller.frame_ready)
        for _ in range(4):  # 4 x 0.5 s = 2 s -> 4 chunks
            roller.push(rng.randn(8, 125) * 10)
        self.assertTrue(roller.frame_ready)
        self.assertEqual(roller.frame().shape, (8, 8))
        self.assertEqual(roller.frame_epoch().shape, (8, 500))
        with self.assertRaises(ValueError):
            roller.push(rng.randn(4, 62))

    def test_fast_featurize_matches_train(self):
        from gnn.train import featurize_fast
        rng = np.random.RandomState(0)
        F = featurize_fast(rng.randn(3, 8, 500) * 20)
        self.assertEqual(F.shape, (3, 8, 8))
        self.assertTrue(np.isfinite(F).all())


@unittest.skipUnless(HAVE_ML, "numpy/torch not installed")
class TestSpatialGraph(unittest.TestCase):
    def test_spatial_adj(self):
        from gnn.graph import spatial_adj
        A = spatial_adj()
        self.assertEqual(A.shape, (8, 8))
        self.assertTrue(np.allclose(np.diag(A), 1.0))
        self.assertTrue(np.allclose(A, A.T))
        # neighbours (F3-C3) beat opposite corners (F3-O2)
        self.assertGreater(A[0, 2], A[0, 7])

    def test_functional_adj(self):
        from gnn.graph import functional_adj
        rng = np.random.RandomState(0)
        drive = rng.randn(500)
        epoch = rng.randn(8, 500) * 0.5
        epoch[0] = drive + rng.randn(500) * 0.1
        epoch[1] = drive + rng.randn(500) * 0.1
        A = functional_adj(epoch)
        self.assertEqual(A.shape, (8, 8))
        self.assertTrue(np.allclose(np.diag(A), 1.0))
        self.assertGreater(A[0, 1], 0.8)   # coupled pair
        self.assertLess(A[0, 5], 0.5)      # independent pair

    def test_combined_blend(self):
        from gnn.graph import combined_adj, functional_adj, spatial_adj
        A = combined_adj(spatial_adj(), functional_adj(
            np.random.RandomState(0).randn(8, 500)))
        self.assertEqual(A.shape, (8, 8))
        with self.assertRaises(ValueError):
            combined_adj(spatial_adj(), np.zeros((4, 4)))

    def test_spatial_gnn_matches_parent_without_dynamic(self):
        from gnn.graph import SpatialGNN, as_spatial
        from gnn.model import EEGGraphNet
        torch.manual_seed(0)
        base = EEGGraphNet()
        base.eval()
        spatial = as_spatial(base)
        x = torch.randn(2, 8, 12)
        with torch.no_grad():
            self.assertTrue(torch.allclose(base(x), spatial(x), atol=1e-6))
            latent = spatial.encode(x)
        self.assertEqual(tuple(latent.shape), (2, 64))
        # dynamic edges change the message passing but keep shapes
        out = spatial(x, A_func=torch.eye(8).unsqueeze(0).expand(2, -1, -1))
        self.assertEqual(tuple(out.shape), (2, 5))

    def test_frame_graph(self):
        from gnn.graph import frame_graph
        rng = np.random.RandomState(0)
        feats, A = frame_graph(rng.randn(8, 12), rng.randn(8, 500))
        self.assertEqual(feats.shape, (8, 12))
        self.assertEqual(A.shape, (8, 8))
        with self.assertRaises(ValueError):
            frame_graph(rng.randn(4, 12))


@unittest.skipUnless(HAVE_ML, "numpy/torch not installed")
class TestFewShotAdapter(unittest.TestCase):
    def _support(self):
        rng = np.random.RandomState(0)
        X0 = rng.randn(10, 8, 12).astype(np.float32) + 1.0
        X1 = rng.randn(10, 8, 12).astype(np.float32) - 1.0
        return (np.concatenate([X0, X1]),
                np.array([0] * 10 + [1] * 10))

    def test_adapt_improves_support_fit(self):
        from gnn.adapt import adapt_head
        from gnn.model import EEGGraphNet
        X, y = self._support()
        torch.manual_seed(0)
        _adapted, info = adapt_head(EEGGraphNet(n_classes=2), X, y,
                                    steps=10, lr=0.1)
        self.assertEqual(info["support_n"], 20)
        self.assertLessEqual(info["loss_last"], info["loss_first"])
        self.assertGreaterEqual(info["after_acc"], info["before_acc"])
        self.assertFalse(info["dynamic_edges"])

    def test_adapt_with_dynamic_edges(self):
        from gnn.adapt import adapt_head
        from gnn.graph import SpatialGNN
        X, y = self._support()
        A = np.stack([np.eye(8, dtype=np.float32)] * len(X))
        torch.manual_seed(0)
        _adapted, info = adapt_head(SpatialGNN(n_classes=2), X, y, A=A,
                                    steps=2)
        self.assertTrue(info["dynamic_edges"])

    def test_adapter_roundtrip(self):
        import tempfile
        from gnn.adapt import (adapt_head, apply_adapter, load_adapter,
                               save_adapter)
        from gnn.model import EEGGraphNet
        X, y = self._support()
        torch.manual_seed(0)
        adapted, info = adapt_head(EEGGraphNet(n_classes=2), X, y, steps=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "adapter.pt")
            save_adapter(path, adapted, classes=["A", "B"],
                         thresholds={"ema": 0.7, "confidence": 0.65,
                                     "cooldown_s": 0.4},
                         calibration={"adapt": info})
            loaded = load_adapter(path)
            self.assertEqual(loaded["classes"], ["A", "B"])
            self.assertAlmostEqual(loaded["thresholds"]["confidence"], 0.65)
            fresh = EEGGraphNet(n_classes=2)
            apply_adapter(fresh, loaded)
            adapted.eval()
            fresh.eval()  # dropout off on both sides for comparison
            x = torch.tensor(X[:4])
            with torch.no_grad():
                self.assertTrue(torch.allclose(adapted(x), fresh(x),
                                               atol=1e-6))


@unittest.skipUnless(HAVE_ML, "numpy/torch not installed")
class TestMuseClient(unittest.TestCase):
    def test_api_key_env_wins(self):
        from unittest import mock
        from gnn import muse as muse_mod
        with mock.patch.dict(os.environ, {"META_TOKEN": "ak-env",
                                          "MODEL_API_KEY": ""}):
            os.environ.pop("MODEL_API_KEY", None)
            self.assertEqual(muse_mod.api_key(), "ak-env")

    def test_chat_posts_to_meta_api(self):
        from unittest import mock
        from gnn.muse import MuseClient
        seen = {}

        def fake_urlopen(req, timeout=None, **_kw):
            seen["url"] = req.full_url
            seen["auth"] = req.headers.get("Authorization")
            seen["body"] = json.loads(req.data.decode())
            return _FakeResp({"choices": [{"message": {"content": "ok"}}]})

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            text = MuseClient(api_key="k-test").chat(
                [{"role": "user", "content": "hi"}], max_tokens=8)
        self.assertEqual(text, "ok")
        self.assertEqual(seen["url"],
                         "https://api.meta.ai/v1/chat/completions")
        self.assertEqual(seen["auth"], "Bearer k-test")
        self.assertEqual(seen["body"]["model"], "muse-spark-1.3")
        self.assertIn("reasoning_effort", seen["body"])

    def test_analyze_calibration_live(self):
        from unittest import mock
        from gnn.muse import MuseClient
        client = MuseClient(api_key="k-test")
        with mock.patch("urllib.request.urlopen",
                        _mock_chat("Here you go:\n"
                                   + json.dumps(FAKE_MUSE_CALIB))):
            out = client.analyze_calibration({"separability": 0.6})
        self.assertFalse(out["offline"])
        self.assertAlmostEqual(out["thresholds"]["confidence"], 0.65)
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(client.calls[0]["ok"])

    def test_analyze_calibration_offline_fallback(self):
        from gnn.muse import MuseClient
        out = MuseClient(api_key="").analyze_calibration(
            {"separability": 0.2})
        self.assertTrue(out["offline"])
        self.assertGreaterEqual(out["thresholds"]["confidence"], 0.6)

    def test_referee_abstains_offline(self):
        from gnn.muse import MuseClient
        out = MuseClient(api_key="").interpret_command({"smoothed": {}})
        self.assertEqual(out["command"], "ABSTAIN")
        self.assertTrue(out["offline"])

    def test_ping(self):
        from unittest import mock
        from gnn.muse import MuseClient
        with mock.patch("urllib.request.urlopen", _mock_chat("ok")):
            self.assertTrue(MuseClient(api_key="k").ping())


@unittest.skipUnless(HAVE_ML, "numpy/torch not installed")
class TestCalibration(unittest.TestCase):
    def test_separability(self):
        from gnn.calibration import silhouette_separability
        rng = np.random.RandomState(0)
        X = np.concatenate([rng.randn(8, 4) + 3, rng.randn(8, 4) - 3])
        y = np.array([0] * 8 + [1] * 8)
        self.assertGreater(silhouette_separability(X, y), 0.5)
        self.assertEqual(silhouette_separability(X, np.zeros(16, int)), 0.0)

    def test_run_calibration_headless(self):
        import tempfile
        from gnn.calibration import run_calibration, synthetic_read_frame
        from gnn.graph import SpatialGNN
        from gnn.muse import MuseClient
        torch.manual_seed(0)
        read_frame = synthetic_read_frame()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_calibration(
                read_frame, SpatialGNN(), cues=(("LEFT", 0.06),
                                                ("RIGHT", 0.06)),
                adapt_steps=3, muse_client=MuseClient(api_key=""),
                out_dir=tmp, poll_s=0.01)
            self.assertGreaterEqual(result["summary"]["n_frames"], 2)
            self.assertTrue(os.path.exists(result["adapter_path"]))
            self.assertTrue(os.path.exists(result["report_path"]))
            self.assertTrue(result["muse"]["offline"])
            self.assertIn("confidence", result["thresholds"])


@unittest.skipUnless(HAVE_ML, "numpy/torch not installed")
class TestWizardHeadless(unittest.TestCase):
    def test_wizard_demo_flow(self):
        try:
            import pygame  # noqa: F401
        except ImportError:
            self.skipTest("pygame not installed")
        import tempfile
        import threading
        import time
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame
        from game.calibrate import run_wizard
        from gnn.muse import MuseClient

        if not pygame.get_init():
            pygame.init()
        with tempfile.TemporaryDirectory() as tmp:
            holder = {}
            worker = threading.Thread(
                target=lambda: holder.update(
                    result=run_wizard(
                        source="demo", out_dir=tmp,
                        cues=[("LEFT", 0.15), ("RIGHT", 0.15)],
                        poll_s=0.02,
                        muse_client=MuseClient(api_key=""))),
                daemon=True)
            worker.start()
            time.sleep(0.4)
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN,
                                                 key=pygame.K_SPACE))
            deadline = time.time() + 20
            adapter = os.path.join(tmp, "adapter.pt")
            while not os.path.exists(adapter) and time.time() < deadline:
                time.sleep(0.1)
            self.assertTrue(os.path.exists(adapter),
                            "wizard never finished adapting")
            time.sleep(0.3)
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN,
                                                 key=pygame.K_SPACE))
            worker.join(timeout=20)
            self.assertFalse(worker.is_alive(), "wizard thread hung")
            result = holder.get("result")
            self.assertIsNotNone(result)
            self.assertGreaterEqual(result["summary"]["n_frames"], 2)
        pygame.quit()


@unittest.skipUnless(HAVE_ML, "numpy/torch not installed")
class TestLiveBridgeMeta(unittest.TestCase):
    def test_step_features_classic(self):
        import torch.nn as nn
        from gnn.live import BrainBridge

        class Fixed(nn.Module):
            def forward(self, x):
                return torch.tensor([[3.0, 0, 0, 0, 0]] * len(x))

        bridge = BrainBridge(Fixed(), confidence=0.0, cooldown_s=0.0, ema=0.0)
        cmd, probs = bridge.step_features(np.zeros((8, 12)))
        self.assertEqual(cmd, "UP")
        self.assertEqual(len(probs), 5)

    def test_muse_referee_gray_zone(self):
        import torch.nn as nn
        from gnn.live import BrainBridge

        class Uncertain(nn.Module):
            def forward(self, x):  # top prob ~0.65: gray under conf=0.9
                return torch.tensor([[2.0, 0, 0, 0, 0]] * len(x))

        class FakeMuse:
            def __init__(self):
                self.n = 0

            def interpret_command(self, context):
                self.n += 1
                self.ctx = context
                return {"command": "RIGHT", "reason": "steady drift right",
                        "offline": False}

        fake = FakeMuse()
        bridge = BrainBridge(Uncertain(), confidence=0.9, cooldown_s=0.0,
                             ema=0.0, muse_client=fake, muse_cooldown_s=0.0)
        cmd, _probs = bridge.step_features(np.zeros((8, 12)))
        self.assertEqual(cmd, "RIGHT")
        self.assertEqual(fake.n, 1)
        self.assertIn("recent_raw", fake.ctx)

    def test_spatial_step_with_dynamic_edges(self):
        from gnn.graph import SpatialGNN, as_spatial
        from gnn.live import BrainBridge
        from gnn.model import EEGGraphNet
        torch.manual_seed(0)
        bridge = BrainBridge(as_spatial(EEGGraphNet()), confidence=0.0,
                             cooldown_s=0.0)
        self.assertIsInstance(bridge.model, SpatialGNN)
        cmd, probs = bridge.step(np.random.RandomState(0).randn(8, 500))
        self.assertIn(cmd, ("UP", "DOWN", "LEFT", "RIGHT", "INTERACT"))
        self.assertEqual(len(probs), 5)


if __name__ == "__main__":
    unittest.main()
