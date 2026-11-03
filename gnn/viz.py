"""Real-time EEG + feature visualizer (BrainAccess Mini).

Shows scrolling raw traces, per-channel band powers, and the trained
model's live prediction as data streams in::

    .venv/Scripts/python -m gnn.viz --source replay
    .venv/Scripts/python -m gnn.viz --source replay --model gnn/checkpoints/model.pt
    .venv/Scripts/python -m gnn.viz --source live --device "BA MINI 001"

``replay`` loops EEG_recordings (no hardware). ``live`` needs the
BrainAccess SDK + mne + the headset (Windows). Esc/Q quits.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gnn.model import (CHANNELS, CLASSES, FS, BANDS, CONFIDENCE,
                       preprocess, extract_node_features, load_checkpoint)

TRACE_S = 5.0
WINDOW_S = 2.0


def band_logpowers(preprocessed, fs=FS):
    """(C, T) preprocessed -> (C, n_bands) mean log10 power per band."""
    from scipy.signal import welch
    f, psd = welch(preprocessed, fs, nperseg=min(256, preprocessed.shape[-1]),
                   axis=-1)
    out = np.zeros((preprocessed.shape[0], len(BANDS)))
    for j, (lo, hi) in enumerate(BANDS):
        idx = (f >= lo) & (f < hi)
        if idx.any():
            out[:, j] = np.log10(psd[:, idx].mean(axis=1) + 1e-12)
    return out


class Dashboard:
    def __init__(self, source, model=None, classes=None, size=(1280, 720)):
        import pygame
        self.pygame = pygame
        pygame.init()
        pygame.display.set_caption("EEG live — traces + features")
        self.screen = pygame.display.set_mode(size)
        self.W, self.H = size
        self.font = pygame.font.SysFont("consolas", 16)
        self.big = pygame.font.SysFont("consolas", 30, bold=True)
        self.source = source
        self.model = model
        self.classes = classes or CLASSES
        self.scales = {c: 50.0 for c in CHANNELS}
        self.probs = np.ones(len(self.classes)) / len(self.classes)
        self.smooth = self.probs.copy()
        self.fps = 0.0

    def poll(self):
        window = self.source.read_window(seconds=TRACE_S, channels=CHANNELS)
        if window is None:
            return None, None, None
        clean = preprocess(window)
        powers = band_logpowers(clean)
        epoch = clean[:, -int(WINDOW_S * FS):]
        if self.model is not None and epoch.shape[1] >= 256:
            import torch
            feats = extract_node_features(epoch)
            with torch.no_grad():
                logits = self.model(
                    torch.tensor(feats[None].astype(np.float32)))
                self.probs = torch.softmax(logits, 1)[0].numpy()
            self.smooth = 0.7 * self.smooth + 0.3 * self.probs
        return window, powers, epoch

    def draw_traces(self, window):
        pg = self.pygame
        area = pg.Rect(10, 50, int(self.W * 0.62), self.H - 110)
        pg.draw.rect(self.screen, (10, 14, 24), area, border_radius=8)
        n_ch = window.shape[0]
        row_h = area.h // n_ch
        for i, ch in enumerate(CHANNELS[:n_ch]):
            trace = window[i]
            target = float(np.percentile(np.abs(trace - np.median(trace)),
                                         95)) + 1e-6
            s = self.scales
            s[ch] = 0.95 * s[ch] + 0.05 * target
            mid = area.y + i * row_h + row_h // 2
            amp = row_h * 0.42 / s[ch]
            xs = np.linspace(area.x + 60, area.right - 10, len(trace))
            ys = mid - (trace - np.median(trace)) * amp
            ys = np.clip(ys, area.y + i * row_h + 2,
                         area.y + (i + 1) * row_h - 2)
            label = self.font.render(ch, True, (140, 200, 255))
            self.screen.blit(label, (area.x + 12, mid - 9))
            if len(xs) > 1:
                pg.draw.lines(self.screen, (110, 230, 170),
                              False, list(zip(xs, ys)), 1)

    def draw_powers(self, powers):
        pg = self.pygame
        x0 = int(self.W * 0.62) + 24
        w = self.W - x0 - 14
        panel = pg.Rect(x0, 50, w, int((self.H - 110) * 0.52))
        pg.draw.rect(self.screen, (10, 14, 24), panel, border_radius=8)
        title = self.font.render("band log-power (uV^2/Hz)", True,
                                 (255, 225, 160))
        self.screen.blit(title, (panel.x + 12, panel.y + 8))
        bands = ["d", "a", "b1", "b2", "g"]
        cols = [(150, 150, 220), (120, 220, 150), (255, 200, 120),
                (255, 150, 120), (200, 140, 255)]
        top, bottom = panel.y + 34, panel.bottom - 24
        lo, hi = powers.min(), powers.max()
        span = max(1e-6, hi - lo)
        bw = (panel.w - 24) / (len(bands) * 8 + 9)
        for j in range(len(bands)):
            for i in range(min(8, powers.shape[0])):
                v = (powers[i, j] - lo) / span
                h = v * (bottom - top)
                bx = panel.x + 12 + (j * 8 + i) * (bw + 1)
                pg.draw.rect(self.screen, cols[j],
                             (bx, bottom - h, bw, h))
            lab = self.font.render(bands[j], True, (180, 190, 205))
            cx = panel.x + 12 + (j * 8 + 3.5) * (bw + 1)
            self.screen.blit(lab, (cx, bottom + 4))

    def draw_prediction(self):
        pg = self.pygame
        x0 = int(self.W * 0.62) + 24
        w = self.W - x0 - 14
        top = 50 + int((self.H - 110) * 0.52) + 12
        panel = pg.Rect(x0, top, w, self.H - 60 - top)
        pg.draw.rect(self.screen, (10, 14, 24), panel, border_radius=8)
        best = int(np.argmax(self.smooth))
        gated = self.smooth[best] >= CONFIDENCE
        cmd = self.classes[best] if (gated and self.model is not None) else "—"
        colour = (140, 255, 170) if gated else (120, 130, 150)
        txt = self.big.render(f"{cmd}  {self.smooth[best]:.2f}", True, colour)
        self.screen.blit(txt, (panel.x + 12, panel.y + 8))
        y = panel.y + 52
        for i, cls in enumerate(self.classes):
            bar_w = (panel.w - 130) * float(self.smooth[i])
            lab = self.font.render(f"{cls:9s}", True, (200, 210, 225))
            self.screen.blit(lab, (panel.x + 12, y))
            pg.draw.rect(self.screen, (50, 60, 90),
                         (panel.x + 110, y + 2, panel.w - 130, 14))
            pg.draw.rect(self.screen,
                         (140, 255, 170) if i == best and gated
                         else (110, 150, 220),
                         (panel.x + 110, y + 2, bar_w, 14))
            y += 24

    def run(self, frames=None):
        pg = self.pygame
        clock = pg.time.Clock()
        n = 0
        running = True
        while running and (frames is None or n < frames):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN and event.key in (
                        pg.K_ESCAPE, pg.K_q):
                    running = False
            t0 = time.perf_counter()
            window, powers, _epoch = self.poll()
            self.screen.fill((22, 27, 42))
            head = self.big.render("EEG live", True, (235, 240, 255))
            self.screen.blit(head, (14, 8))
            sub = self.font.render(
                f"{type(self.source).__name__}  {self.fps:4.0f} fps",
                True, (150, 160, 180))
            self.screen.blit(sub, (170, 18))
            if window is not None:
                self.draw_traces(window)
                self.draw_powers(powers)
            else:
                wait = self.font.render("buffering...", True, (200, 200, 120))
                self.screen.blit(wait, (20, 80))
            self.draw_prediction()
            pg.display.flip()
            self.fps = 0.9 * self.fps + 0.1 / max(1e-3,
                                                 time.perf_counter() - t0)
            clock.tick(20)
            n += 1
        pg.quit()


def main(argv=None):
    ap = argparse.ArgumentParser(description="EEG realtime visualizer")
    ap.add_argument("--source", choices=("replay", "live"), default="replay")
    ap.add_argument("--data", default="EEG_recordings")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--frames", type=int, default=None,
                    help="quit after N frames (headless smoke)")
    args = ap.parse_args(argv)
    from gnn.source import ReplaySource, LiveSource
    model, classes = None, CLASSES
    if args.model:
        model, cfg = load_checkpoint(args.model)
        classes = cfg["classes"]
        print(f"Loaded {args.model} (val_acc={cfg['best_val_acc']})")
    if args.source == "replay":
        source = ReplaySource(args.data, expected_channels=CHANNELS)
    else:
        if not args.device:
            raise SystemExit("--device required for live source")
        source = LiveSource(args.device)
    with source:
        Dashboard(source, model, classes).run(frames=args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
