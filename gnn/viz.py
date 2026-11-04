"""Real-time EEG board: traces, features, spectra, and headset status.

A comprehensive live rendering of the BrainAccess Mini stream::

    .venv/Scripts/python -m gnn.viz --source replay
    .venv/Scripts/python -m gnn.viz --source replay --model gnn/checkpoints/model.pt
    .venv/Scripts/python -m gnn.viz --source live --device "BA MINI 001"

``replay`` loops EEG_recordings (no hardware). ``live`` needs the
BrainAccess SDK + mne + the headset (Windows).

Panels: raw / CAR-spatial-filtered traces (C toggles), per-channel band
powers, the 12 GNN node features per channel as a heatmap, the live
prediction, a log spectrum of the selected channel (keys 1-8), contact
quality, battery, and accelerometer. Esc/Q quits.
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
FEATURE_NAMES = ["dP", "dE", "aP", "aE", "b1P", "b1E", "b2P", "b2E",
                 "gP", "gE", "Hm", "Hc"]
BAND_SHORT = ["d", "a", "b1", "b2", "g"]
BAND_COLOURS = [(150, 150, 220), (120, 220, 150), (255, 200, 120),
                (255, 150, 120), (200, 140, 255)]


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


def channel_psd(preprocessed, fs=FS):
    """(C, T) -> (f, psd(C, F)) Welch spectrum for the PSD panel."""
    from scipy.signal import welch
    f, psd = welch(preprocessed, fs, nperseg=min(256, preprocessed.shape[-1]),
                   axis=-1)
    return np.asarray(f), np.asarray(psd)


def _heat_colour(value, span=2.0):
    """Diverging blue->black->red for standardized features."""
    v = max(-1.0, min(1.0, value / span))
    if v >= 0:
        return (40 + int(215 * v), 40, 50)
    return (40, 60 + int(120 * -v), 140 + int(115 * -v))


class Dashboard:
    def __init__(self, source, model=None, classes=None, size=(1440, 900)):
        import pygame
        self.pygame = pygame
        pygame.init()
        pygame.display.set_caption("EEG board — live traces + features")
        self.screen = pygame.display.set_mode(size)
        self.W, self.H = size
        self.font = pygame.font.SysFont("consolas", 15)
        self.big = pygame.font.SysFont("consolas", 26, bold=True)
        self.source = source
        self.model = model
        self.classes = classes or CLASSES
        self.show_car = False
        self.selected = 0
        self.scales = {c: 50.0 for c in CHANNELS}
        self.contact_max = {c: 1.0 for c in CHANNELS}
        self.probs = np.ones(len(self.classes)) / len(self.classes)
        self.smooth = self.probs.copy()
        self.fps = 0.0
        self.battery = None
        self.accel = None
        self.contacts = None

    # ------------------------------------------------------------ data
    def poll(self):
        window = self.source.read_window(seconds=TRACE_S, channels=CHANNELS)
        for attr, reader in (("battery", "read_battery"),
                             ("accel", "read_accel"),
                             ("contacts", "read_contacts")):
            try:
                fn = getattr(self.source, reader, None)
                if fn is not None:
                    value = fn() if attr != "contacts" else fn(CHANNELS)
                    if value is not None:
                        setattr(self, attr, value)
            except Exception:
                pass
        if window is None:
            return None
        clean = preprocess(window)
        epoch = clean[:, -int(WINDOW_S * FS):]
        feats = None
        if epoch.shape[1] >= 256:
            feats = extract_node_features(epoch)
            if self.model is not None:
                import torch
                with torch.no_grad():
                    logits = self.model(
                        torch.tensor(feats[None].astype(np.float32)))
                    self.probs = torch.softmax(logits, 1)[0].numpy()
                self.smooth = 0.7 * self.smooth + 0.3 * self.probs
        f, psd = channel_psd(clean)
        return {"raw": window, "clean": clean,
                "powers": band_logpowers(clean), "feats": feats,
                "f": f, "psd": psd}

    # ------------------------------------------------------------ panels
    def _panel(self, rect, title):
        pg = self.pygame
        pg.draw.rect(self.screen, (10, 14, 24), rect, border_radius=8)
        label = self.font.render(title, True, (255, 225, 160))
        self.screen.blit(label, (rect.x + 12, rect.y + 6))
        return pg.Rect(rect.x + 8, rect.y + 28, rect.w - 16, rect.h - 36)

    def draw_header(self):
        header = f"EEG board   {type(self.source).__name__}   {self.fps:4.0f} fps"
        self.screen.blit(self.big.render(header, True, (235, 240, 255)),
                         (14, 6))
        batt = f"{self.battery}%" if self.battery is not None else "n/a"
        txt = self.font.render(f"battery {batt}", True, (150, 255, 170)
                               if self.battery is not None else (120, 130, 150))
        self.screen.blit(txt, (self.W - txt.get_width() - 16, 14))

    def draw_traces(self, rect, data):
        pg = self.pygame
        inner = self._panel(
            rect, "CAR spatial-filtered" if self.show_car else "raw (uV)")
        traces = data["clean"] if self.show_car else data["raw"]
        n_ch = traces.shape[0]
        row_h = inner.h // n_ch
        for i, ch in enumerate(CHANNELS[:n_ch]):
            trace = traces[i]
            target = float(np.percentile(np.abs(trace - np.median(trace)),
                                         95)) + 1e-6
            self.scales[ch] = 0.95 * self.scales[ch] + 0.05 * target
            mid = inner.y + i * row_h + row_h // 2
            amp = row_h * 0.42 / self.scales[ch]
            if i == self.selected:
                pg.draw.rect(self.screen, (30, 42, 66),
                             (inner.x, inner.y + i * row_h + 1,
                              inner.w, row_h - 2), border_radius=4)
            xs = np.linspace(inner.x + 52, inner.right - 8, len(trace))
            ys = mid - (trace - np.median(trace)) * amp
            ys = np.clip(ys, inner.y + i * row_h + 2,
                         inner.y + (i + 1) * row_h - 2)
            label = self.font.render(ch, True, (140, 200, 255)
                                     if i != self.selected else (255, 240, 180))
            self.screen.blit(label, (inner.x + 8, mid - 8))
            if len(xs) > 1:
                pg.draw.lines(self.screen, (110, 230, 170), False,
                              list(zip(xs, ys)), 1)

    def draw_powers(self, rect, powers):
        inner = self._panel(rect, "band log-power (uV^2/Hz)")
        lo, hi = powers.min(), powers.max()
        span = max(1e-6, hi - lo)
        bw = (inner.w - 8) / (len(BAND_SHORT) * 8 + 7)
        for j in range(len(BAND_SHORT)):
            for i in range(min(8, powers.shape[0])):
                v = (powers[i, j] - lo) / span
                h = v * (inner.h - 20)
                bx = inner.x + 4 + (j * 8 + i) * (bw + 1)
                self.pygame.draw.rect(
                    self.screen, BAND_COLOURS[j],
                    (bx, inner.bottom - 20 - h, bw, h))
            lab = self.font.render(BAND_SHORT[j], True, (180, 190, 205))
            cx = inner.x + 4 + (j * 8 + 3.5) * (bw + 1)
            self.screen.blit(lab, (cx, inner.bottom - 18))

    def draw_features(self, rect, feats):
        inner = self._panel(rect, "GNN node features (8 ch x 12, z-scored)")
        if feats is None:
            wait = self.font.render("buffering 2 s window...", True,
                                    (200, 200, 120))
            self.screen.blit(wait, (inner.x + 8, inner.y + 8))
            return
        n_ch, n_f = feats.shape
        cw = (inner.w - 44) / n_f
        rh = (inner.h - 20) / n_ch
        for j, name in enumerate(FEATURE_NAMES[:n_f]):
            lab = self.font.render(name, True, (150, 160, 180))
            self.screen.blit(lab, (inner.x + 44 + j * cw + 1, inner.y + 1))
        for i in range(n_ch):
            ch = self.font.render(CHANNELS[i], True, (140, 200, 255))
            self.screen.blit(ch, (inner.x + 4, inner.y + 20 + i * rh + 1))
            for j in range(n_f):
                self.pygame.draw.rect(
                    self.screen, _heat_colour(feats[i, j]),
                    (inner.x + 44 + j * cw + 1, inner.y + 20 + i * rh + 1,
                     max(1, cw - 2), max(1, rh - 2)))

    def draw_prediction(self, rect):
        inner = self._panel(rect, "prediction")
        best = int(np.argmax(self.smooth))
        gated = self.smooth[best] >= CONFIDENCE
        cmd = self.classes[best] if (gated and self.model is not None) else "-"
        colour = (140, 255, 170) if gated else (120, 130, 150)
        txt = self.big.render(f"{cmd}  {self.smooth[best]:.2f}", True, colour)
        self.screen.blit(txt, (inner.x + 4, inner.y + 2))
        y = inner.y + 44
        for i, cls in enumerate(self.classes):
            bar_w = (inner.w - 120) * float(self.smooth[i])
            lab = self.font.render(f"{cls:9s}", True, (200, 210, 225))
            self.screen.blit(lab, (inner.x + 4, y))
            self.pygame.draw.rect(self.screen, (50, 60, 90),
                                  (inner.x + 100, y + 2, inner.w - 120, 13))
            self.pygame.draw.rect(
                self.screen,
                (140, 255, 170) if i == best and gated else (110, 150, 220),
                (inner.x + 100, y + 2, bar_w, 13))
            y += 21

    def draw_psd(self, rect, f, psd):
        pg = self.pygame
        ch = CHANNELS[self.selected % psd.shape[0]]
        inner = self._panel(rect, f"spectrum — {ch} (log)")
        curve = np.log10(psd[self.selected % psd.shape[0]] + 1e-12)
        mask = (f >= 1) & (f <= 45)
        f, curve = f[mask], curve[mask]
        lo, hi = curve.min(), curve.max()
        span = max(1e-6, hi - lo)
        shade = pg.Surface((inner.w, inner.h), pg.SRCALPHA)
        for (blo, bhi), colour in zip(BANDS, BAND_COLOURS):
            x0 = inner.w * (blo - 1) / 44
            x1 = inner.w * (bhi - 1) / 44
            shade.fill((*colour, 48), (x0, 0, x1 - x0, inner.h))
        self.screen.blit(shade, (inner.x, inner.y))
        xs = inner.x + (f - 1) / 44 * inner.w
        ys = inner.bottom - (curve - lo) / span * (inner.h - 16) - 4
        if len(xs) > 1:
            pg.draw.lines(self.screen, (140, 220, 255), False,
                          list(zip(xs, ys)), 2)
        fl = self.font.render("1 Hz", True, (150, 160, 180))
        fh = self.font.render("45 Hz", True, (150, 160, 180))
        self.screen.blit(fl, (inner.x + 2, inner.bottom - 16))
        self.screen.blit(fh, (inner.right - 44, inner.bottom - 16))

    def draw_contacts(self, rect):
        inner = self._panel(rect, "contact quality (raw values)")
        n = len(CHANNELS)
        cw = inner.w / n
        for i, ch in enumerate(CHANNELS):
            value = (self.contacts or {}).get(ch)
            if value is None:
                text, frac = "n/a", 0.0
            else:
                cmax = self.contact_max
                cmax[ch] = max(value, 0.99 * cmax[ch], 1e-9)
                text, frac = f"{value:.1f}", min(1.0, value / cmax[ch])
            cx = inner.x + i * cw
            lab = self.font.render(ch, True, (140, 200, 255))
            self.screen.blit(lab, (cx + 4, inner.y + 2))
            bar_h = (inner.h - 52) * frac
            self.pygame.draw.rect(self.screen, (50, 60, 90),
                                  (cx + 6, inner.y + 22, cw - 12,
                                   inner.h - 52))
            self.pygame.draw.rect(self.screen, (120, 200, 255),
                                  (cx + 6, inner.y + 22 + inner.h - 52 - bar_h,
                                   cw - 12, bar_h))
            val = self.font.render(text, True, (200, 210, 225))
            self.screen.blit(val, (cx + 4, inner.bottom - 20))

    def draw_aux(self, rect):
        inner = self._panel(rect, "headset")
        lines = []
        if self.accel is None:
            lines.append("accel: n/a")
        else:
            x, y, z = self.accel
            lines.append(f"accel: {x:+.2f} {y:+.2f} {z:+.2f} m/s2")
        lines.append(f"battery: {self.battery}%"
                     if self.battery is not None else "battery: n/a")
        lines += ["", "keys:", "C raw/CAR", "1-8 channel", "Q quit"]
        y = inner.y + 4
        for line in lines:
            txt = self.font.render(line, True, (200, 210, 225))
            self.screen.blit(txt, (inner.x + 6, y))
            y += 19

    # ------------------------------------------------------------ loop
    def handle_keys(self):
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
            if event.type == pg.KEYDOWN:
                if event.key in (pg.K_ESCAPE, pg.K_q):
                    return False
                if event.key == pg.K_c:
                    self.show_car = not self.show_car
                if pg.K_1 <= event.key <= pg.K_8:
                    self.selected = event.key - pg.K_1
        return True

    def run(self, frames=None):
        pg = self.pygame
        clock = pg.time.Clock()
        n = 0
        running = True
        mid = int(self.W * 0.60)
        top, bottom = 46, self.H - 236
        while running and (frames is None or n < frames):
            running = self.handle_keys()
            t0 = time.perf_counter()
            data = self.poll()
            self.screen.fill((22, 27, 42))
            self.draw_header()
            if data is None:
                wait = self.font.render("buffering...", True, (200, 200, 120))
                self.screen.blit(wait, (20, 80))
            else:
                self.draw_traces(pg.Rect(10, top, mid - 18, bottom - top),
                                 data)
                rx, rw = mid + 6, self.W - mid - 16
                rh = (bottom - top - 16) // 3
                self.draw_powers(pg.Rect(rx, top, rw, rh), data["powers"])
                self.draw_features(pg.Rect(rx, top + rh + 8, rw, rh),
                                   data["feats"])
                self.draw_prediction(pg.Rect(rx, top + 2 * rh + 16, rw, rh))
                brow = self.H - 226
                bh = 216
                self.draw_psd(pg.Rect(10, brow, 560, bh),
                              data["f"], data["psd"])
                self.draw_contacts(pg.Rect(580, brow, 480, bh))
                self.draw_aux(pg.Rect(1070, brow, self.W - 1080, bh))
            pg.display.flip()
            self.fps = 0.9 * self.fps + 0.1 / max(1e-3,
                                                 time.perf_counter() - t0)
            clock.tick(20)
            n += 1
        pg.quit()


def main(argv=None):
    ap = argparse.ArgumentParser(description="EEG realtime board")
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
