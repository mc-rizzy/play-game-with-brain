"""15-second Calibration Wizard: the game learns each new brain, live.

Run inside the game (press C) or standalone::

    python -m game.calibrate --source demo
    python -m game.calibrate --source replay --model gnn/checkpoints/model.pt
    python -m game.calibrate --source live --device "BA MINI 048" \\
        --model gnn/checkpoints/model.pt

Flow: 5 cues x 3 s (LEFT/RIGHT/UP/DOWN/INTERACT) -> collect labeled GNN
frames -> MAML-style head adaptation (``gnn/adapt.py``) -> Muse analyzes
the fresh EEG summary and tunes the live gates (``gnn/muse.py``) -> save
``adapter.pt`` + ``calibration_report.json`` for ``gnn/live.py --adapter``.

Sources: ``demo`` (synthetic, no hardware — always works), ``replay``
(loops EEG_recordings), ``live`` (BrainAccess headset via the SDK).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.calibration import CUES_15S

CUE_GLYPH = {"LEFT": "<", "RIGHT": ">", "UP": "^", "DOWN": "v",
             "INTERACT": "*"}
CUE_HINT = {"LEFT": "imagine moving your LEFT hand",
            "RIGHT": "imagine moving your RIGHT hand",
            "UP": "imagine both hands lifting / JUMP",
            "DOWN": "imagine pressing down / PLACE",
            "INTERACT": "imagine grasping / ATTACK"}


def _window_reader(source, channels, window_s=2.0):
    """Wrap a gnn.source stream as a wizard read_frame callback."""
    from gnn.model import FS, preprocess, extract_node_features

    want = int(window_s * FS)

    def read_frame():
        window = source.read_window(seconds=window_s, channels=channels)
        if window is None or window.shape[1] < want:
            return None
        clean = preprocess(window[:, -want:])
        feats = extract_node_features(clean)
        return feats, clean

    return read_frame


class _Reader:
    """Context manager: source lifecycle + read_frame + set_cue hook."""

    def __init__(self, source, device=None, data="EEG_recordings",
                 channels=None):
        self.kind = source
        self.device = device
        self.data = data
        self.channels = channels
        self._source = None
        self._demo = None

    def __enter__(self):
        from gnn.model import CHANNELS
        channels = self.channels or CHANNELS
        if self.kind == "demo":
            from gnn.calibration import synthetic_read_frame
            self._demo = synthetic_read_frame(n_ch=len(channels))
            return self._demo
        from gnn.source import ReplaySource, LiveSource
        if self.kind == "replay":
            self._source = ReplaySource(self.data,
                                        expected_channels=channels)
        elif self.kind == "live":
            if not self.device:
                raise SystemExit("--device required for live calibration")
            self._source = LiveSource(self.device)
        else:
            raise SystemExit(f"unknown source {self.kind!r}")
        self._source.__enter__()
        return _window_reader(self._source, channels)

    def set_cue(self, read_frame, cue):
        hook = getattr(read_frame, "set_cue", None)
        if hook is not None:
            hook(cue)

    def __exit__(self, *exc):
        if self._source is not None:
            self._source.__exit__(*exc)
            self._source = None


def _wrap(font, text, width):
    words, lines, line = str(text).split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if font.size(trial)[0] <= width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines or [""]


def run_wizard(screen=None, source="demo", model_path=None, device=None,
               data="EEG_recordings", out_dir="gnn/checkpoints",
               cues=None, muse_client=None, poll_s=0.25):
    """Take over ``screen`` for the wizard; returns the result dict.

    Returns None when the player skips (ESC). Owns the display only when
    ``screen`` is None (standalone mode).
    """
    import pygame

    from gnn import adapt as adapt_mod
    from gnn.calibration import (LABEL_TO_INDEX, summarize_support,
                                 silhouette_separability)
    from gnn.graph import as_spatial, functional_adj
    from gnn.model import CHANNELS, CLASSES
    from gnn.muse import MuseClient

    cues = list(cues or CUES_15S)
    own_display = screen is None
    if own_display:
        pygame.init()
        pygame.display.set_caption("Calibration Wizard — 15 seconds")
        screen = pygame.display.set_mode((900, 600))
    clock = pygame.time.Clock()
    W, H = screen.get_size()
    font = pygame.font.Font(None, 30)
    big = pygame.font.Font(None, 120)
    mid = pygame.font.Font(None, 44)
    small = pygame.font.Font(None, 24)

    def draw_center(lines, y0=60, dye=(235, 240, 255), font_=font):
        y = y0
        for line in lines:
            img = font_.render(line, True, dye)
            screen.blit(img, (W // 2 - img.get_width() // 2, y))
            y += font_.get_linesize() + 4
        return y

    def events():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return "skip"
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return "space"
        return None

    # ---- intro ----
    while True:
        signal = events()
        if signal in ("quit", "skip"):
            return None
        if signal == "space":
            break
        screen.fill((22, 27, 42))
        draw_center(["BRAIN CALIBRATION  -  15 SECONDS",
                     "Imagine each command when it appears.",
                     "Keep still, eyes on the cue."], 150,
                    font_=mid)
        y = draw_center(
            [f"{CUE_GLYPH[c]}  {c}  ({s:.0f}s)" for c, s in cues], 300,
            (150, 220, 255))
        draw_center(["SPACE to start   -   ESC to skip"], y + 20,
                    (160, 170, 190))
        pygame.display.flip()
        clock.tick(30)

    # ---- model ----
    if model_path:
        from gnn.model import load_checkpoint
        base, _cfg = load_checkpoint(model_path)
        model = as_spatial(base)
    else:
        from gnn.graph import SpatialGNN
        model = SpatialGNN(channels=CHANNELS, n_classes=len(CLASSES))
        model.eval()

    # ---- cue loop ----
    X, y, A = [], [], []
    reader = _Reader(source, device=device, data=data, channels=CHANNELS)
    total = sum(s for _, s in cues)
    try:
        with reader as read_frame:
            elapsed = 0.0
            for cue, seconds in cues:
                reader.set_cue(read_frame, cue)
                label = LABEL_TO_INDEX[cue]
                cue_end = time.time() + seconds
                next_poll = 0.0
                n0 = len(X)
                while True:
                    signal = events()
                    if signal in ("quit", "skip"):
                        return None
                    now = time.time()
                    left = cue_end - now
                    if left <= 0:
                        break
                    if now >= next_poll:
                        next_poll = now + poll_s
                        frame = read_frame()
                        if frame is not None:
                            feats, epoch = frame
                            X.append(feats)
                            y.append(label)
                            if epoch is not None:
                                A.append(functional_adj(epoch))
                    screen.fill((22, 27, 42))
                    glyph = big.render(CUE_GLYPH[cue], True, (140, 255, 170))
                    screen.blit(glyph, (W // 2 - glyph.get_width() // 2,
                                        120))
                    draw_center([cue, CUE_HINT[cue]], 260, font_=mid)
                    bar = pygame.Rect(150, 430, W - 300, 26)
                    pygame.draw.rect(screen, (50, 60, 90), bar,
                                     border_radius=6)
                    frac = min(1.0, (elapsed + (seconds - left)) / total)
                    pygame.draw.rect(
                        screen, (140, 255, 170),
                        (bar.x, bar.y, int(bar.w * frac), bar.h),
                        border_radius=6)
                    draw_center(
                        [f"{left:.1f}s  -  {len(X) - n0} frames this cue  "
                         f"({len(X)} total)"], 470, (160, 170, 190),
                        font_=small)
                    pygame.display.flip()
                    clock.tick(60)
                elapsed += seconds
    except SystemExit:
        raise
    except Exception as exc:
        screen.fill((22, 27, 42))
        draw_center(["CALIBRATION SOURCE FAILED", str(exc)[:80],
                     "Press ESC"], 200, (255, 150, 150), font_=mid)
        pygame.display.flip()
        while events() not in ("quit", "skip", "space"):
            clock.tick(30)
        return None

    if len(X) < len(cues):
        return None

    # ---- adapt + Muse ----
    screen.fill((22, 27, 42))
    draw_center(["Adapting to your brain...",
                 "few-shot head tuning  +  Muse coaching"], 220,
                font_=mid)
    pygame.display.flip()

    import numpy as np
    Xa = np.stack(X).astype("float32")
    ya = np.array(y, dtype=np.int64)
    Astack = (np.stack(A).astype("float32") if len(A) == len(X) else None)
    adapted, adapt_info = adapt_mod.adapt_head(model, Xa, ya, A=Astack)
    summary = summarize_support(Xa, ya, CLASSES)
    summary["separability"] = round(
        float(silhouette_separability(Xa, ya)), 3)
    muse = muse_client or MuseClient()
    muse_result = muse.analyze_calibration(summary)
    result = {"summary": summary, "adapt_info": adapt_info,
              "muse": muse_result, "muse_calls": list(muse.calls),
              "thresholds": muse_result["thresholds"]}
    os.makedirs(out_dir, exist_ok=True)
    adapter_path = os.path.join(out_dir, "adapter.pt")
    import json
    report_path = os.path.join(out_dir, "calibration_report.json")
    adapt_mod.save_adapter(
        adapter_path, adapted, classes=CLASSES, channels=CHANNELS,
        thresholds=muse_result["thresholds"],
        calibration={"adapt": adapt_info, "summary": summary},
        muse={"result": muse_result, "calls": list(muse.calls)})
    with open(report_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    result.update(adapter_path=adapter_path, report_path=report_path,
                  adapted_model=adapted)
    print(f"[wizard] {len(Xa)} frames, "
          f"acc {adapt_info['before_acc']:.2f}->{adapt_info['after_acc']:.2f},"
          f" sep={summary['separability']:.2f} -> {adapter_path}")

    # ---- result ----
    while True:
        signal = events()
        if signal in ("quit", "skip", "space"):
            break
        screen.fill((22, 27, 42))
        draw_center(["CALIBRATED  -  this brain is understood"], 40,
                    (140, 255, 170), font_=mid)
        acc = result["adapt_info"]
        draw_center(
            [f"fit {acc['before_acc']:.0%} -> {acc['after_acc']:.0%}  |  "
             f"separability {summary['separability']:.2f}"], 110,
            (255, 240, 180))
        lines = [f"- {tip}" for tip in muse_result["coaching"][:3]]
        lines.append(f"Verdict: {muse_result['verdict']}")
        lines.append(f"Gates: conf={muse_result['thresholds']['confidence']:.2f}"
                     f" cooldown={muse_result['thresholds']['cooldown_s']:.2f}s"
                     f" ema={muse_result['thresholds']['ema']:.2f}")
        call = muse.calls[-1] if muse.calls else {}
        if call.get("ok"):
            lines.append(f"Muse {call.get('model')} live "
                         f"({call.get('latency_s')}s) - report saved")
        else:
            lines.append("Muse offline - heuristic gates applied")
        y = 165
        for line in lines:
            for wrapped in _wrap(font, line, W - 120):
                img = font.render(wrapped, True, (235, 240, 255))
                screen.blit(img, (60, y))
                y += 30
            y += 4
        draw_center(["SPACE to play!"], H - 70, (160, 170, 190))
        pygame.display.flip()
        clock.tick(30)
    if own_display:
        pygame.quit()
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="15-second brain calibration")
    ap.add_argument("--source", choices=("demo", "replay", "live"),
                    default="demo")
    ap.add_argument("--model", default=None,
                    help="base GNN checkpoint (fresh model if omitted)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--data", default="EEG_recordings")
    ap.add_argument("--out", default="gnn/checkpoints")
    ap.add_argument("--cue-s", type=float, default=None,
                    help="override every cue length (demo smoke)")
    args = ap.parse_args(argv)
    cues = ([(c, args.cue_s) for c, _s in CUES_15S] if args.cue_s
            else CUES_15S)
    result = run_wizard(source=args.source, model_path=args.model,
                        device=args.device, data=args.data,
                        out_dir=args.out, cues=cues)
    if result is None:
        print("Calibration skipped.")
        return 1
    print(f"Adapter -> {result['adapter_path']}")
    print(f"Report  -> {result['report_path']}")
    print(f"Muse: {result['muse']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
