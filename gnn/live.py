"""Play the game with brainwaves: live GNN inference -> game commands.

Dual-layer meta-pipeline: the Feature Extraction Layer turns the stream
into rolling PSD node features (``gnn/features.py``), the Spatial GNN
Layer (``gnn/graph.py``) message-passes over spatial + functional
connectivity edges, and the meta-model head was few-shot adapted to this
player by the Calibration Wizard (``--adapter``). EMA smoothing +
confidence gate + cooldown steer the game by writing
``game/ai_input.json``, which the running game merges with the keyboard —
both work simultaneously.

Run in two terminals (workspace root)::

    .venv/Scripts/python -m game.main
    .venv/Scripts/python -m gnn.live --source live --device "BA MINI 001" --model gnn/checkpoints/model.pt --adapter gnn/checkpoints/adapter.pt

No headset? Replay recordings instead::

    .venv/Scripts/python -m gnn.live --source replay --model ... --steps 40

``--muse-fallback`` lets Muse (Meta Model API) referee gray-zone frames
instead of dropping them. Ctrl-C stops the bridge and resets the game
input to neutral.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gnn.model import (CHANNELS, CLASSES, FS, WINDOW_S, STEP_S,
                       EMA, CONFIDENCE, COOLDOWN_S,
                       preprocess, extract_node_features, load_checkpoint)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_INPUT = os.path.join(ROOT, "game", "ai_input.json")


def write_game_input(move=0.0, jump=False, attack=False, place=False):
    payload = {"move": move, "jump": bool(jump), "attack": bool(attack),
               "place": bool(place)}
    tmp = GAME_INPUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, GAME_INPUT)
    return payload


def reset_game_input():
    try:
        if os.path.exists(GAME_INPUT):
            os.remove(GAME_INPUT)
    except OSError:
        pass


class BrainBridge:
    """Sliding-window classify -> gate -> game-input state machine."""

    def __init__(self, model, classes=None, ema=EMA, confidence=CONFIDENCE,
                 cooldown_s=COOLDOWN_S, step_s=STEP_S, mains=50,
                 muse_client=None, muse_cooldown_s=10.0, context_fn=None,
                 dynamic=False):
        self.model = model
        self.classes = classes or CLASSES
        self.ema = ema
        self.confidence = confidence
        self.cooldown_s = cooldown_s
        self.step_s = step_s
        self.mains = mains
        self.muse = muse_client
        self.muse_cooldown_s = muse_cooldown_s
        self.context_fn = context_fn
        # Dynamic (functional-connectivity) edges only help when the head
        # was tuned with them (wizard adapter); the base checkpoint
        # trained on static+learned edges, so inference must match it.
        self.dynamic = bool(dynamic)
        self.smooth = np.ones(len(self.classes)) / len(self.classes)
        self.cooldown_left = 0.0
        self.prev_command = None
        self._recent = []  # last raw top-1 predictions (for the referee)
        self._last_muse_t = 0.0

    def step(self, window):
        """Classify one (C, T) raw window -> (command|None, probs)."""
        clean = preprocess(window, mains=self.mains)
        feats = extract_node_features(clean, mains=self.mains)
        A_func = None
        if self.dynamic and hasattr(self.model, "func_weight"):
            from gnn.graph import functional_adj
            A_func = functional_adj(clean)
        return self.step_features(feats, A_func=A_func)

    def step_features(self, feats, A_func=None):
        """Classify precomputed (N, F) node features -> (command|None, probs)."""
        import torch
        x = torch.tensor(np.asarray(feats, dtype=np.float32)[None])
        with torch.no_grad():
            if (self.dynamic and A_func is not None
                    and hasattr(self.model, "func_weight")):
                a = torch.tensor(np.asarray(A_func, dtype=np.float32)[None])
                logits = self.model(x, A_func=a)
            else:
                logits = self.model(x)
            probs = torch.softmax(logits, 1)[0].numpy()
        self.smooth = self.ema * self.smooth + (1 - self.ema) * probs
        self.cooldown_left = max(0.0, self.cooldown_left - self.step_s)
        self._recent.append(self.classes[int(np.argmax(probs))])
        self._recent = self._recent[-8:]
        best = int(np.argmax(self.smooth))
        if self.cooldown_left > 0:
            self.prev_command = None
            return None, probs
        if self.smooth[best] >= self.confidence:
            command = self.classes[best]
            self.cooldown_left = self.cooldown_s
            self.prev_command = command
            return command, probs
        # Gray zone: throttled Muse referee instead of a blind drop.
        if (self.muse is not None and self.smooth[best] >= 0.30
                and time.time() - self._last_muse_t > self.muse_cooldown_s):
            self._last_muse_t = time.time()
            context = {"smoothed": {c: round(float(p), 3) for c, p in
                                    zip(self.classes, self.smooth)},
                       "recent_raw": list(self._recent)}
            if self.context_fn is not None:
                try:
                    context["game"] = self.context_fn()
                except Exception:
                    pass
            verdict = self.muse.interpret_command(context)
            if verdict["command"] in self.classes:
                self.cooldown_left = self.cooldown_s
                self.prev_command = verdict["command"]
                print(f"[muse] referee -> {verdict['command']} "
                      f"({verdict['reason']})")
                return verdict["command"], probs
        self.prev_command = None
        return None, probs

    def command_to_input(self, command):
        """Map a gated command to a game input payload.

        Moves/jump are held levels; attack/place fire on rising edge only.
        """
        payload = {"move": 0.0, "jump": False, "attack": False,
                   "place": False}
        if command == "LEFT":
            payload["move"] = -1.0
        elif command == "RIGHT":
            payload["move"] = 1.0
        elif command == "UP":
            payload["jump"] = True
        elif command == "DOWN":
            payload["place"] = True
        elif command == "INTERACT":
            payload["attack"] = True
        return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description="Brain-to-game live bridge")
    ap.add_argument("--source", choices=("replay", "live"), default="replay")
    ap.add_argument("--data", default="EEG_recordings")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None,
                    help="wizard adapter.pt: personalized head + gates")
    ap.add_argument("--device", default=None)
    ap.add_argument("--mains", type=int, default=50)
    ap.add_argument("--muse-fallback", action="store_true",
                    help="Muse referees gray-zone frames (Meta Model API)")
    ap.add_argument("--no-interleave", action="store_true",
                    help="replay trials grouped by label instead of cycling")
    ap.add_argument("--steps", type=int, default=None,
                    help="stop after N windows (replay smoke)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    from gnn.source import ReplaySource, LiveSource
    from gnn.graph import as_spatial
    model, cfg = load_checkpoint(args.model)
    model = as_spatial(model)  # same weights + functional-connectivity edges
    print(f"Model {args.model} (val_acc={cfg['best_val_acc']}, "
          f"ch={cfg['channels']})")
    if cfg["channels"] != CHANNELS:
        print(f"WARNING: checkpoint channels {cfg['channels']} != {CHANNELS}")

    ema, confidence, cooldown_s = EMA, CONFIDENCE, COOLDOWN_S
    dynamic = False
    if args.adapter:
        from gnn.adapt import apply_adapter, load_adapter
        adapter = load_adapter(args.adapter)
        apply_adapter(model, adapter)
        gates = adapter.get("thresholds", {})
        ema = float(gates.get("ema", ema))
        confidence = float(gates.get("confidence", confidence))
        cooldown_s = float(gates.get("cooldown_s", cooldown_s))
        dynamic = bool(adapter.get("calibration", {}).get(
            "adapt", {}).get("dynamic_edges", False))
        print(f"Adapter {args.adapter}: personalized head + "
              f"gates(ema={ema:.2f} conf={confidence:.2f} cd={cooldown_s:.2f}s)")
    print(f"Graph edges: static+learned"
          f"{'+functional (adapter-tuned)' if dynamic else ''}")

    muse_client = None
    if args.muse_fallback:
        from gnn.muse import MuseClient
        muse_client = MuseClient()
        print(f"Muse referee: {'live (' + muse_client.model + ')'
                                if muse_client.available else 'OFFLINE (no key)'}")

    fin = model.gc1.lin.in_features
    fast = (fin == 8)
    if fast:
        from gnn.features import RollingPSD
        from gnn.graph import functional_adj
        roller = RollingPSD(n_ch=len(CHANNELS), mains=args.mains)
        print("Feature layer: rolling 0.5 s PSD chunks -> 8-dim node feats")
    else:
        print("Feature layer: 2 s windows -> 12-dim node feats")

    if args.source == "replay":
        source = ReplaySource(args.data, expected_channels=CHANNELS,
                              interleave=not args.no_interleave)
    else:
        if not args.device:
            raise SystemExit("--device required for live source")
        source = LiveSource(args.device)

    bridge = BrainBridge(model, cfg["classes"], ema=ema,
                         confidence=confidence, cooldown_s=cooldown_s,
                         mains=args.mains, muse_client=muse_client,
                         dynamic=dynamic)
    window_n = int(WINDOW_S * FS)
    print(f"Bridge live -> {GAME_INPUT} "
          f"(window={WINDOW_S}s step={STEP_S}s gate={confidence:.2f})")
    print("Ctrl-C to stop (game input resets to neutral).")
    steps = 0
    try:
        with source:
            while args.steps is None or steps < args.steps:
                t0 = time.time()
                if fast:
                    chunk = source.read_window(seconds=STEP_S,
                                               channels=CHANNELS)
                    if chunk is None or chunk.shape[1] < int(STEP_S * FS):
                        time.sleep(0.05)
                        continue
                    roller.push(chunk)
                    feats = roller.frame()
                    if feats is None:
                        time.sleep(0.05)
                        continue
                    command, probs = bridge.step_features(
                        feats, A_func=functional_adj(roller.frame_epoch()))
                else:
                    window = source.read_window(seconds=WINDOW_S,
                                                channels=CHANNELS)
                    if window is None or window.shape[1] < window_n:
                        time.sleep(0.05)
                        continue
                    command, probs = bridge.step(window)
                payload = bridge.command_to_input(command)
                write_game_input(**payload)
                steps += 1
                if not args.quiet:
                    best = CLASSES[int(np.argmax(probs))]
                    print(f"[{steps:4d}] cmd={command or '-':9s} "
                          f"raw={best:9s} p={probs.max():.2f} "
                          f"smooth={bridge.smooth.max():.2f} {payload}")
                dt = time.time() - t0
                time.sleep(max(0.0, STEP_S - dt))
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        reset_game_input()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
