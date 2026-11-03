"""Play the game with brainwaves: live GNN inference -> game commands.

Reads 2 s EEG windows every 0.25 s, applies bandpass + notch + CAR spatial
filtering, extracts graph node features, classifies with the trained GNN
(EMA smoothing + confidence gate + cooldown), and steers the game by
writing ``game/ai_input.json``, which the running game merges with the
keyboard — both work simultaneously.

Run in two terminals (workspace root)::

    .venv/Scripts/python -m game.main
    .venv/Scripts/python -m gnn.live --source live --device "BA MINI 001" --model gnn/checkpoints/model.pt

No headset? Replay recordings instead::

    .venv/Scripts/python -m gnn.live --source replay --model ... --steps 40

Ctrl-C stops the bridge and resets the game input to neutral.
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
                 cooldown_s=COOLDOWN_S, step_s=STEP_S, mains=50):
        self.model = model
        self.classes = classes or CLASSES
        self.ema = ema
        self.confidence = confidence
        self.cooldown_s = cooldown_s
        self.step_s = step_s
        self.mains = mains
        self.smooth = np.ones(len(self.classes)) / len(self.classes)
        self.cooldown_left = 0.0
        self.prev_command = None

    def step(self, window):
        """Classify one (C, T) raw window -> (command|None, probs)."""
        import torch
        clean = preprocess(window, mains=self.mains)
        feats = extract_node_features(clean, mains=self.mains)
        with torch.no_grad():
            logits = self.model(torch.tensor(feats[None].astype(np.float32)))
            probs = torch.softmax(logits, 1)[0].numpy()
        self.smooth = self.ema * self.smooth + (1 - self.ema) * probs
        self.cooldown_left = max(0.0, self.cooldown_left - self.step_s)
        best = int(np.argmax(self.smooth))
        if (self.smooth[best] < self.confidence
                or self.cooldown_left > 0):
            self.prev_command = None
            return None, probs
        command = self.classes[best]
        self.cooldown_left = self.cooldown_s
        self.prev_command = command
        return command, probs

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
    ap.add_argument("--device", default=None)
    ap.add_argument("--mains", type=int, default=50)
    ap.add_argument("--steps", type=int, default=None,
                    help="stop after N windows (replay smoke)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    from gnn.source import ReplaySource, LiveSource
    model, cfg = load_checkpoint(args.model)
    print(f"Model {args.model} (val_acc={cfg['best_val_acc']}, "
          f"ch={cfg['channels']})")
    if cfg["channels"] != CHANNELS:
        print(f"WARNING: checkpoint channels {cfg['channels']} != {CHANNELS}")

    if args.source == "replay":
        source = ReplaySource(args.data, expected_channels=CHANNELS)
    else:
        if not args.device:
            raise SystemExit("--device required for live source")
        source = LiveSource(args.device)

    bridge = BrainBridge(model, cfg["classes"], mains=args.mains)
    window_n = int(WINDOW_S * FS)
    print(f"Bridge live -> {GAME_INPUT} "
          f"(window={WINDOW_S}s step={STEP_S}s gate={CONFIDENCE})")
    print("Ctrl-C to stop (game input resets to neutral).")
    steps = 0
    try:
        with source:
            while args.steps is None or steps < args.steps:
                t0 = time.time()
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
