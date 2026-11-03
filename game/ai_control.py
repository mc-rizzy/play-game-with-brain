"""Unified keyboard + AI action interface.

Keyboard always works. An AI/EEG model running in the same workspace can
drive the game simultaneously through any of these routes (first match wins,
merged with OR semantics so both sources work at once):

1. Same process (easiest)::

       from game import ai_control
       ai_control.set_action(move=1.0, jump=True)   # called by the model loop
       ai_control.clear_action()                    # back to neutral

2. Same machine, separate process: write ``ai_input.json`` next to this file
   (or the workspace ``game/ai_input.json``)::

       {"move": -1, "jump": false, "attack": true}

   or with a GNN class label::

       {"command": "LEFT"}

   Valid commands: UP, DOWN, LEFT, RIGHT, INTERACT (see ``gnn/model.py``).

3. Same process with the workspace GNN: use :class:`EEGModelController`,
   which imports ``gnn.model.EEGGraphNet`` lazily and maps its 5-class
   output to game actions with EMA smoothing + a confidence gate.

Merging rule (so keyboard and AI work simultaneously):
- ``move``: keyboard + AI summed, clamped to [-1, 1].
- buttons: logical OR of keyboard and AI.
"""

import json
import os

# GNN class order from gnn/model.py: ['UP','DOWN','LEFT','RIGHT','INTERACT']
GNN_COMMANDS = ("UP", "DOWN", "LEFT", "RIGHT", "INTERACT")

_ACTION_KEYS = ("move", "jump", "attack", "place", "torch", "craft", "slot")


def neutral_action():
    return {
        "move": 0.0, "jump": False, "attack": False, "place": False,
        "torch": False, "craft": False, "slot": None,
    }


# ---------------------------------------------------------------- in-memory
_state = neutral_action()


def set_action(move=0.0, jump=False, attack=False, place=False,
               torch_=False, craft=False, slot=None):
    """Set the AI action (same-process route). Values merge with keyboard."""
    _state["move"] = max(-1.0, min(1.0, float(move)))
    _state["jump"] = bool(jump)
    _state["attack"] = bool(attack)
    _state["place"] = bool(place)
    _state["torch"] = bool(torch_)
    _state["craft"] = bool(craft)
    _state["slot"] = slot
    return dict(_state)


def set_action_from_command(command):
    """Set the AI action from a GNN class label. Returns the action dict."""
    action = map_command_to_action(command)
    _state.update(action)
    return dict(_state)


def clear_action():
    _state.update(neutral_action())
    return dict(_state)


def get_action():
    return dict(_state)


# ------------------------------------------------------------- file polling
def _candidate_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    return (
        os.path.join(here, "ai_input.json"),
        os.path.join(root, "game", "ai_input.json"),
        os.path.join(root, "ai_input.json"),
        os.path.join(os.getcwd(), "ai_input.json"),
        os.path.join(os.getcwd(), "game", "ai_input.json"),
    )


def map_command_to_action(command):
    """Map a GNN class label to a game action dict."""
    action = neutral_action()
    cmd = str(command or "").strip().upper()
    if cmd == "LEFT":
        action["move"] = -1.0
    elif cmd == "RIGHT":
        action["move"] = 1.0
    elif cmd == "UP":
        action["jump"] = True
    elif cmd == "DOWN":
        action["place"] = True
    elif cmd == "INTERACT":
        action["attack"] = True
    return action


def _action_from_mapping(data):
    action = neutral_action()
    if not isinstance(data, dict):
        return action
    if "command" in data:
        action.update(map_command_to_action(data.get("command")))
    for key in ("move", "jump", "attack", "place", "torch", "craft"):
        if key in data:
            action[key] = data[key]
    # Alternate torch spelling used by some scripts.
    if "torch_" in data:
        action["torch"] = data["torch_"]
    if "slot" in data:
        try:
            action["slot"] = int(data["slot"])
        except (TypeError, ValueError):
            action["slot"] = None
    # Alternate held-key style: {"left": true, "right": false, ...}
    move = 0.0
    if data.get("left"):
        move -= 1.0
    if data.get("right"):
        move += 1.0
    if move:
        action["move"] = max(-1.0, min(1.0, move))
    if data.get("up"):
        action["jump"] = True
    if data.get("interact"):
        action["attack"] = True
    try:
        action["move"] = max(-1.0, min(1.0, float(action["move"])))
    except (TypeError, ValueError):
        action["move"] = 0.0
    for key in ("jump", "attack", "place", "torch", "craft"):
        action[key] = bool(action[key])
    return action


def load_file_action():
    """Read ``ai_input.json`` if present, else return a neutral action."""
    for path in _candidate_paths():
        try:
            if not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            action = _action_from_mapping(data)
            action["_source"] = path
            return action
        except (OSError, ValueError):
            continue
    return neutral_action()


def poll_ai_action():
    """Merge in-memory + file AI actions (buttons OR, moves summed)."""
    mem = _state
    action = neutral_action()
    try:
        file_action = load_file_action()
    except Exception:
        file_action = neutral_action()
    try:
        move = float(mem.get("move", 0.0)) + float(file_action.get("move", 0.0))
    except (TypeError, ValueError):
        move = 0.0
    action["move"] = max(-1.0, min(1.0, move))
    for key in ("jump", "attack", "place", "torch", "craft"):
        action[key] = bool(mem.get(key)) or bool(file_action.get(key))
    action["slot"] = mem.get("slot")
    if action["slot"] is None:
        action["slot"] = file_action.get("slot")
    return action


def merge_with_keyboard(keyboard_action, ai_action=None):
    """Merge keyboard + AI so both control the game simultaneously."""
    ai_action = ai_action if ai_action is not None else poll_ai_action()
    merged = neutral_action()
    try:
        kb_move = float(keyboard_action.get("move", 0.0))
    except (TypeError, ValueError):
        kb_move = 0.0
    try:
        ai_move = float(ai_action.get("move", 0.0))
    except (TypeError, ValueError):
        ai_move = 0.0
    merged["move"] = max(-1.0, min(1.0, kb_move + ai_move))
    for key in ("jump", "attack", "place", "torch", "craft"):
        merged[key] = bool(keyboard_action.get(key)) or bool(ai_action.get(key))
    merged["slot"] = keyboard_action.get("slot")
    if merged["slot"] is None:
        merged["slot"] = ai_action.get("slot")
    return merged


# ------------------------------------------------------- workspace GNN hook
class EEGModelController:
    """Optional live hook for the workspace ``gnn/model.py`` classifier.

    Lazily imports torch + the GNN only when constructed, so the game runs
    fine without ML dependencies. Call :meth:`action_from_probs` from the
    model loop each tick; the controller applies EMA smoothing and a
    confidence gate (uncertain frames yield no action).
    """

    def __init__(self, n_classes=5, ema=0.7, confidence=0.60, cooldown=0.5):
        self.n_classes = n_classes
        self.ema = ema
        self.confidence = confidence
        self.cooldown = cooldown
        self._smooth = [1.0 / n_classes] * n_classes
        self._cooldown_left = 0.0
        self._model = None
        self._model_error = None

    @property
    def model(self):
        if self._model is None and self._model_error is None:
            try:
                import sys
                root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if root not in sys.path:
                    sys.path.insert(0, root)
                from gnn.model import EEGGraphNet  # noqa: PLC0415
                self._model = EEGGraphNet(n_classes=self.n_classes)
                self._model.eval()
            except Exception as exc:  # torch missing, no weights, etc.
                self._model_error = exc
        return self._model

    @property
    def available(self):
        return self.model is not None

    @property
    def error(self):
        _ = self.model
        return self._model_error

    def _softmax(self, logits):
        import math
        top = max(logits)
        exps = [math.exp(v - top) for v in logits]
        total = sum(exps) or 1.0
        return [v / total for v in exps]

    def action_from_probs(self, probs, dt=0.25):
        """Map class probabilities to an action dict (gated + debounced)."""
        self._cooldown_left = max(0.0, self._cooldown_left - dt)
        if len(probs) != self.n_classes:
            return neutral_action()
        self._smooth = [
            self.ema * old + (1.0 - self.ema) * new
            for old, new in zip(self._smooth, probs)
        ]
        best = max(range(self.n_classes), key=lambda i: self._smooth[i])
        if self._smooth[best] < self.confidence or self._cooldown_left > 0:
            return neutral_action()
        self._cooldown_left = self.cooldown
        command = GNN_COMMANDS[best] if best < len(GNN_COMMANDS) else None
        return map_command_to_action(command)

    def action_from_logits(self, logits, dt=0.25):
        return self.action_from_probs(self._softmax(list(logits)), dt=dt)

    def push_to_game(self, action):
        """Publish an action dict to the in-memory AI state."""
        _state.update({k: action.get(k, _state[k]) for k in _ACTION_KEYS
                       if k in action})
        return dict(_state)
