"""Simple logical autopilot: a rule-based demo of what the AI should do.

Press P to toggle (no on-screen text by design). While on, the autopilot
*replaces* the brain/GNN input with heuristic play; press P again to hand
control back to the brain. The keyboard always merges on top either way.

The heuristics deliberately play human, not optimal: idle pauses and
look-arounds while wandering, a startle beat when an enemy pops up
close, swing bursts with breathing room instead of frame-perfect trading,
jump-swings at bats overhead, cautious approaches, committed retreats at
low HP, and a recovery pause after hard landings.

Only the core command set is used (move/jump/attack/slot), matching what
the GNN outputs; mouse-driven verbs (mine/torch/place-at-cursor) are out
of scope for a logical demo.
"""

import random

from game import ai_control
from game.config import TILE
from game.tiles import HOTBAR, SWORD


class Autopilot:
    """Stateful heuristic player. One instance lives on the Game."""
    # Some example autopilot here
