"""Simple logical autopilot: a rule-based demo of what the AI should do.

Press P to toggle (no on-screen text by design). While on, the autopilot
*replaces* the brain/GNN input with heuristic play — wander, hop over
walls, avoid cliffs, swim, chase and fight nearby enemies, retreat at low
HP. Press P again to hand control back to the brain. The keyboard always
merges on top either way.

Only the core command set is used (move/jump/attack/slot), matching what
the GNN outputs; mouse-driven verbs (mine/torch/place-at-cursor) are out
of scope for a logical demo.
"""

import random

from game import ai_control
from game.config import TILE
from game.tiles import HOTBAR, SWORD

AGRO_X = TILE * 6      # chase enemies within this horizontal range
AGRO_Y = TILE * 3
STRIKE_X = TILE * 1.6  # stand and swing within this box
STRIKE_Y = TILE * 2.0
FLEE_HP = 3            # retreat from enemies at or below this HP


class Autopilot:
    """Stateful heuristic player. One instance lives on the Game."""

    def __init__(self, direction=1):
        self.dir = 1 if direction >= 0 else -1
        self.t = 0.0
        self.wander_t = 5.0
        self.stuck_t = 0.0
        self.last_x = None

    def action(self, game, dt):
        """Compute this tick's action dict from live game state."""
        player = game.player
        world = game.world
        self.t += dt
        act = ai_control.neutral_action()

        # --- combat: nearest enemy in agro range ---
        foe, foe_dx = None, 0.0
        for enemy in game.enemies:
            if enemy.hp <= 0:
                continue
            dx = enemy.rect.centerx - player.rect.centerx
            dy = enemy.rect.centery - player.rect.centery
            if abs(dx) < AGRO_X and abs(dy) < AGRO_Y:
                if foe is None or abs(dx) < abs(foe_dx):
                    foe, foe_dx = enemy, dx
        sword_slot = HOTBAR.index(SWORD)
        if foe is not None:
            act["slot"] = sword_slot
            if foe_dx != 0:
                self.dir = 1 if foe_dx > 0 else -1
            striking = (abs(foe_dx) < STRIKE_X
                        and abs(foe.rect.centery - player.rect.centery)
                        < STRIKE_Y)
            if player.hp <= FLEE_HP:  # hurt: back away from the foe
                act["move"] = float(-self.dir)
                return self._finish(act, player)
            if striking:  # plant feet, pulse the swing for repeat hits
                act["move"] = 0.0
                act["attack"] = (self.t % 0.5) < 0.25
                return self._finish(act, player)
            act["move"] = float(self.dir)  # closing in; jumps/cliffs below

        # --- water: swim up, keep drifting ---
        if world.water_at_pixel(player.rect.centerx, player.rect.centery):
            act["move"] = float(self.dir)
            act["jump"] = True
            return self._finish(act, player)

        # --- wall ahead: hop it ---
        if self._wall_ahead(world, player):
            act["move"] = float(self.dir)
            act["jump"] = True
            return self._finish(act, player)

        # --- cliff ahead: turn around (water below is safe) ---
        if self._cliff_ahead(game, player):
            self.dir = -self.dir
            self.stuck_t = 0.0

        # --- wander: mostly straight, occasional turn ---
        self.wander_t -= dt
        if self.wander_t <= 0:
            self.wander_t = 5.0 + random.random() * 5.0
            if random.random() < 0.25:
                self.dir = -self.dir
        if foe is None:
            act["move"] = float(self.dir)

        # --- stuck: hop, then turn around ---
        if self.last_x is not None and player.on_ground and act["move"]:
            if abs(player.rect.x - self.last_x) < 1.0:
                self.stuck_t += dt
            else:
                self.stuck_t = 0.0
            if self.stuck_t > 0.6:
                act["jump"] = True
            if self.stuck_t > 1.5:
                self.dir = -self.dir
                act["move"] = float(self.dir)
                self.stuck_t = 0.0
        else:
            self.stuck_t = 0.0
        return self._finish(act, player)

    def _finish(self, act, player):
        self.last_x = player.rect.x
        return act

    def _wall_ahead(self, world, player):
        if self.dir > 0:
            nose = player.rect.right + 4
        else:
            nose = player.rect.left - 4
        tx = int(nose // TILE)
        torso = world.solid(tx, int(player.rect.centery // TILE))
        head = world.solid(tx, int((player.rect.top + 6) // TILE))
        return torso or head

    def _cliff_ahead(self, game, player):
        world = game.world
        ahead_x = player.rect.centerx + self.dir * TILE * 1.2
        feet = player.rect.bottom
        for i in range(1, 5):
            if world.solid(int(ahead_x // TILE),
                           int((feet + i * TILE) // TILE)):
                return False
        # No ground, but water below means a safe swim, not a fall.
        if world.water_at_pixel(ahead_x, feet + TILE):
            return False
        return True
