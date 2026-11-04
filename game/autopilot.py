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

AGRO_X = TILE * 6      # chase enemies within this horizontal range
AGRO_Y = TILE * 3
STRIKE_X = TILE * 1.6  # stand and swing within this box
STRIKE_Y = TILE * 2.0
STARTLE_X = TILE * 2.5  # close-range surprises cause a startle beat
FLEE_HP = 4            # retreat from enemies at or below this HP
LAND_PAUSE_V = 700.0   # falls faster than this need a recovery pause


class Autopilot:
    """Stateful heuristic player. One instance lives on the Game."""

    def __init__(self, direction=1):
        self.dir = 1 if direction >= 0 else -1
        self.t = 0.0
        self.wander_t = 5.0
        self.pause_t = 0.0
        self.stuck_t = 0.0
        self.last_x = None
        self.flee_dir = 0
        self.flee_lost_t = 0.0
        self.known_foes = set()
        self._was_ground = True
        self._fall_peak = 0.0

    def action(self, game, dt):
        """Compute this tick's action dict from live game state."""
        player = game.player
        world = game.world
        self.t += dt
        act = ai_control.neutral_action()
        self._track_landing(player)

        foe, foe_dx, foe_dy = self._nearest_foe(game)
        if foe is None:
            self.known_foes.clear()
        just_startled = False
        if foe is not None and id(foe) not in self.known_foes:
            self.known_foes.add(id(foe))
            if abs(foe_dx) < STARTLE_X and player.hp > FLEE_HP:
                self.pause_t = max(self.pause_t, 0.3)
                just_startled = True

        striking = (foe is not None and abs(foe_dx) < STRIKE_X
                    and abs(foe_dy) < STRIKE_Y)

        # Hurt and threatened: committed retreat, no heroics.
        if self.flee_dir != 0 or (foe is not None and player.hp <= FLEE_HP):
            return self._flee(act, game, foe, foe_dx, dt)
        # Close combat: brace through a fresh startle, then fight.
        if striking:
            if self.pause_t > 0 or just_startled:
                self.pause_t = max(0.0, self.pause_t - dt)
                act["slot"] = HOTBAR.index(SWORD)
                return self._finish(act, player)
            return self._fight(act, game, foe_dx, foe_dy)
        # Water: always swim, never pause-drown.
        if world.water_at_pixel(player.rect.centerx, player.rect.centery):
            act["move"] = float(self.dir)
            act["jump"] = True
            return self._finish(act, player)
        # Pause: stand still (hesitation, startle, landing recovery).
        if self.pause_t > 0:
            self.pause_t -= dt
            if foe is not None:
                act["slot"] = HOTBAR.index(SWORD)
            return self._finish(act, player)
        # Wall ahead: hop it.
        if self._wall_ahead(world, player):
            act["move"] = float(self.dir)
            act["jump"] = True
            return self._finish(act, player)
        # Cliff ahead: turn around (water below is safe).
        if self._cliff_ahead(game, player):
            self.dir = -self.dir
            self.stuck_t = 0.0
        # Wander rhythm: mostly straight, sometimes pause or turn.
        self.wander_t -= dt
        if self.wander_t <= 0:
            self.wander_t = 4.0 + random.random() * 4.0
            roll = random.random()
            if roll < 0.35:
                self.pause_t = 0.4 + random.random() * 0.8
            elif roll < 0.6:
                self.dir = -self.dir
        if foe is not None:  # cautious approach, weapon ready
            if foe_dx != 0:
                self.dir = 1 if foe_dx > 0 else -1
            act["move"] = 0.85 * self.dir
            act["slot"] = HOTBAR.index(SWORD)
        else:
            act["move"] = float(self.dir)
        # Stuck: hop, then turn around.
        hop, flip = self._stuck_update(player, dt, act["move"])
        if hop:
            act["jump"] = True
        if flip:
            self.dir = -self.dir
            act["move"] = float(self.dir)
        return self._finish(act, player)

    # ------------------------------------------------------------ helpers
    def _finish(self, act, player):
        self.last_x = player.rect.x
        return act

    def _nearest_foe(self, game):
        player = game.player
        foe, foe_dx, foe_dy = None, 0.0, 0.0
        for enemy in game.enemies:
            if enemy.hp <= 0:
                continue
            dx = enemy.rect.centerx - player.rect.centerx
            dy = enemy.rect.centery - player.rect.centery
            if abs(dx) < AGRO_X and abs(dy) < AGRO_Y:
                if foe is None or abs(dx) < abs(foe_dx):
                    foe, foe_dx, foe_dy = enemy, dx, dy
        return foe, foe_dx, foe_dy

    def _track_landing(self, player):
        """Flag hard landings for a recovery pause."""
        if not player.on_ground:
            self._was_ground = False
            vy = getattr(player.vel, "y", 0.0) or 0.0
            self._fall_peak = max(self._fall_peak, vy)
        elif not self._was_ground:
            self._was_ground = True
            if self._fall_peak > LAND_PAUSE_V:
                self.pause_t = 0.45 + random.random() * 0.3
            self._fall_peak = 0.0

    def _stuck_update(self, player, dt, moving):
        """Returns (hop, flip) for stuck handling; resets when moving."""
        if self.last_x is None or not player.on_ground or not moving:
            self.stuck_t = 0.0
            return False, False
        if abs(player.rect.x - self.last_x) < 1.0:
            self.stuck_t += dt
        else:
            self.stuck_t = 0.0
        if self.stuck_t > 1.5:
            self.stuck_t = 0.0
            return True, True
        return self.stuck_t > 0.6, False

    def _fight(self, act, game, foe_dx, foe_dy):
        """Planted swing bursts with breathing room, jump-swings upward."""
        if foe_dx != 0:
            self.dir = 1 if foe_dx > 0 else -1
        act["slot"] = HOTBAR.index(SWORD)
        act["move"] = 0.0
        burst = (self.t % 1.2) < 0.55
        act["attack"] = burst and (self.t % 0.5) < 0.25
        if foe_dy < -TILE:  # foe overhead (bats): swing on the way up
            act["jump"] = True
        return self._finish(act, game.player)

    def _flee(self, act, game, foe, foe_dx, dt):
        """Committed retreat: one direction, hop walls, never flee off."""
        player = game.player
        world = game.world
        if foe is None:
            self.flee_lost_t += dt
            if self.flee_lost_t > 1.0:  # threat gone: stop, breathe
                self.flee_dir = 0
                self.flee_lost_t = 0.0
                return self._finish(act, player)
        else:
            self.flee_lost_t = 0.0
            if foe_dx != 0:
                self.flee_dir = -1 if foe_dx > 0 else 1
            elif self.flee_dir == 0:
                self.flee_dir = -self.dir
            self.dir = self.flee_dir
        act["move"] = float(self.flee_dir)
        act["slot"] = HOTBAR.index(SWORD)
        if world.water_at_pixel(player.rect.centerx, player.rect.centery):
            act["jump"] = True  # swim for it, keep running
            return self._finish(act, player)
        if self._wall_ahead(world, player):
            act["jump"] = True
        if self._cliff_ahead(game, player):
            self.flee_dir = -self.flee_dir
            self.dir = self.flee_dir
            act["move"] = float(self.flee_dir)
        hop, _flip = self._stuck_update(player, dt, act["move"])
        if hop:  # hop only: never turn back toward the threat
            act["jump"] = True
        return self._finish(act, player)

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
