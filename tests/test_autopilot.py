"""Autopilot tests: heuristic logic on a stub world + P-toggle integration.

Needs pygame (game/requirements.txt). Skips cleanly when missing.
Run:  .venv/Scripts/python -m unittest discover -s tests   (Windows)
"""
import os
import unittest
from types import SimpleNamespace

try:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
except ImportError:
    pygame = None

from game.config import TILE


def _stub_world(water=False):
    """Ground plane at ty>=10 with a gap (tx 5-8) and a wall (tx 14)."""
    return SimpleNamespace(
        solid=lambda tx, ty: (
            False if 5 <= tx <= 8 and ty >= 10
            else (ty >= 10 or (tx == 14 and 7 <= ty <= 9))),
        water_at_pixel=lambda _x, _y: water)


def _stub_player(x, hp=10):
    return SimpleNamespace(
        rect=pygame.Rect(x, 8 * TILE, int(TILE * 0.8), int(TILE * 1.5)),
        vel=pygame.Vector2(0, 0),
        on_ground=True, hp=hp)


def _stub_game(player, world, enemies=()):
    return SimpleNamespace(player=player, world=world, enemies=list(enemies))


def _foe(x, y, hp=10):
    return SimpleNamespace(rect=pygame.Rect(x, y, 64, 64), hp=hp)


@unittest.skipIf(pygame is None, "pygame not installed")
class TestAutopilotLogic(unittest.TestCase):
    def test_action_shape(self):
        from game import ai_control
        from game.autopilot import Autopilot
        game = _stub_game(_stub_player(100), _stub_world())
        act = Autopilot().action(game, 1 / 60.0)
        self.assertEqual(set(act), set(ai_control.neutral_action()))
        self.assertGreaterEqual(act["move"], -1.0)
        self.assertLessEqual(act["move"], 1.0)

    def test_wall_ahead_jumps(self):
        from game.autopilot import Autopilot
        game = _stub_game(_stub_player(1060), _stub_world())
        act = Autopilot(direction=1).action(game, 1 / 60.0)
        self.assertTrue(act["jump"])
        self.assertEqual(act["move"], 1.0)

    def test_cliff_ahead_turns(self):
        from game.autopilot import Autopilot
        game = _stub_game(_stub_player(330), _stub_world())
        agent = Autopilot(direction=1)
        act = agent.action(game, 1 / 60.0)
        self.assertEqual(agent.dir, -1)
        self.assertEqual(act["move"], -1.0)

    def test_water_swims_up(self):
        from game.autopilot import Autopilot
        game = _stub_game(_stub_player(100), _stub_world(water=True))
        act = Autopilot(direction=1).action(game, 1 / 60.0)
        self.assertTrue(act["jump"])

    def test_strike_range_braces_then_bursts(self):
        from game.autopilot import Autopilot
        from game.tiles import HOTBAR, SWORD
        player = _stub_player(100)
        foe = _foe(player.rect.centerx + 100 - 32,
                   player.rect.centery - 32)
        game = _stub_game(player, _stub_world(), [foe])
        agent = Autopilot(direction=1)
        first = agent.action(game, 0.1)
        self.assertFalse(first["attack"])  # startle beat, no instant swing
        attacks = [first] + [agent.action(game, 0.1) for _ in range(23)]
        self.assertTrue(all(a["slot"] == HOTBAR.index(SWORD)
                            for a in attacks))
        self.assertTrue(all(a["move"] == 0.0 for a in attacks))
        self.assertTrue(any(a["attack"] for a in attacks))

    def test_far_enemy_approached_cautiously(self):
        from game.autopilot import Autopilot
        player = _stub_player(100)
        foe = _foe(player.rect.centerx + 300 - 32,
                   player.rect.centery - 32)
        game = _stub_game(player, _stub_world(), [foe])
        act = Autopilot(direction=1).action(game, 1 / 60.0)
        self.assertGreater(act["move"], 0.5)
        self.assertLessEqual(act["move"], 1.0)
        self.assertFalse(act["attack"])

    def test_idle_pause_while_wandering(self):
        from unittest import mock
        from game.autopilot import Autopilot
        game = _stub_game(_stub_player(100), _stub_world())
        agent = Autopilot(direction=1)
        with mock.patch("game.autopilot.random.random", return_value=0.1):
            paused = any(agent.action(game, 0.1)["move"] == 0.0
                         for _ in range(80))
        self.assertTrue(paused, "wander should sometimes pause")

    def test_hard_landing_pauses(self):
        from game.autopilot import Autopilot
        player = _stub_player(100)
        game = _stub_game(player, _stub_world())
        agent = Autopilot(direction=1)
        player.on_ground = False
        player.vel.y = 900.0
        agent.action(game, 1 / 60.0)  # airborne: record the fall
        player.on_ground = True
        act = agent.action(game, 1 / 60.0)  # touchdown: recovery pause
        self.assertEqual(act["move"], 0.0)

    def test_low_hp_retreats(self):
        from game.autopilot import Autopilot
        player = _stub_player(100, hp=2)
        foe = _foe(player.rect.centerx + 100 - 32,
                   player.rect.centery - 32)
        game = _stub_game(player, _stub_world(), [foe])
        act = Autopilot(direction=1).action(game, 1 / 60.0)
        self.assertEqual(act["move"], -1.0)


@unittest.skipIf(pygame is None, "pygame not installed")
class TestAutopilotToggle(unittest.TestCase):
    def _boot(self):
        from game.game import Game
        real_set_mode = pygame.display.set_mode
        pygame.display.set_mode = lambda *a, **k: real_set_mode((1280, 720))
        try:
            return Game()
        finally:
            pygame.display.set_mode = real_set_mode

    def test_p_toggles_silently(self):
        game = self._boot()
        try:
            self.assertFalse(game.autopilot_on)
            msg_before = game.msg
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN,
                                                 key=pygame.K_p))
            self.assertTrue(game.handle_events())
            self.assertTrue(game.autopilot_on)
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN,
                                                 key=pygame.K_p))
            self.assertTrue(game.handle_events())
            self.assertFalse(game.autopilot_on)
            self.assertEqual(game.msg, msg_before)  # no explanatory text
        finally:
            pygame.quit()

    def test_autopilot_plays_and_overrides_brain(self):
        import json
        from game import ai_control
        from gnn.live import GAME_INPUT, reset_game_input
        game = self._boot()
        try:
            ai_control.clear_action()
            reset_game_input()
            # Stand on flat ground so the first steps are deterministic.
            surf = game.world.surface
            flat = next(tx for tx in range(len(surf) - 3)
                        if len(set(surf[tx:tx + 3])) == 1)
            game.player.rect.midbottom = (flat * TILE + TILE // 2,
                                          surf[flat] * TILE)
            game.player.vel.update(0, 0)
            # Hostile brain input says LEFT the whole time.
            with open(GAME_INPUT, "w") as fh:
                json.dump({"move": -1.0}, fh)
            game.autopilot_on = True
            x0 = game.player.rect.x
            for _ in range(10):
                game.update(1 / 60.0)
            self.assertGreater(
                game.player.rect.x, x0,
                "autopilot must replace the brain file, not merge it")
            game.autopilot_on = False
        finally:
            reset_game_input()
            ai_control.clear_action()
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
