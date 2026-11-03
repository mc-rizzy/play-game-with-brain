"""Headless pygame smoke test: boot the game and run frames.

Requires pygame (game/requirements.txt). Skips cleanly when missing.
Run:  .venv/Scripts/python -m unittest discover -s tests   (Windows)
"""
import os
import unittest

try:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame  # noqa: F401
except ImportError:
    pygame = None


@unittest.skipIf(pygame is None, "pygame not installed")
class TestSmoke(unittest.TestCase):
    def test_boot_and_run_frames(self):
        import pygame
        from game import ai_control
        from game.game import Game

        # Force windowed under the dummy driver (no real fullscreen).
        real_set_mode = pygame.display.set_mode
        pygame.display.set_mode = lambda *a, **k: real_set_mode((1280, 720))
        try:
            game = Game()
        finally:
            pygame.display.set_mode = real_set_mode

        try:
            self.assertEqual((game.W, game.H), (1280, 720))
            self.assertAlmostEqual(game.ui_scale, 1.0)
            self.assertIn("player_idle", game.entity)
            self.assertGreater(len(game.icons), 0)

            # AI-only input moves the player (no keyboard held headless).
            ai_control.set_action(move=1.0)
            x0 = game.player.rect.x
            try:
                for _ in range(30):
                    game.update(1 / 60.0)
            finally:
                ai_control.clear_action()
            self.assertGreater(game.player.rect.x, x0,
                               "AI move=+1 should push the player right")

            # Draw every layer without errors.
            for _ in range(5):
                game.draw_world()
                game.draw_entities()
                game.draw_foreground()
                game.draw_ui()
            game.craft_open = True
            game.draw_ui()
            game.craft_open = False

            # Lighting overlay was built and cached.
            self.assertIsNotNone(game.light_screen_surface)
            key = game._light_cache_key
            game.draw_lighting(*game.visible)
            self.assertEqual(key, game._light_cache_key)

            # GUI rescales on larger windows.
            game.W, game.H = 1920, 1080
            game.refresh_ui_scale()
            self.assertGreater(game.ui_scale, 1.0)
            self.assertGreater(game.slot_size, 54)
            game.refresh_icons()
            game.draw_ui()
        finally:
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
