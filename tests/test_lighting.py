"""Lighting regression tests (stdlib only, no pygame needed)."""
import unittest

from game.tiles import AIR, TORCH, STONE
from game.world import World
from game.lighting import Lighting


def open_world():
    world = World()
    # Carve an open-air pocket so the torch test is isolated.
    for x in range(45, 58):
        for y in range(10, 30):
            world.set(x, y, AIR)
    return world


class TestLighting(unittest.TestCase):
    def test_diagonal_spread(self):
        world = open_world()
        world.set(50, 20, TORCH)
        light = Lighting(world)
        light.compute(0.0)  # night: only the torch lights the pocket
        ortho = light.brightness_at(51, 20)
        diag = light.brightness_at(51, 21)
        self.assertGreater(ortho, 0, "orthogonal neighbour should be lit")
        self.assertGreater(diag, 0, "diagonal neighbour should be lit")
        self.assertLess(diag, ortho,
                         "diagonal should cost more than orthogonal")

    def test_no_diagonal_leak_through_corners(self):
        world = open_world()
        world.set(50, 20, TORCH)
        # Seal both orthogonal neighbours with stone.
        world.set(51, 20, STONE)
        world.set(50, 21, STONE)
        light = Lighting(world)
        light.compute(0.0)
        sealed = light.brightness_at(51, 21)

        world2 = open_world()
        world2.set(50, 20, TORCH)
        light2 = Lighting(world2)
        light2.compute(0.0)
        open_diag = light2.brightness_at(51, 21)
        self.assertLess(sealed, open_diag,
                         "solid corner should block diagonal spread")

    def test_display_smoothing_converges(self):
        world = open_world()
        light = Lighting(world)
        light.compute(1.0)
        light.update_display(10.0)
        first = light.light_at_display(50, 15)
        self.assertEqual(first, light.light_at(50, 15))
        # Recompute at night, then ease the display towards it.
        light.compute(0.0)
        for _ in range(600):
            light.update_display(1 / 60.0)
        night = light.light_at(50, 15)
        shown = light.light_at_display(50, 15)
        for a, b in zip(night, shown):
            self.assertLessEqual(abs(a - b), 2)

    def test_compute_is_cached(self):
        world = open_world()
        light = Lighting(world)
        self.assertTrue(light.compute(0.5))
        version = light.version
        self.assertFalse(light.compute(0.5))
        self.assertEqual(light.version, version)


if __name__ == "__main__":
    unittest.main()
