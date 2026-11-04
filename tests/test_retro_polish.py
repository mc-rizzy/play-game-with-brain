"""Retro polish regression tests: 16x16 art, blocky lighting, UI layout."""
import os
import unittest

try:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame  # noqa: F401
except ImportError:
    pygame = None


class TestRetroArt(unittest.TestCase):
    def test_tiles_authored_at_16(self):
        from game.sprites import ART
        self.assertEqual(ART, 16)

    def test_sun_intensity_is_continuous(self):
        # Daylight scales at render time: full 0..255 precision (no
        # banding), and a sun-only change never triggers a re-solve.
        from game.world import World
        from game.lighting import Lighting
        world = World()
        light = Lighting(world)
        self.assertTrue(light.compute(0.5))
        self.assertEqual(light.sunval, 127)
        self.assertFalse(light.compute(0.51))
        self.assertEqual(light.sunval, 130)

    def test_display_reads_live_sim(self):
        from game.tiles import AIR
        from game.world import World
        from game.lighting import Lighting
        world = World()
        for x in range(45, 58):
            for y in range(10, 30):
                world.set(x, y, AIR)
        light = Lighting(world)
        light.compute(1.0)
        self.assertEqual(light.light_at_display(50, 15),
                         light.light_at(50, 15))
        light.compute(0.0)
        for _ in range(600):
            light.update_display(1 / 60.0)
        self.assertEqual(light.light_at_display(50, 15),
                         light.light_at(50, 15))

    def test_breaking_caster_lifts_shadow_same_tick(self):
        # Day, floating stone in open air: its column is shadowed, and
        # breaking it restores full sun with no compute() call in
        # between (the edit path recomputes eagerly, nothing is lazy).
        from game.tiles import AIR, STONE
        from game.world import World
        from game.lighting import Lighting
        world = World()
        for x in range(45, 58):
            for y in range(0, 30):
                world.set(x, y, AIR)
        light = Lighting(world)
        light.compute(1.0)
        self.assertEqual(light.brightness_at(50, 16), 255)
        world.set(50, 12, STONE)
        self.assertLess(light.brightness_at(50, 16), 200)
        world.set(50, 12, AIR)
        self.assertEqual(light.brightness_at(50, 16), 255)

    def test_sun_softens_sideways(self):
        # Each sub-cell observes its horizontal neighbours: shadow
        # edges render as umbra -> penumbra -> open gradients.
        from game.tiles import AIR, STONE
        from game.world import World
        from game.lighting import Lighting
        world = World()
        for x in range(45, 58):
            for y in range(0, 30):
                world.set(x, y, AIR)
        light = Lighting(world)
        light.compute(1.0)
        world.set(50, 12, STONE)
        umbra = light.sun_smooth(100, 33)
        penumbra = light.sun_smooth(102, 33)
        open_sky = light.sun_smooth(106, 33)
        self.assertLess(umbra, penumbra)
        self.assertLess(penumbra, open_sky)
        self.assertEqual(open_sky, 255)

    def test_light_updates_on_block_break(self):
        from game.tiles import AIR, TORCH
        from game.world import World
        from game.lighting import Lighting
        world = World()
        for x in range(45, 58):
            for y in range(10, 30):
                world.set(x, y, AIR)
        world.set(50, 20, TORCH)
        light = Lighting(world)
        light.compute(0.0)
        lit = light.brightness_at(51, 20)
        self.assertGreater(lit, 0)
        # Breaking the torch must darken its neighbours on recompute.
        world.set(50, 20, AIR)
        self.assertTrue(light.compute(0.0))
        self.assertLess(light.brightness_at(51, 20), lit)

    def test_water_falls_into_broken_block(self):
        from game.tiles import AIR, STONE, WATER
        from game.world import World
        from game.config import WORLD_W
        world = World()
        for _ in range(600):
            if not world.water_tick():
                break
    def test_water_falls_into_broken_block(self):
        from game.tiles import AIR, STONE, WATER
        from game.world import World
        from game.config import WORLD_W
        from game.world import TILE_DROPS
        world = World()
        for _ in range(600):
            if not world.water_tick():
                break
        # Stone shaft with a floor; water poured at the top must land
        # on the floor with all drops conserved.
        for y in range(20, 26):
            world.set(59, y, STONE)
            world.set(61, y, STONE)
        for y in range(21, 26):
            world.set(60, y, AIR)
        world.set(60, 26, STONE)
        world.set(60, 20, WATER)
        before = sum(world.wtc)
        for _ in range(400):
            if not world.water_tick():
                break
        self.assertEqual(sum(world.wtc), before)
        self.assertEqual(world.wtc[25 * WORLD_W + 60], TILE_DROPS)
        self.assertFalse(world.water_tick(), "water should reach rest")

    def test_water_levels_flat_in_open_basin(self):
        # Poured water must spread by pressure until wet neighbours
        # differ by at most one drop (a flat surface, not a pile);
        # film edges below the pour threshold hold by tension.
        from game.tiles import AIR, STONE, WATER
        from game.world import World
        from game.config import WORLD_W, POUR_MIN
        world = World()
        for _ in range(600):
            if not world.water_tick():
                break
        for x in range(48, 63):
            world.set(x, 30, STONE)
            world.set(x, 29, AIR)
        world.set(55, 29, WATER)
        before = sum(world.wtc)
        for _ in range(600):
            if not world.water_tick():
                break
        self.assertEqual(sum(world.wtc), before)
        row = [world.wtc[29 * WORLD_W + x] for x in range(48, 63)]
        self.assertGreater(sum(1 for v in row if v), 1,
                           f"pour should spread, got {row}")
        for j in range(len(row) - 1):
            a, b = row[j], row[j + 1]
            if a and b:
                self.assertLessEqual(abs(a - b), 1,
                                     f"surface not flat: {row}")
            elif a or b:
                self.assertLess(max(a, b), POUR_MIN,
                                f"film edge should hold: {row}")
        self.assertFalse(world.water_tick(), "water should reach rest")

    def test_water_volume_conserved_and_settles(self):
        from game.world import World
        world = World()
        before = sum(world.wtc)
        self.assertGreater(before, 0)
        for _ in range(600):
            if not world.water_tick():
                break
        self.assertEqual(sum(world.wtc), before,
                         "flow must not create/destroy water")
        self.assertFalse(world.water_tick(), "water should reach rest")

    def test_houses_stand_by_spawn(self):
        # Two plank houses (windows, torches, clear spawn) stand
        # within sight of spawn.
        from game.tiles import PLANKS, GLASS, TORCH
        from game.world import World
        from game.config import WORLD_W
        world = World()
        sx = WORLD_W // 2
        tiles = [world.g[x][y] for x in range(sx - 22, sx + 23)
                 for y in range(world.surface[x])]
        self.assertGreaterEqual(tiles.count(PLANKS), 30, "plank houses")
        self.assertGreaterEqual(tiles.count(GLASS), 4, "windows")
        self.assertGreaterEqual(tiles.count(TORCH), 4, "house torches")
        for y in range(world.surface[sx] - 3, world.surface[sx]):
            self.assertFalse(world.solid(sx, y), "spawn must be clear")

    def test_leaves_pass_most_sunlight(self):
        # A 3x3 leaves canopy (a tree crown) must read as dappled
        # shade, not a hard shadow: well above a stone canopy of the
        # same size, which stays dark.
        from game.tiles import AIR, STONE, LEAVES
        from game.world import World
        from game.lighting import Lighting

        def shade_under(block):
            world = World()
            for x in range(45, 58):
                for y in range(0, 30):
                    world.set(x, y, AIR)
            for x in range(49, 52):
                for y in range(10, 13):
                    world.set(x, y, block)
            light = Lighting(world)
            light.compute(1.0)
            return light.brightness_at(50, 16)

        self.assertGreater(shade_under(LEAVES), 120,
                           "tree crowns should pass most sunlight")
        self.assertLess(shade_under(STONE), 60,
                        "stone must still cast hard shadows")


@unittest.skipIf(pygame is None, "pygame not installed")
class TestUILayout(unittest.TestCase):
    def test_entity_feet_touch_last_row(self):
        # Blits are bottom-aligned, so any transparent row at the
        # bottom of the art reads as floating above the ground.
        import pygame
        from game.config import ASSETS_DIR
        for name in ("player_idle", "player_walk1", "player_walk2",
                     "slime_0", "slime_1", "shade_0", "shade_1",
                     "skeleton_0", "skeleton_1"):
            img = pygame.image.load(f"{ASSETS_DIR}/{name}.png")
            w, h = img.get_size()
            opaque = any(img.get_at((x, h - 1))[3] != 0
                         for x in range(w))
            self.assertTrue(opaque, f"{name} floats: last row is empty")

    def test_hearts_top_right_inventory_top_left(self):
        import pygame
        from game.game import Game
        real_set_mode = pygame.display.set_mode
        pygame.display.set_mode = lambda *a, **k: real_set_mode((1280, 720))
        try:
            game = Game()
        finally:
            pygame.display.set_mode = real_set_mode
        try:
            game.draw_world()
            game.draw_entities()
            game.draw_foreground()
            game.draw_ui()

            def has_color(x0, x1, y0, y1, pred):
                for x in range(x0, x1, 4):
                    for y in range(y0, y1, 4):
                        r, g, b, _ = game.screen.get_at((x, y))
                        if pred(r, g, b):
                            return True
                return False

            is_heart_red = lambda r, g, b: r > 200 and g < 120 and b < 120
            is_gold = lambda r, g, b: r > 240 and g > 200 and b < 160
            W = game.W
            # Hearts live top-right, not top-left.
            self.assertTrue(
                has_color(W - 400, W - 10, 10, 70, is_heart_red),
                "expected heart red pixels in the top-right")
            self.assertFalse(
                has_color(10, 300, 10, 60, is_heart_red),
                "hearts should not be in the top-left")
            # Inventory selection frame lives top-left.
            self.assertTrue(
                has_color(10, 520, 10, 70, is_gold),
                "expected gold selection frame in the top-left")
        finally:
            pygame.quit()


@unittest.skipIf(pygame is None, "pygame not installed")
class TestSmallFixes(unittest.TestCase):
    def _make_game(self):
        import pygame
        from game.game import Game
        real_set_mode = pygame.display.set_mode
        pygame.display.set_mode = lambda *a, **k: real_set_mode((1280, 720))
        try:
            game = Game()
        finally:
            pygame.display.set_mode = real_set_mode
        self.addCleanup(pygame.quit)
        return game

    def test_broken_block_shows_sky(self):
        # Dug-out cells blend with the sky background (the lighting
        # overlay still shades them by depth) instead of rendering a
        # block lookalike or a black patch.
        from game.tiles import AIR
        from game.config import TILE, WORLD_W
        game = self._make_game()
        tx = WORLD_W // 2
        sy = game.world.surface[tx]
        for x in (tx - 1, tx, tx + 1):
            for y in range(sy + 2):
                game.world.set(x, y, AIR)
        game.cam.x = tx * TILE - game.W / 2
        game.cam.y = sy * TILE - game.H / 2
        game.draw_world()
        sx = int(tx * TILE - game.cam.x)
        syy = int((sy + 1) * TILE - game.cam.y)
        blues = []
        for dx in range(30, 51, 5):
            for dy in range(30, 51, 5):
                r, g, b = game.screen.get_at((sx + dx, syy + dy))[:3]
                blues.append(b)
                self.assertGreater(
                    b, r, "dug cells must read sky-blue, not brown")
        self.assertGreater(min(blues), 150,
                           "open dug cells must be bright sky")

    def test_still_water_asset_is_plain_blue_16(self):
        from game.sprites import make_still_water
        still = make_still_water()
        self.assertEqual(still.get_size(), (16, 16))
        colours = set()
        for x in range(16):
            for y in range(16):
                r, g, b, _ = still.get_at((x, y))
                colours.add((r, g, b))
                self.assertLess(r, 90, "still water has no wave foam")
        self.assertLessEqual(len(colours), 2,
                             f"still water must be flat blue: {colours}")

    def test_water_animates_only_where_player_swims(self):
        # Only SURFACE water the player wades through animates: water
        # under water stays still, as does water away from the player.
        from game.tiles import WATER
        from game.config import TILE
        game = self._make_game()
        world = game.world
        world.set(60, 3, WATER)
        world.set(60, 4, WATER)
        world.set(90, 3, WATER)
        game.player.rect.topleft = (60 * TILE, 3 * TILE)
        self.assertTrue(game.water_tile_animated(60, 3),
                        "surface water at the player must animate")
        self.assertFalse(game.water_tile_animated(60, 4),
                         "water under water must stay still")
        self.assertFalse(game.water_tile_animated(90, 3),
                         "distant surface water must stay still")
        self.assertFalse(game.water_tile_animated(100, 3),
                         "dry tiles never animate")

    def test_player_jumps_out_of_water(self):
        # Holding jump must breach well clear of the surface (high
        # enough to climb back onto a one-block shore), not just bob.
        import pygame
        from game.config import TILE, WORLD_W
        from game import ai_control
        game = self._make_game()
        world = game.world
        lakes = [x for x in range(100) if world.surface[x] > 40]
        lx = lakes[len(lakes) // 2]
        ly = world.surface[lx] - 2
        self.assertTrue(world.wtc[ly * WORLD_W + lx])
        self.assertTrue(world.wtc[(ly + 1) * WORLD_W + lx])
        game.player.rect.topleft = (lx * TILE + 8, ly * TILE)
        game.player.vel.update(0, 0)
        game.enemies = []
        game.spawn_timer = 9999
        for _ in range(60):
            game.update(1 / 60)
        p = game.player.rect
        self.assertTrue(world.water_at_pixel(p.centerx, p.centery),
                        "player should start submerged")
        water_top = p.centery
        while water_top > 0 and world.water_at_pixel(p.centerx,
                                                     water_top):
            water_top -= 4
        real_keys = pygame.key.get_pressed
        real_poll = ai_control.poll_ai_action

        class Keys:
            def __getitem__(self, k):
                return k == pygame.K_SPACE

        pygame.key.get_pressed = lambda: Keys()
        ai_control.poll_ai_action = ai_control.neutral_action
        try:
            top = p.centery
            for _ in range(240):
                game.update(1 / 60)
                top = min(top, game.player.rect.centery)
        finally:
            pygame.key.get_pressed = real_keys
            ai_control.poll_ai_action = real_poll
        self.assertLess(top, water_top - 60,
                        "jump must breach 60px+ above the surface")

    def test_sword_hits_in_front_at_torso_height(self):
        # The swing must connect in a frontal arc at torso height
        # (past the old stubby low box) and never behind the back.
        from game.tiles import HOTBAR, SWORD
        from game.entities import Enemy
        game = self._make_game()
        game.slot = HOTBAR.index(SWORD)
        game.player.face = 1
        p = game.player.rect
        ahead = Enemy(p.right + 40, p.centery - 32, "slime")
        behind = Enemy(p.left - 40 - 64, p.centery - 32, "slime")
        game.enemies = [ahead, behind]
        game.attack()
        self.assertLess(ahead.hp, ahead.max_hp,
                        "frontal swing must connect past old reach")
        self.assertEqual(behind.hp, behind.max_hp,
                         "swing must not hit behind the player")
        game.draw_entities()  # mid-swing render must not crash

    def test_noon_sky_has_pixel_sun_and_shaded_clouds(self):
        # The noon sun is 16x16 blocky art: the near corner chunk is
        # filled (a smooth disc would leave it sky) while the far
        # grid corners stay cut; clouds carry shaded undersides.
        import pygame
        from game.utils import sky_and_daylight
        game = self._make_game()
        game.tod = 0.5
        _, _, game.daylight = sky_and_daylight(0.5)
        game.light.sunval = 255
        real_ticks = pygame.time.get_ticks
        pygame.time.get_ticks = lambda: 100000
        try:
            game.draw_world()
        finally:
            pygame.time.get_ticks = real_ticks
        bx, by = game.W // 2, int(game.H * 0.12)

        def at(x, y):
            return game.screen.get_at((x, y))[:3]

        def is_sun_warm(r, g, b):
            return r > 240 and g > 190 and b < 185

        self.assertTrue(is_sun_warm(*at(bx, by)),
                        "sun core should be warm at noon")
        self.assertTrue(is_sun_warm(*at(bx + 16, by + 16)),
                        "blocky 16-grid sun must fill the near corner")
        self.assertFalse(is_sun_warm(*at(bx + 31, by + 31)),
                         "blocky sun must cut the far grid corners")
        self.assertTrue(is_sun_warm(*at(bx + 18, by)),
                        "sun mid-band should be warm past the old edge")

        def band_has(pred):
            for x in range(0, game.W, 8):
                for y in range(0, 200, 4):
                    if pred(*at(x, y)):
                        return True
            return False

        self.assertTrue(
            band_has(lambda r, g, b: r > 230 and g > 235 and b > 230),
            "expected white cloud pixels in the sky band")
        self.assertTrue(
            band_has(lambda r, g, b: 150 < r < 195
                     and 175 < g < 215 and 200 < b < 240),
            "expected shaded cloud undersides in the sky band")

    def test_night_sky_has_moon_and_stars(self):
        # The moon and stars are light sources: they must punch
        # through the night lighting overlay, not be multiplied away.
        import pygame
        from game.utils import sky_and_daylight
        game = self._make_game()
        game.tod = 0.0
        _, _, game.daylight = sky_and_daylight(0.0)
        game.light.sunval = int(game.daylight * 255)
        real_ticks = pygame.time.get_ticks
        pygame.time.get_ticks = lambda: 100000
        try:
            game.draw_world()
        finally:
            pygame.time.get_ticks = real_ticks
        bx, by = game.W // 2, int(game.H * 0.12)

        def bright(x, y):
            r, g, b = game.screen.get_at((x, y))[:3]
            return r > 200 and g > 210 and b > 225

        moon = any(bright(x, y)
                   for x in range(bx - 30, bx + 31, 3)
                   for y in range(by - 30, by + 31, 3))
        self.assertTrue(moon, "moon should shine through the night")
        stars = any(bright(x, y)
                    for x in range(0, game.W, 3)
                    for y in range(0, 400, 3)
                    if abs(x - bx) > 60 or abs(y - by) > 60)
        self.assertTrue(stars, "stars should shine through the night")

    def test_celestial_art_fits_16_grid(self):
        # Sun, moon and clouds are authored on the 16x16 pixel grid
        # with three shades each.
        from game.sprites import SUN_MAP, CLOUD_MAPS, moon_surface
        self.assertEqual(len(SUN_MAP), 16)
        glyphs = set()
        for row in SUN_MAP:
            self.assertEqual(len(row), 16)
            glyphs.update(row)
        self.assertTrue({"Y", "M", "L"} <= glyphs,
                        f"sun needs 3 yellows, got {glyphs}")
        self.assertTrue(set("YML.") >= glyphs)
        self.assertEqual(len(CLOUD_MAPS), 3)
        for shape in CLOUD_MAPS:
            glyphs = set("".join(shape))
            self.assertTrue({"H", "W", "S"} <= glyphs,
                            f"cloud needs 3 whites, got {glyphs}")
            self.assertTrue(set("HWS.") >= glyphs)
        moon = moon_surface()
        self.assertEqual(moon.get_size(), (16, 16))
        shades = set()
        for x in range(16):
            for y in range(16):
                r, g, b, a = moon.get_at((x, y))
                if a:
                    shades.add((r, g, b))
        self.assertEqual(len(shades), 3,
                         f"moon needs 3 shades, got {shades}")

    def test_setting_sun_tucks_behind_hills(self):
        # Bodies punch through the lighting overlay but must still
        # set behind terrain: at tod 0.62 the sun's top clears the
        # hill crest while its center hides inside the hill.
        import math
        from game.utils import sky_and_daylight
        from game import ai_control
        game = self._make_game()
        real_poll = ai_control.poll_ai_action
        ai_control.poll_ai_action = ai_control.neutral_action
        try:
            for _ in range(90):
                game.update(1 / 60)
        finally:
            ai_control.poll_ai_action = real_poll
        game.tod = 0.62
        _, _, game.daylight = sky_and_daylight(0.62)
        game.light.sunval = int(game.daylight * 255)
        game.draw_world()
        angle = (0.62 - 0.25) * 2 * math.pi
        bx = int(game.W * 0.5 + math.cos(angle) * game.W * 0.42)
        by = int(game.H * 0.62 - abs(math.sin(angle)) * game.H * 0.5)

        def warm(x, y):
            r, g, b = game.screen.get_at((x, y))[:3]
            return r > 240 and g > 190

        self.assertTrue(warm(bx, by - 30),
                        "sun top should clear the crest")
        self.assertFalse(warm(bx, by),
                         "sun center should hide in the hill")

    def _flat_arena(self):
        from game.tiles import AIR, DIRT
        from game.world import World
        world = World()
        H = 30
        for x in range(60, 72):
            for y in range(H, H + 3):
                world.set(x, y, DIRT)
            for y in range(H):
                world.set(x, y, AIR)
        return world, H

    def test_bat_flies_toward_player(self):
        # Bats beat their wings to the player instead of plummeting.
        from game.entities import Player, Enemy
        from game.config import TILE
        world, H = self._flat_arena()
        player = Player(70 * TILE, H * TILE - 120)
        bat = Enemy(62 * TILE, (H - 4) * TILE, "bat")
        x0 = abs(bat.rect.centerx - player.rect.centerx)
        for _ in range(120):
            bat.update(world, player, 1 / 60)
        x1 = abs(bat.rect.centerx - player.rect.centerx)
        self.assertLess(x1, x0 - 40, "bat should close in")
        self.assertLess(bat.rect.bottom, H * TILE,
                        "bat must stay airborne, not land")

    def test_skeleton_outpaces_shade_chase(self):
        # Skeletons are relentless pogo-chasers: strictly faster
        # over ground than a shade from the same start.
        import random
        random.seed(7)
        from game.entities import Player, Enemy
        from game.config import TILE
        world, H = self._flat_arena()
        player = Player(70 * TILE, H * TILE - 120)
        sk = Enemy(62 * TILE, H * TILE - 120, "skeleton")
        sh = Enemy(62 * TILE, H * TILE - 128, "shade")
        for _ in range(120):
            sk.update(world, player, 1 / 60)
            sh.update(world, player, 1 / 60)
        self.assertGreater(sk.rect.x, sh.rect.x + 40,
                           "skeleton must outrun the shade")

    def test_spawns_mix_kinds_by_light(self):
        # Day brings slimes and skeletons, night shades and bats,
        # and the deep brings bats even by day.
        import random
        from game.config import TILE, WORLD_W, WORLD_H
        game = self._make_game()
        game.enemies = []
        random.seed(11)

        def force_spawns(n):
            kinds = set()
            for _ in range(n):
                game.spawn_timer = 0
                game.update_enemies(1 / 60)
                self.assertEqual(len(game.enemies), 1)
                kinds.add(game.enemies.pop().kind)
            return kinds

        game.daylight = 0.9
        self.assertEqual(force_spawns(30), {"slime", "skeleton"})
        game.daylight = 0.1
        self.assertEqual(force_spawns(30), {"shade", "bat"})
        game.player.rect.topleft = ((WORLD_W // 2) * TILE,
                                    (WORLD_H - 8) * TILE)
        game.daylight = 0.9
        deep = force_spawns(10)
        self.assertIn("bat", deep)
        self.assertTrue(deep <= {"shade", "bat"})

    def test_enemies_float_atop_water(self):
        # A slime dropped into the lake must bob up to the surface
        # instead of sitting on the lakebed.
        from game.world import World
        from game.entities import Player, Enemy
        from game.config import TILE, WORLD_W
        world = World()
        lakes = [x for x in range(100) if world.surface[x] > 40]
        lx = lakes[len(lakes) // 2]
        ly = world.surface[lx] - 2
        bed_top = (ly + 2) * TILE
        player = Player((lx + 20) * TILE, ly * TILE)
        slime = Enemy(lx * TILE + 8, (ly + 1) * TILE, "slime")
        slime.vel.update(0, 0)
        for _ in range(300):
            slime.update(world, player, 1 / 60)
        self.assertLess(slime.rect.top, bed_top - 64 - 40,
                        "slime should rise well off the lakebed")
        self.assertLess(slime.rect.top, ly * TILE + 10,
                        "slime head should reach the waterline")
        self.assertTrue(world.water_at_pixel(slime.rect.centerx,
                                             slime.rect.bottom - 4),
                        "slime feet stay wet while floating")

    def test_inventory_starts_empty(self):
        from game.tiles import HOTBAR, SWORD
        game = self._make_game()
        for item in HOTBAR:
            if item == SWORD:
                self.assertEqual(game.inv.get(item, 0), 1,
                                 "starter sword stays equipped")
            else:
                self.assertEqual(game.inv.get(item, 0), 0,
                                 f"slot {item} should start empty")

    def test_water_and_bedrock_are_not_mineable(self):
        # Clicking water (or bedrock) must not start the mining
        # animation: no target, no cracks, no progress.
        import pygame
        from game.config import TILE, WORLD_W, WORLD_H
        game = self._make_game()
        world = game.world
        lakes = [x for x in range(100) if world.surface[x] > 40]
        lx = lakes[len(lakes) // 2]
        ly = world.surface[lx] - 2
        self.assertTrue(world.wtc[ly * WORLD_W + lx])
        bx = WORLD_W // 2
        cases = (
            ((lx - 1) * TILE, ly * TILE, lx, ly),          # water
            (bx * TILE, (WORLD_H - 3) * TILE, bx, WORLD_H - 1),  # bedrock
        )
        real_pressed = pygame.mouse.get_pressed
        real_pos = pygame.mouse.get_pos
        try:
            for px, py, tx, ty in cases:
                game.player.rect.topleft = (px, py)
                mx = int(tx * TILE + TILE / 2 - game.cam.x)
                my = int(ty * TILE + TILE / 2 - game.cam.y)
                pygame.mouse.get_pressed = lambda: (True, False, False)
                pygame.mouse.get_pos = lambda mx=mx, my=my: (mx, my)
                game.update_mining(0.1)
                self.assertIsNone(
                    game.mine_target,
                    f"tile {(tx, ty)} must not be mineable")
                self.assertEqual(game.mine_prog, 0.0)
        finally:
            pygame.mouse.get_pressed = real_pressed
            pygame.mouse.get_pos = real_pos

    def test_attack_needs_sword_selected(self):
        # Clicking an enemy with a block selected must neither play
        # the swing nor deal invisible damage.
        from game.tiles import HOTBAR, SWORD, DIRT
        from game.entities import Enemy
        game = self._make_game()
        game.player.face = 1
        p = game.player.rect
        enemy = Enemy(p.right + 40, p.centery - 32, "slime")
        game.enemies = [enemy]
        game.slot = HOTBAR.index(DIRT)
        game.attack()
        self.assertEqual(enemy.hp, enemy.max_hp,
                         "blocks must not deal damage")
        self.assertEqual(game.player.swing, 0.0,
                         "no swing without the sword")
        game.slot = HOTBAR.index(SWORD)
        game.attack()
        self.assertLess(enemy.hp, enemy.max_hp,
                        "sword must still connect")

    def test_mined_blocks_land_in_hotbar_for_placement(self):
        # Every mined block (grass, ores, ...) must sit in the
        # hotbar and place back down — nothing strands in a
        # display-only category.
        import pygame
        from game.tiles import AIR, GRASS, COAL, HOTBAR, PLACEABLE
        from game.config import TILE, WORLD_W
        for item in (GRASS, COAL):
            self.assertIn(item, HOTBAR)
            self.assertIn(item, PLACEABLE)
        game = self._make_game()
        world = game.world
        tx = WORLD_W // 2
        ty = world.surface[tx] - 3
        world.set(tx, ty, AIR)
        game.inv[GRASS] = 2
        game.slot = HOTBAR.index(GRASS)
        real_pos = pygame.mouse.get_pos
        mx = int(tx * TILE + TILE / 2 - game.cam.x)
        my = int(ty * TILE + TILE / 2 - game.cam.y)
        pygame.mouse.get_pos = lambda mx=mx, my=my: (mx, my)
        try:
            game.place_selected_block()
        finally:
            pygame.mouse.get_pos = real_pos
        self.assertEqual(world.g[tx][ty], GRASS,
                         "mined grass must place back down")
        self.assertEqual(game.inv[GRASS], 1)
        game.draw_ui()  # full hotbar must draw without errors


if __name__ == "__main__":
    unittest.main()
