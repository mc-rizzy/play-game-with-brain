import math
import random

import pygame

from game.config import (
    TILE, WORLD_W, WORLD_H, DAY_LENGTH, GRAVITY, REACH,
    WINDOWED_SIZE, PLAYER_SPEED, PLAYER_SWIM_SPEED, PLAYER_JUMP,
    MAX_PARTICLES, UI_BASE_W, UI_BASE_H, UI_MIN_SCALE, UI_MAX_SCALE,
    SUBCELLS, DROPS_PER_SUB, LIGHT_LEVELS, WATER_TICK_DT,
)
from game.tiles import *
from game.sprites import (
    make_sprites, make_water_frames, make_entity_sprites,
    make_still_water,
    SUN_MAP, CLOUD_MAPS, paint_map, moon_surface,
)
from game.world import World, W2
from game.lighting import Lighting
from game.entities import Player, Enemy, move_collide
from game.utils import lerp_col, sky_and_daylight
from game import ai_control
from game.autopilot import Autopilot


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Cozy Blocks — a slice-of-life adventure")

        self.fullscreen = True
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.W, self.H = self.screen.get_size()

        self.clock = pygame.time.Clock()
        self.ui_scale = 1.0
        self.font = None
        self.big = None
        self.refresh_ui_scale()

        self.sprites = make_sprites()
        self.entity = make_entity_sprites()
        self.icons = {}
        self.refresh_icons()

        self.world = World()
        self.light = Lighting(self.world)
        self.light.compute(1.0)
        self.light.update_display(10.0)

        self.water_frames = make_water_frames()
        self.water_still = pygame.transform.scale(
            make_still_water(), (TILE, TILE))
        # Celestial art shares the tiles' 5px cells; the moon never
        # tints so it builds once, sun/clouds rebuild on light bands.
        self._moon_art = pygame.transform.scale(moon_surface(), (80, 80))
        self._sky_key = None
        self._sun_art = None
        self._cloud_near = []
        self._cloud_far = []
        self.visible = (0, 0, 0, 0)
        self._text_cache = {}
        self._swing_cache = None
        self._heart_cache = {}
        self._prev_ai_buttons = {"attack": False, "place": False,
                                 "torch": False, "craft": False}
        self.autopilot = Autopilot()
        self.autopilot_on = False  # P toggles; silent by design

        spawn_x = WORLD_W // 2
        self.spawn = (
            spawn_x * TILE,
            (self.world.surface[spawn_x] - 2) * TILE,
        )

        self.player = Player(*self.spawn)
        self.cam = pygame.Vector2(
            self.player.rect.centerx - self.W / 2,
            self.player.rect.centery - self.H / 2,
        )

        # Survival start: empty pockets, only the starter sword
        # equipped. Everything else must be mined or crafted.
        self.inv = {
            WOOD: 0,
            DIRT: 0,
            STONE: 0,
            COAL: 0,
            IRON: 0,
            GOLD: 0,
            SAND: 0,
            GRASS: 0,
            LEAVES: 0,
            TALLGRASS: 0,
            PLANKS: 0,
            BRICK: 0,
            GLASS: 0,
            TORCH: 0,
            LANTERN: 0,
            FLOWER: 0,
            SWORD: 1,
        }

        self.slot = HOTBAR.index(SWORD)
        self.craft_open = False

        self.tod = 0.32
        self.day = 1
        self.daylight = 1.0
        self.spawn_timer = 4.0

        self.enemies = []
        self.particles = []
        self.texts = []

        self.mine_target = None
        self.mine_prog = 0.0
        self._water_t = 0.0

        self.clouds = [
            (
                random.randrange(4000),
                random.randrange(170),
                random.uniform(6, 16),
            )
            for _ in range(12)
        ]
        self.stars = [
            (random.random(), random.random() * 0.6)
            for _ in range(130)
        ]

        self.craft_rects = []
        self.msg = "Welcome home. Gather wood, build, and light up the night!"
        self.msg_t = 6.0

        self.sky_surface = None
        self.sky_key = None
        self.light_tile_surface = None
        self.light_screen_surface = None
        self._light_cache_key = None
        self._light_palette = None

    # ------------------------------------------------------------ display

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen

        if self.fullscreen:
            self.screen = pygame.display.set_mode(
                (0, 0), pygame.FULLSCREEN
            )
        else:
            self.screen = pygame.display.set_mode(
                WINDOWED_SIZE, pygame.RESIZABLE
            )

        self.W, self.H = self.screen.get_size()
        self.sky_surface = None
        self.light_tile_surface = None
        self.light_screen_surface = None
        self._light_cache_key = None
        self.refresh_ui_scale()
        self.refresh_icons()

    def refresh_ui_scale(self):
        """Scale all GUI elements from the window size (1.0 at 1280x720)."""
        scale = min(self.W / UI_BASE_W, self.H / UI_BASE_H)
        self.ui_scale = max(UI_MIN_SCALE, min(UI_MAX_SCALE, scale))
        s = self.ui_scale
        # Built-in font reads as a game UI font; consolas looked like a
        # code editor. Sizes tuned so normal/big stay crisp at 1x-2x.
        self.font = pygame.font.Font(None, max(20, int(24 * s)))
        self.big = pygame.font.Font(None, max(28, int(34 * s)))
        self._text_cache = {}
        self._heart_cache = {}
        # Hearts crop the 10x9 core out of the 16x16 PNG, then scale by
        # an integer factor (2x at 1280x720, 3x at 1080p, ...).
        heart_mult = max(2, int(round(2 * s)))
        self.heart_size = (10 * heart_mult, 9 * heart_mult)
        # Integer multiples of the 16px art so icons stay crisp.
        icon_mult = max(2, int(round(38 * s / 16)))
        self.icon_size = 16 * icon_mult
        # Scaled hotbar metrics (Terraria-style top-left bar), shrunk
        # to clear the hearts panel on narrow windows.
        reserve = 10 * self.heart_size[0] + int(60 * s)
        fit = (self.W - int(14 * s) - reserve) // len(HOTBAR)
        self.slot_size = max(int(36 * s), min(int(54 * s), fit))
        self.slot_inner = self.slot_size - max(2, int(4 * s))

    def refresh_icons(self):
        if not hasattr(self, "icon_size"):
            return
        # Build icons from the 16px source art with integer scaling so
        # they stay crisp (downscaling the 80px tile sprites blurs).
        import random as _random
        from game.sprites import make_small_tile
        size = self.icon_size
        icons = {}
        for item in HOTBAR:
            if item == SWORD:
                continue
            rng = _random.Random(item * 1000)
            small = make_small_tile(item, rng)
            icons[item] = pygame.transform.scale(small, (size, size))
        sword = self.entity.get("sword")
        if sword is not None:
            try:
                icons[SWORD] = pygame.transform.scale(sword, (size, size))
            except Exception:
                icons[SWORD] = pygame.transform.scale(
                    self.sprites[WOOD][0], (size, size))
        self.icons = icons

    def text(self, content, colour, big=False):
        """Cached font rendering (avoids re-rendering every frame)."""
        key = (content, colour, big)
        cached = self._text_cache.get(key)
        if cached is None:
            font = self.big if big else self.font
            cached = font.render(content, True, colour)
            if len(self._text_cache) > 256:
                self._text_cache.clear()
            self._text_cache[key] = cached
        return cached

    # ------------------------------------------------------------ helpers

    def mouse_tile(self):
        mx, my = pygame.mouse.get_pos()
        return (
            int((mx + self.cam.x) // TILE),
            int((my + self.cam.y) // TILE),
        )

    def water_tile_animated(self, tx, ty):
        """Only SURFACE water the player wades through animates: a
        wet tile with no water above it that the player overlaps.
        Water under water, and water away from the player, is still."""
        if not (0 <= tx < WORLD_W and 0 <= ty < WORLD_H):
            return False
        wtc = self.world.wtc
        if not wtc[ty * WORLD_W + tx]:
            return False
        if ty > 0 and wtc[(ty - 1) * WORLD_W + tx]:
            return False
        r = self.player.rect
        return (r.left // TILE <= tx <= (r.right - 1) // TILE
                and r.top // TILE <= ty <= (r.bottom - 1) // TILE)

    def try_craft(self, recipe):
        name, result, count, cost = recipe

        if all(self.inv.get(k, 0) >= v for k, v in cost.items()):
            for k, v in cost.items():
                self.inv[k] -= v

            self.inv[result] = self.inv.get(result, 0) + count
            self.msg = f"Crafted {name}!"
            self.msg_t = 2.5

    def spawn_particles(self, x, y, colour, count=10):
        available = MAX_PARTICLES - len(self.particles)
        count = min(count, available)

        for _ in range(max(0, count)):
            self.particles.append(
                [
                    x,
                    y,
                    random.uniform(-110, 110),
                    random.uniform(-200, -30),
                    random.uniform(0.3, 0.7),
                    colour,
                ]
            )

    def attack(self):
        # Blocks don't deal damage: only the sword swings and hits.
        if HOTBAR[self.slot] != SWORD:
            return
        player = self.player
        player.swing = 0.22

        damage = 3
        # Frontal arc at torso height: a full tile of reach ahead.
        if player.face > 0:
            hx = player.rect.right
        else:
            hx = player.rect.left - 96
        hitbox = pygame.Rect(
            hx,
            player.rect.centery - 40,
            96,
            80,
        )

        for enemy in self.enemies:
            if hitbox.colliderect(enemy.rect):
                enemy.hp -= damage
                enemy.hurt_t = 0.25
                enemy.vel.x = player.face * 330
                enemy.vel.y = -330
                self.spawn_particles(
                    enemy.rect.centerx,
                    enemy.rect.centery,
                    (130, 235, 130),
                    8,
                )

    # ------------------------------------------------------------ update

    def update(self, dt):
        player = self.player
        world = self.world

        previous_tod = self.tod
        self.tod = (self.tod + dt / DAY_LENGTH) % 1.0

        if self.tod < previous_tod:
            self.day += 1

        _, _, self.daylight = sky_and_daylight(self.tod)

        # Sun intensity is continuous (render-side); the torch flood
        # advances one budgeted physics step every frame.
        self.light.compute(self.daylight)
        self.light.update_display(dt)

        keys = pygame.key.get_pressed()

        # Keyboard + AI merge so both work simultaneously.
        kb_move = (
            (keys[pygame.K_d] or keys[pygame.K_RIGHT])
            - (keys[pygame.K_a] or keys[pygame.K_LEFT])
        )
        kb_jump = bool(keys[pygame.K_SPACE] or keys[pygame.K_w])
        try:
            if self.autopilot_on:
                ai = self.autopilot.action(self, dt)
            else:
                ai = ai_control.poll_ai_action()
        except Exception:
            ai = {"move": 0.0, "jump": False, "attack": False,
                  "place": False, "torch": False, "craft": False,
                  "slot": None}
        merged = ai_control.merge_with_keyboard(
            {"move": kb_move, "jump": kb_jump}, ai)
        move = merged["move"]
        jump = merged["jump"]
        self._ai_discrete(ai)

        in_water = world.water_at_pixel(player.rect.centerx,
                                       player.rect.centery)

        player.vel.x = move * (
            PLAYER_SWIM_SPEED if in_water else PLAYER_SPEED
        )

        if move:
            player.face = 1 if move > 0 else -1
            player.walk += dt * 9

        if in_water:
            head_out = not world.water_at_pixel(
                player.rect.centerx, player.rect.top + 2)
            if jump:
                # Breach: head out means a full jump clean onto the
                # shore; fully submerged, swim up hard instead.
                player.vel.y = PLAYER_JUMP if head_out else -320
            player.vel.y += GRAVITY * 0.22 * dt
            player.vel.y = min(player.vel.y, 145)
        else:
            player.vel.y += GRAVITY * dt
            if jump and player.on_ground:
                player.vel.y = PLAYER_JUMP

        player.on_ground = move_collide(world, player.rect, player.vel, dt)
        player.hurt_t = max(0.0, player.hurt_t - dt)
        player.swing = max(0.0, player.swing - dt)

        if player.rect.top > WORLD_H * TILE:
            player.rect.topleft = self.spawn
            player.hp = player.max_hp

        self.update_mining(dt)
        self.update_enemies(dt)
        self.update_water(dt)

        for particle in list(self.particles):
            particle[0] += particle[2] * dt
            particle[1] += particle[3] * dt
            particle[3] += 620 * dt
            particle[4] -= dt

            if particle[4] <= 0:
                self.particles.remove(particle)

        for text in list(self.texts):
            text[1] -= 34 * dt
            text[3] -= dt

            if text[3] <= 0:
                self.texts.remove(text)

        self.msg_t = max(0.0, self.msg_t - dt)

        target = pygame.Vector2(
            player.rect.centerx - self.W / 2,
            player.rect.centery - self.H / 2,
        )
        self.cam += (target - self.cam) * min(1.0, dt * 6)
        self.cam.x = max(0, min(WORLD_W * TILE - self.W, self.cam.x))
        self.cam.y = max(0, min(WORLD_H * TILE - self.H, self.cam.y))

    def _ai_discrete(self, ai):
        """Edge-triggered AI buttons (attack/place/torch/craft/slot)."""
        prev = self._prev_ai_buttons
        if ai.get("attack") and not prev["attack"]:
            self.attack()
        if ai.get("place") and not prev["place"]:
            self.place_selected_block()
        if ai.get("torch") and not prev["torch"]:
            self.quick_torch()
        if ai.get("craft") and not prev["craft"]:
            self.craft_open = not self.craft_open
        slot = ai.get("slot")
        if isinstance(slot, int) and 0 <= slot < len(HOTBAR):
            self.slot = slot
        for key in prev:
            prev[key] = bool(ai.get(key))

    def run_calibration_wizard(self):
        """Embedded 15 s Calibration Wizard (C key).

        Replays local EEG trials when available, else demo data; adapts
        the GNN head to this player, asks Muse to tune the live gates,
        and saves adapter.pt for the live bridge. ESC skips.
        """
        import os
        try:
            from game.calibrate import run_wizard
            from gnn.recordings import find_trials
        except Exception as exc:
            self.msg = f"Calibration unavailable: {exc}"
            self.msg_t = 5.0
            return
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data = os.path.join(root, "EEG_recordings")
        out = os.path.join(root, "gnn", "checkpoints")
        model_path = os.path.join(out, "model.pt")
        try:
            source = "replay" if find_trials(data) else "demo"
        except Exception:
            source = "demo"
        try:
            result = run_wizard(
                self.screen, source=source,
                model_path=(model_path
                            if os.path.exists(model_path) else None),
                data=data, out_dir=out)
        except Exception as exc:
            self.msg = f"Calibration failed: {exc}"
            self.msg_t = 5.0
            return
        if result is None:
            self.msg = "Calibration skipped."
            self.msg_t = 4.0
        else:
            self.msg = f"Calibrated! {result['muse'].get('verdict', '')}"
            self.msg_t = 8.0
        self.clock.tick()  # drop wizard time so the next dt stays small

    def quick_torch(self):
        tile_x, tile_y = self.mouse_tile()
        if (
            self.inv.get(TORCH, 0) > 0
            and self.world.get(tile_x, tile_y) == AIR
        ):
            self.world.set(tile_x, tile_y, TORCH)
            self.inv[TORCH] -= 1

    def update_mining(self, dt):
        if self.craft_open or not pygame.mouse.get_pressed()[0]:
            self.mine_target = None
            self.mine_prog = 0.0
            return

        tile_x, tile_y = self.mouse_tile()
        target_centre = (tile_x * TILE + TILE // 2, tile_y * TILE + TILE // 2)

        if math.dist(self.player.rect.center, target_centre) > REACH:
            self.mine_target = None
            self.mine_prog = 0.0
            return

        tile = self.world.get(tile_x, tile_y)

        if tile == AIR:
            self.mine_target = None
            self.mine_prog = 0.0
            return

        if TILE_DEFS[tile][3] >= 1000:
            # Unmineable (water, bedrock): no target, no cracks, no
            # progress bar that could never finish.
            self.mine_target = None
            self.mine_prog = 0.0
            return

        if (tile_x, tile_y) != self.mine_target:
            self.mine_target = (tile_x, tile_y)
            self.mine_prog = 0.0

        self.mine_prog += dt
        hardness = TILE_DEFS[tile][3]

        if self.mine_prog >= hardness and hardness < 1000:
            self.world.set(tile_x, tile_y, AIR)
            self.inv[tile] = self.inv.get(tile, 0) + 1

            self.spawn_particles(
                tile_x * TILE + TILE // 2,
                tile_y * TILE + TILE // 2,
                TILE_DEFS[tile][4],
                10,
            )

            self.mine_target = None
            self.mine_prog = 0.0

    def update_water(self, dt):
        # Unconditional physics: the flow sim ticks on a fixed clock,
        # no sleep flags, no lazy wakeups.
        self._water_t -= dt
        if self._water_t > 0:
            return
        self._water_t = WATER_TICK_DT
        self.world.water_tick()

    def update_enemies(self, dt):
        player = self.player
        world = self.world

        self.spawn_timer -= dt
        night = self.daylight < 0.3

        if self.spawn_timer <= 0 and len(self.enemies) < 7:
            self.spawn_timer = 3.5

            side = random.choice((-1, 1))
            spawn_px = int(
                player.rect.centerx + side * random.uniform(650, 850)
            )
            tile_x = max(2, min(WORLD_W - 3, spawn_px // TILE))
            ptx = max(0, min(WORLD_W - 1,
                             player.rect.centerx // TILE))
            deep = (player.rect.centery // TILE
                    > world.surface[ptx] + 8)

            spawn_ty = world.surface[tile_x] - 2
            if night:
                kind = "bat" if random.random() < 0.4 else "shade"
            elif deep:
                kind = "bat" if random.random() < 0.7 else "shade"
                py = max(2, min(WORLD_H - 4,
                                player.rect.centery // TILE))
                for cand in range(py, 1, -1):
                    if (world.get(tile_x, cand) == AIR
                            and world.get(tile_x, cand + 1) == AIR):
                        spawn_ty = cand
                        break
            else:
                kind = ("skeleton" if random.random() < 0.3
                        else "slime")

            self.enemies.append(Enemy(
                tile_x * TILE,
                spawn_ty * TILE,
                kind,
            ))

        for enemy in list(self.enemies):
            enemy.update(world, player, dt)

            if enemy.hp <= 0:
                self.enemies.remove(enemy)
                drop = (COAL if enemy.kind in ("shade", "skeleton")
                        else FLOWER)
                self.inv[drop] = self.inv.get(drop, 0) + 1

                self.texts.append(
                    [
                        enemy.rect.centerx,
                        enemy.rect.y,
                        f"+1 {ITEM_NAMES[drop]}",
                        1.2,
                        (255, 235, 150),
                    ]
                )
                self.spawn_particles(
                    enemy.rect.centerx,
                    enemy.rect.centery,
                    (200, 255, 180),
                    16,
                )

            elif (
                enemy.rect.colliderect(player.rect)
                and enemy.touch_t <= 0
                and player.hurt_t <= 0
            ):
                player.hp -= 1
                player.hurt_t = 0.8
                enemy.touch_t = 1.0
                player.vel.x = math.copysign(
                    360, player.rect.centerx - enemy.rect.centerx
                )
                player.vel.y = -380

                if player.hp <= 0:
                    player.hp = player.max_hp
                    player.rect.topleft = self.spawn
                    self.msg = "You woke up at home, safe and sound."
                    self.msg_t = 4.0

        if (
            self.daylight > 0.5
            and player.hp < player.max_hp
            and random.random() < dt * 0.08
        ):
            player.hp += 1

    # ------------------------------------------------------------ drawing

    def get_sky_surface(self):
        top, bottom, _ = sky_and_daylight(self.tod)
        # Quantize so the gradient rebuilds a few times per minute,
        # not every frame, as tod interpolates continuously.
        top = tuple((c // 8) * 8 for c in top)
        bottom = tuple((c // 8) * 8 for c in bottom)
        key = (top, bottom, self.W, self.H)

        if self.sky_surface is not None and self.sky_key == key:
            return self.sky_surface

        surface = pygame.Surface((self.W, self.H))
        strip_height = 16

        for y in range(0, self.H, strip_height):
            colour = lerp_col(top, bottom, y / max(1, self.H))
            surface.fill(colour, (0, y, self.W, strip_height))

        try:
            surface = surface.convert()
        except pygame.error:
            pass
        self.sky_surface = surface
        self.sky_key = key
        return surface

    def draw_world(self):
        self.screen.blit(self.get_sky_surface(), (0, 0))
        # Clouds hang behind the terrain and take the night dimming
        # like everything else; the sun, moon and stars are light
        # sources, so they draw after the lighting overlay and punch
        # through the dark.
        self.draw_clouds()

        x0 = max(0, int(self.cam.x // TILE))
        y0 = max(0, int(self.cam.y // TILE))
        x1 = min(WORLD_W, x0 + self.W // TILE + 2)
        y1 = min(WORLD_H, y0 + self.H // TILE + 2)

        world = self.world
        sprites = self.sprites
        screen = self.screen
        camx, camy = self.cam.x, self.cam.y
        self.visible = (x0, y0, x1, y1)
        for ty in range(y0, y1):
            row_y = ty * TILE - camy
            for tx in range(x0, x1):
                tile = world.g[tx][ty]
                sx = int(tx * TILE - camx)
                sy = int(row_y)

                if tile == AIR:
                    # Air is air: dug-out cells blend with the sky
                    # background, and the lighting overlay shades them
                    # by depth like everything else. Wet cells read as
                    # AIR here; water draws over them later in
                    # draw_foreground.
                    continue

                screen.blit(sprites[tile][(tx + ty) & 1], (sx, sy))

        self.draw_mining_cracks()
        self.draw_lighting(x0, y0, x1, y1)
        self.draw_stars()
        self.draw_sun_and_moon()

    def draw_foreground(self):
        x0, y0, x1, y1 = self.visible
        frame = (pygame.time.get_ticks() // 220) % 4
        ws = self.world.ws
        wtc = self.world.wtc
        half = TILE // SUBCELLS
        half_max = SUBCELLS * DROPS_PER_SUB

        for ty in range(y0, y1):
            for tx in range(x0, x1):
                if not wtc[ty * WORLD_W + tx]:
                    continue
                if self.water_tile_animated(tx, ty):
                    # Phase offset per column for the water the
                    # player is moving through.
                    img = self.water_frames[(frame + tx) % 4]
                else:
                    img = self.water_still
                sx = int(tx * TILE - self.cam.x)
                sy = int(ty * TILE - self.cam.y)
                # Staircase halves: each half-column renders its drop
                # count bottom-aligned (~1.5px steps, buttery smooth).
                top = (ty * SUBCELLS) * W2 + tx * SUBCELLS
                bot = top + W2
                for dx in range(SUBCELLS):
                    drops = ws[top + dx] + ws[bot + dx]
                    if not drops:
                        continue
                    h = max(1, drops * TILE // half_max)
                    self.screen.blit(img, (sx + dx * half, sy + TILE - h),
                                     (dx * half, TILE - h, half, h))

    def _refresh_sky_art(self):
        """Rebuild tinted sun/cloud art when the light band changes."""
        day = self.daylight
        angle = (self.tod - 0.25) * 2 * math.pi
        low = max(0.0, min(1.0, 1.0 - math.sin(angle) * 1.6))
        key = (int(day * 16), int(low * 6))
        if key == self._sky_key:
            return
        self._sky_key = key
        sun_pal = {
            "Y": lerp_col((255, 140, 60), (255, 178, 66), 1.0 - low),
            "M": lerp_col((255, 172, 84), (255, 210, 98), 1.0 - low),
            "L": lerp_col((255, 208, 130), (255, 240, 170), 1.0 - low),
        }
        self._sun_art = pygame.transform.scale(
            paint_map(SUN_MAP, sun_pal), (80, 80))
        dusk = max(0.0, 1.0 - abs(day - 0.35) / 0.3)
        hi = lerp_col((64, 70, 104), (255, 255, 255), day)
        body = lerp_col((38, 42, 66), (242, 246, 252), day)
        body = lerp_col((250, 220, 200), body, 1.0 - dusk * 0.25)
        shade = lerp_col((16, 20, 36), (172, 194, 224), day)
        shade = lerp_col((236, 172, 142), shade, 1.0 - dusk * 0.55)
        pal = {"H": hi, "W": body, "S": shade}
        far_pal = {k: lerp_col(shade, v, 0.7) for k, v in pal.items()}
        self._cloud_near, self._cloud_far = [], []
        for shape in CLOUD_MAPS:
            small = paint_map(shape, pal)
            w, h = small.get_size()
            self._cloud_near.append(
                pygame.transform.scale(small, (w * 5, h * 5)))
            small = paint_map(shape, far_pal)
            w, h = small.get_size()
            self._cloud_far.append(
                pygame.transform.scale(small, (w * 3, h * 3)))

    def draw_stars(self):
        if self.daylight >= 0.45:
            return
        now = pygame.time.get_ticks() // 350
        for i, (sx, sy) in enumerate(self.stars):
            size = 2 if i % 3 == 0 else 3
            if (i * 13 + now) % 5 == 0:
                col = (130, 140, 180)
            else:
                col = (225, 232, 255)
            self.screen.fill(
                col, (sx * self.W, sy * self.H, size, size))

    def draw_sun_and_moon(self):
        angle = (self.tod - 0.25) * 2 * math.pi
        for kind, body_angle in (("sun", angle),
                                 ("moon", angle + math.pi)):
            if math.sin(body_angle) <= -0.15:
                continue
            bx = int(self.W * 0.5
                     + math.cos(body_angle) * self.W * 0.42)
            by = int(self.H * 0.62
                     - abs(math.sin(body_angle)) * self.H * 0.5)
            if kind == "sun":
                self._draw_sun(bx, by, body_angle)
            else:
                self._draw_moon(bx, by)

    def _sky_open(self, sx, sy):
        """True when a screen pixel looks at open sky, not terrain."""
        tx = int((sx + self.cam.x) // TILE)
        ty = int((sy + self.cam.y) // TILE)
        if ty < 0:
            return True
        if not (0 <= tx < WORLD_W and 0 <= ty < WORLD_H):
            return False
        t = self.world.g[tx][ty]
        return t == AIR or not TILE_DEFS[t][1]

    def _blit_sky_clipped(self, surf, cx, cy, cell, flags=0):
        """Blit centred art cell by cell, skipping cells behind
        terrain so bodies set and rise behind hills."""
        w, h = surf.get_size()
        ox, oy = cx - w // 2, cy - h // 2
        for gy in range(0, h, cell):
            for gx in range(0, w, cell):
                if self._sky_open(ox + gx + cell // 2,
                                  oy + gy + cell // 2):
                    self.screen.blit(
                        surf, (ox + gx, oy + gy),
                        (gx, gy, min(cell, w - gx), min(cell, h - gy)),
                        special_flags=flags)

    def _draw_sun(self, bx, by, body_angle):
        # Low sun burns bigger and oranger; noon sun is tight, warm.
        low = max(0.0, min(1.0, 1.0 - math.sin(body_angle) * 1.6))
        halo_col = lerp_col((255, 150, 70), (255, 238, 200),
                             1.0 - low)
        for size, alpha in ((140 + int(low * 60), 24 + int(low * 30)),
                            (96 + int(low * 30), 32 + int(low * 34))):
            # Premultiplied add-glow: BLEND_ADD ignores per-pixel
            # alpha, so the falloff is baked into the colour.
            k = alpha / 255
            halo = pygame.Surface((size, size))
            halo.fill((int(halo_col[0] * k), int(halo_col[1] * k),
                       int(halo_col[2] * k)))
            self._blit_sky_clipped(halo, bx, by, 14,
                                   flags=pygame.BLEND_ADD)
        self._blit_sky_clipped(self._sun_art, bx, by, 5)

    def _draw_moon(self, bx, by):
        halo = pygame.Surface((96, 96))
        halo.fill((18, 19, 23))  # premultiplied faint moonglow
        self._blit_sky_clipped(halo, bx, by, 14,
                               flags=pygame.BLEND_ADD)
        self._blit_sky_clipped(self._moon_art, bx, by, 5)

    def draw_clouds(self):
        self._refresh_sky_art()
        now = pygame.time.get_ticks() * 0.001
        for i, (cloud_x, cloud_y, speed) in enumerate(self.clouds):
            near = i % 2 == 0
            art = self._cloud_near if near else self._cloud_far
            par = 0.24 if near else 0.14
            x = int((cloud_x + now * speed - self.cam.x * par)
                    % (self.W + 280) - 140)
            self.screen.blit(art[i % len(art)], (x, cloud_y))

    def draw_mining_cracks(self):
        if not self.mine_target:
            return

        tx, ty = self.mine_target
        tile = self.world.get(tx, ty)

        if tile == AIR:
            return

        sx = int(tx * TILE - self.cam.x)
        sy = int(ty * TILE - self.cam.y)
        hardness = max(0.01, TILE_DEFS[tile][3])
        stages = min(4, int(self.mine_prog / hardness * 5))
        # Pixel cracks: chunky stepped blocks + corner progress pips.
        cell = max(4, TILE // 10)
        crack_col = (28, 24, 24)
        for i in range(stages + 1):
            ox = (i * 37) % (TILE - cell * 2)
            oy = (i * 53) % (TILE - cell * 3)
            self.screen.fill(crack_col,
                             (sx + ox, sy + oy, cell, cell))
            self.screen.fill(crack_col,
                             (sx + ox + cell // 2, sy + oy + cell,
                              cell, cell))
        # Progress pips along the bottom edge.
        pip_w = TILE // 5
        for i in range(5):
            col = (255, 220, 120) if i <= stages else (40, 36, 44)
            self.screen.fill(col, (sx + i * pip_w + 2, sy + TILE - 6,
                                   pip_w - 4, 4))

    def draw_lighting(self, x0, y0, x1, y1):
        # Retro pixel lighting at 2x2 sub-cells per block: hard-edged
        # half-tile blocks in 64 bands, nearest upscaled (no blending).
        # Combines analytic sun (scaled by daylight) with torch flood.
        width = x1 - x0
        height = y1 - y0
        if width <= 0 or height <= 0:
            return
        sw, sh = width * SUBCELLS, height * SUBCELLS

        if (
            self.light_tile_surface is None
            or self.light_tile_surface.get_size() != (sw, sh)
        ):
            self.light_tile_surface = pygame.Surface((sw, sh))
            self.light_screen_surface = pygame.Surface(
                (width * TILE, height * TILE))
            self._light_cache_key = None

        # Rebuild only when the sim changed, the sun band stepped, or we
        # scrolled to new tiles; sub-tile camera shifts just move the blit.
        sun_q = self.light.sunval >> 2
        cache_key = (x0, y0, width, height, self.light.version,
                     self.light.display_version, sun_q)
        if cache_key != self._light_cache_key:
            # 64-step palette (minimums prevent pure black).
            if self._light_palette is None:
                steps = LIGHT_LEVELS - 1
                self._light_palette = (
                    [18 + (lv * 237) // steps
                     for lv in range(LIGHT_LEVELS)],
                    [22 + (lv * 233) // steps
                     for lv in range(LIGHT_LEVELS)],
                )
            pal_w, pal_b = self._light_palette
            pixels = pygame.PixelArray(self.light_tile_surface)

            sun = self.light.sun
            lr, lg, lb = self.light.r, self.light.g, self.light.b
            sv = sun_q << 2
            top = LIGHT_LEVELS - 1
            shift = 8 - (LIGHT_LEVELS.bit_length() - 1)
            cap = top << shift

            for ly in range(sh):
                sy = (y0 + ly // SUBCELLS) * SUBCELLS + ly % SUBCELLS
                row = sy * W2
                sx0 = x0 * SUBCELLS
                for lx in range(sw):
                    sx = sx0 + lx
                    i = row + sx
                    # Same [1,2,1] lateral blend as sun_smooth().
                    c = sun[i]
                    left = sun[i - 1] if sx > 0 else c
                    right = sun[i + 1] if sx < W2 - 1 else c
                    s = (((left + (c << 1) + right) >> 2) * sv) // 255
                    r = s + lr[i]
                    g = s + lg[i]
                    b = (s * 96) // 100 + lb[i]
                    pixels[lx, ly] = (
                        pal_w[top if r > cap else r >> shift],
                        pal_w[top if g > cap else g >> shift],
                        pal_b[top if b > cap else b >> shift],
                    )

            del pixels

            # Nearest upscale: crisp hard-edged light cells that match
            # the pixel art (much faster than smoothscale).
            pygame.transform.scale(
                self.light_tile_surface,
                self.light_screen_surface.get_size(),
                self.light_screen_surface,
            )
            self._light_cache_key = cache_key

        self.screen.blit(
            self.light_screen_surface,
            (
                int(x0 * TILE - self.cam.x),
                int(y0 * TILE - self.cam.y),
            ),
            special_flags=pygame.BLEND_MULT,
        )

        self.draw_fireflies()

    def draw_fireflies(self):
        if self.daylight >= 0.35:
            return

        now = pygame.time.get_ticks()

        for i in range(12):
            x = int((i * 191 + now * 0.025 * (1 + i % 3)) % self.W)
            y = int(
                self.H * 0.55
                + 70 * math.sin(now * 0.001 + i * 2.1)
            )
            pygame.draw.rect(self.screen, (188, 255, 145), (x, y, 4, 4))

    def draw_entities(self):
        p = self.player
        screen = self.screen
        camx, camy = self.cam.x, self.cam.y
        ent = self.entity
        now = pygame.time.get_ticks()

        for e in self.enemies:
            x = int(e.rect.x - camx)
            y = int(e.rect.y - camy)
            w = e.rect.width
            frame = ((now // 350) + (x // 97)) % 2
            base = e.kind  # slime/shade/bat/skeleton art keys match
            key = f"{base}_{frame}"
            if e.hurt_t > 0:
                key += "_flash"
            img = ent.get(key)
            if img is not None:
                screen.blit(img, (x + (w - img.get_width()) // 2,
                                  y + e.rect.height - img.get_height()))
            else:  # placeholder fallback (assets missing)
                screen.fill((200, 80, 200), (x, y, w, e.rect.height))

            bar = max(5, TILE // 14)
            bx, by = x, y - bar - 4
            screen.fill((12, 12, 16), (bx - 1, by - 1, w + 2, bar + 2))
            screen.fill((58, 20, 20), (bx, by, w, bar))
            fill = int(w * max(0, e.hp) / e.max_hp)
            screen.fill((235, 80, 80), (bx, by, fill, bar))
            if fill > 2:
                screen.fill((255, 150, 150), (bx, by, fill, max(1, bar // 3)))

        # Player pixel art: idle / walk frames, flipped by facing.
        x = int(p.rect.x - camx)
        y = int(p.rect.y - camy)
        walking = abs(p.vel.x) > 5 and p.on_ground
        if walking:
            key = ("player_walk1" if int(p.walk) % 2 == 0
                   else "player_walk2")
        else:
            key = "player_idle"
        if p.face < 0:
            key += "_left"
        if p.hurt_t > 0 and int(p.hurt_t * 20) % 2 == 0:
            key += "_flash"
        img = ent.get(key) or ent.get("player_idle")
        if img is not None:
            # Centered + bottom-aligned: art is wider than the hitbox
            # and feet sit exactly on the rect bottom (the ground).
            screen.blit(img, (x + (p.rect.width - img.get_width()) // 2,
                              y + p.rect.height - img.get_height()))

        # Sword swing: the blade orbits the hand through a frontal
        # arc at torso height (matching the hitbox), trailing a white
        # slash streak. The hilt stays at the hand and the tip reach
        # (~90px) matches the 96px hitbox depth.
        if p.swing > 0 and HOTBAR[self.slot] == SWORD:
            sword = ent.get("sword")
            if sword is not None:
                if self._swing_cache is None:
                    self._swing_cache = pygame.transform.scale(
                        sword, (96, 96))
                fore = 1 if p.face > 0 else -1
                progress = 1.0 - p.swing / 0.22  # 0 -> 1
                sw = math.radians(-70 + 140 * progress)
                hand_x = x + (p.rect.width if p.face > 0 else 0)
                hand_y = y + int(p.rect.height * 0.45)
                reach = 48
                dx, dy = fore * math.cos(sw), math.sin(sw)
                bx = hand_x + dx * reach
                by = hand_y + dy * reach
                # Blade points along the radius, like an arm extension.
                # The art's tip sits 30.3 deg clockwise from straight
                # up, so subtract the native tilt to land on target.
                phi = math.degrees(math.atan2(dx, -dy))
                blade = pygame.transform.rotate(
                    self._swing_cache, 30.3 - phi)
                screen.blit(blade, (int(bx - blade.get_width() // 2),
                                    int(by - blade.get_height() // 2)))
                # Slash streak: fading ticks along the swept arc.
                for k, col in ((1, (240, 244, 252)),
                               (2, (200, 210, 228)),
                               (3, (160, 170, 190))):
                    ta = sw - k * 0.16
                    if ta < math.radians(-70):
                        continue
                    tx = hand_x + fore * math.cos(ta) * reach
                    ty = hand_y + math.sin(ta) * reach
                    screen.fill(col, (int(tx - 3), int(ty - 3), 6, 6))

        for pt in self.particles:
            screen.fill(pt[5], (int(pt[0] - camx), int(pt[1] - camy), 6, 6))
        for t in self.texts:
            screen.blit(self.text(t[2], t[4]),
                        (int(t[0] - camx), int(t[1] - camy)))

    # ------------------------------------------------------------ UI

    def heart_icon(self, full):
        key = ("full" if full else "empty", self.heart_size)
        cached = self._heart_cache.get(key)
        if cached is None:
            src = self.entity.get(
                "heart_full" if full else "heart_empty")
            # Crop the 10x9 heart core out of the 16x16 PNG (3px padding
            # on each side) so the icon scales by an integer factor.
            try:
                core = src.subsurface(pygame.Rect(3, 3, 10, 9))
            except Exception:
                core = src
            cached = pygame.transform.scale(core, self.heart_size)
            self._heart_cache[key] = cached
        return cached

    def draw_panel(self, rect, bg=(16, 20, 32, 220), border=(0, 0, 0),
                   bevel_light=(90, 100, 128), bevel_dark=(8, 10, 18)):
        """Pixel-style panel: flat bg, black border, 1px bevel."""
        s = self.ui_scale
        b = max(1, int(2 * s))
        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        panel.fill(bg)
        w, h = rect.width, rect.height
        # Bevel: light top/left, dark bottom/right.
        panel.fill(bevel_light, (0, 0, w, b))
        panel.fill(bevel_light, (0, 0, b, h))
        panel.fill(bevel_dark, (0, h - b, w, b))
        panel.fill(bevel_dark, (w - b, 0, b, h))
        pygame.draw.rect(panel, border, panel.get_rect(), max(1, b // 2))
        self.screen.blit(panel, rect.topleft)
        return rect

    def blit_text_shadow(self, content, colour, pos, big=False,
                         shadow=(10, 10, 16)):
        s = self.ui_scale
        off = max(1, int(2 * s))
        self.screen.blit(self.text(content, shadow, big=big),
                         (pos[0] + off, pos[1] + off))
        self.screen.blit(self.text(content, colour, big=big), pos)

    def draw_ui(self):
        # Terraria layout: inventory top-left, hearts top-right.
        self.draw_hotbar()
        self.draw_hearts_and_clock()
        self.draw_message()
        self.draw_help()

        if self.craft_open:
            self.draw_craft()

    def draw_hearts_and_clock(self):
        player = self.player
        s = self.ui_scale
        margin = int(14 * s)

        hw, hh = self.heart_size
        gap = max(3, int(5 * s))
        total_w = player.max_hp * hw + (player.max_hp - 1) * gap
        # Panel behind the hearts so they read over bright skies.
        pad = int(8 * s)
        panel = pygame.Rect(
            self.W - margin - total_w - pad * 2,
            margin,
            total_w + pad * 2,
            hh + pad * 2,
        )
        self.draw_panel(panel)
        x = panel.x + pad
        y = panel.y + pad
        for i in range(player.max_hp):
            self.screen.blit(self.heart_icon(i < player.hp),
                             (x + i * (hw + gap), y))

        hours = int(((self.tod - 0.25) % 1.0) * 24)
        minutes = int((self.tod * 1440) % 60)
        is_day = self.daylight > 0.4
        clock_text = f"Day {self.day}  {hours:02d}:{minutes:02d}"
        icon_col = (255, 214, 90) if is_day else (190, 200, 230)
        label = self.text(clock_text, (245, 245, 235))
        icon_side = max(8, int(14 * s))
        clock_pad = int(8 * s)
        clock_w = icon_side + int(8 * s) + label.get_width() + clock_pad * 2
        clock_h = max(label.get_height(), icon_side) + clock_pad * 2
        clock = pygame.Rect(
            self.W - margin - clock_w,
            panel.bottom + int(8 * s),
            clock_w,
            clock_h,
        )
        self.draw_panel(clock)
        ix = clock.x + clock_pad
        iy = clock.y + (clock_h - icon_side) // 2
        # Pixel sun/moon icon: solid square with a darker core.
        self.screen.fill(icon_col, (ix, iy, icon_side, icon_side))
        inner = max(2, icon_side // 3)
        inset = (icon_side - inner) // 2
        self.screen.fill((40, 40, 50) if not is_day else (255, 240, 180),
                         (ix + inset, iy + inset, inner, inner))
        self.screen.blit(label, (ix + icon_side + int(8 * s),
                                 clock.y + (clock_h - label.get_height()) // 2))

    def draw_message(self):
        if self.msg_t <= 0:
            return
        s = self.ui_scale
        text = self.text(self.msg, (255, 248, 220), big=True)
        pad_x, pad_y = int(14 * s), int(9 * s)
        box = pygame.Rect(
            int(self.W / 2 - text.get_width() / 2 - pad_x),
            int(86 * s),
            text.get_width() + pad_x * 2,
            text.get_height() + pad_y * 2,
        )
        self.draw_panel(box, bg=(18, 22, 36, 230),
                        border=(255, 210, 130))
        self.screen.blit(
            text, (int(self.W / 2 - text.get_width() / 2),
                   box.y + pad_y),
        )

    def draw_help(self):
        s = self.ui_scale
        tile_x, tile_y = self.mouse_tile()
        hovered = self.world.get(tile_x, tile_y)
        selected = (
            TILE_DEFS[hovered][0]
            if hovered != AIR
            else ITEM_NAMES[HOTBAR[self.slot]]
        )
        help_text = (
            f"{selected}   |   E: Craft   F: Torch   C: Calibrate brain   "
            f"F11: Fullscreen   LMB: Mine/Attack   RMB: Place"
        )
        rendered = self.text(help_text, (228, 234, 244))
        pad_x, pad_y = int(10 * s), int(5 * s)
        box = pygame.Rect(
            int(self.W / 2 - rendered.get_width() / 2 - pad_x),
            self.H - rendered.get_height() - pad_y * 2 - int(10 * s),
            rendered.get_width() + pad_x * 2,
            rendered.get_height() + pad_y * 2,
        )
        self.draw_panel(box, bg=(10, 14, 24, 170))
        self.screen.blit(rendered, (box.x + pad_x, box.y + pad_y))

    def draw_hotbar(self):
        # Terraria-style: inventory bar pinned to the top-left.
        s = self.ui_scale
        slot_size = self.slot_size
        inner = self.slot_inner
        margin = int(14 * s)
        start_x = margin
        y = margin
        pad = (inner - self.icon_size) // 2
        bw = max(2, int(3 * s))

        for i, item in enumerate(HOTBAR):
            rect = pygame.Rect(
                start_x + i * slot_size, y, inner, inner
            )
            selected = (i == self.slot)

            # Slot body + bevel.
            self.screen.fill((16, 20, 32), rect)
            b = max(1, int(2 * s))
            self.screen.fill((90, 100, 128), (rect.x, rect.y, rect.w, b))
            self.screen.fill((90, 100, 128), (rect.x, rect.y, b, rect.h))
            self.screen.fill((8, 10, 18),
                             (rect.x, rect.bottom - b, rect.w, b))
            self.screen.fill((8, 10, 18),
                             (rect.right - b, rect.y, b, rect.h))
            if selected:
                # Gold selection frame + subtle bright wash.
                wash = pygame.Surface((inner, inner), pygame.SRCALPHA)
                wash.fill((255, 220, 120, 28))
                self.screen.blit(wash, rect.topleft)
                pygame.draw.rect(self.screen, (255, 220, 120), rect, bw)
                # Corner ticks make the selection read at a glance.
                tick = max(4, int(8 * s))
                for cx, cy in ((rect.x, rect.y),
                               (rect.right - tick, rect.y),
                               (rect.x, rect.bottom - tick),
                               (rect.right - tick, rect.bottom - tick)):
                    self.screen.fill((255, 240, 190), (cx, cy, tick, max(2, b)))
            else:
                pygame.draw.rect(self.screen, (70, 76, 96), rect, 1)

            count = self.inv.get(item, 0)
            icon = self.icons.get(item)
            # Empty slots stay empty; the equipped sword always shows.
            if icon is not None and (item == SWORD or count > 0):
                self.screen.blit(icon, (rect.x + pad, rect.y + pad))

            if item != SWORD and count:
                self.blit_text_shadow(
                    str(count), (255, 255, 255),
                    (rect.right - self.text(str(count), (255, 255, 255)).get_width() - int(5 * s),
                     rect.bottom - self.text(str(count), (255, 255, 255)).get_height() - int(3 * s)))

            number = self.text(str(i + 1), (190, 200, 215))
            self.screen.blit(number, (rect.x + int(4 * s),
                                      rect.y + int(2 * s)))

        # Selected-item label under the bar.
        label = ITEM_NAMES[HOTBAR[self.slot]]
        rendered = self.text(label, (255, 240, 200))
        pad_x, pad_y = int(10 * s), int(4 * s)
        box = pygame.Rect(
            margin, y + inner + int(8 * s),
            rendered.get_width() + pad_x * 2,
            rendered.get_height() + pad_y * 2,
        )
        self.draw_panel(box, bg=(10, 14, 24, 190))
        self.screen.blit(rendered, (box.x + pad_x, box.y + pad_y))

    def draw_craft(self):
        s = self.ui_scale
        row_h = int(44 * s)
        step = int(52 * s)
        footer_h = int(34 * s)
        title_h = int(62 * s)
        panel_w = min(self.W - int(40 * s), int(560 * s))
        panel_h = min(self.H - int(40 * s),
                      title_h + len(RECIPES) * step + footer_h)
        panel = pygame.Rect(
            int(self.W / 2 - panel_w / 2),
            int(self.H / 2 - panel_h / 2),
            panel_w,
            panel_h,
        )

        self.draw_panel(panel, bg=(22, 26, 42, 242),
                        border=(255, 210, 130))
        self.blit_text_shadow("Crafting", (255, 225, 160),
                              (panel.x + int(24 * s), panel.y + int(16 * s)),
                              big=True)

        self.craft_rects = []
        y = panel.y + title_h

        for recipe in RECIPES:
            name, result, count, cost = recipe
            can_craft = all(
                self.inv.get(k, 0) >= v for k, v in cost.items()
            )

            rect = pygame.Rect(
                panel.x + int(22 * s), y,
                panel.w - int(44 * s), row_h,
            )
            # Square pixel rows with a left status tick (green/gray).
            self.screen.fill(
                (44, 56, 80) if can_craft else (32, 36, 50), rect)
            pygame.draw.rect(
                self.screen,
                (120, 200, 140) if can_craft else (70, 74, 90), rect, 1)
            self.screen.fill((120, 200, 140) if can_craft else (60, 64, 78),
                             (rect.x, rect.y, max(3, int(5 * s)), rect.h))

            cost_text = ", ".join(
                f"{ITEM_NAMES[k]} x{v} ({self.inv.get(k, 0)})"
                for k, v in cost.items()
            )
            text = self.text(
                f"{name} - {cost_text}",
                (235, 240, 255) if can_craft else (128, 136, 154),
            )
            self.screen.blit(text, (rect.x + int(14 * s),
                                    rect.y + (row_h - text.get_height()) // 2))

            self.craft_rects.append((rect, recipe))
            y += step

        self.screen.blit(
            self.text("Click a recipe to craft - E/Esc to close",
                      (180, 190, 205)),
            (panel.x + int(24 * s), panel.bottom - footer_h + int(6 * s)),
        )

    # ------------------------------------------------------------ events

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.VIDEORESIZE:
                self.W, self.H = event.w, event.h
                self.sky_surface = None
                self.light_tile_surface = None
                self.light_screen_surface = None
                self._light_cache_key = None
                self.refresh_ui_scale()
                self.refresh_icons()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    self.toggle_fullscreen()

                elif event.key == pygame.K_ESCAPE:
                    if self.craft_open:
                        self.craft_open = False
                    else:
                        return False

                elif event.key == pygame.K_e:
                    self.craft_open = not self.craft_open

                elif pygame.K_1 <= event.key <= pygame.K_9:
                    self.slot = event.key - pygame.K_1

                elif event.key == pygame.K_0:
                    self.slot = 9

                elif event.key == pygame.K_f:
                    self.quick_torch()

                elif event.key == pygame.K_p:
                    self.autopilot_on = not self.autopilot_on

                elif event.key == pygame.K_c:
                    self.run_calibration_wizard()

            elif event.type == pygame.MOUSEWHEEL:
                self.slot = (self.slot - event.y) % len(HOTBAR)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.craft_open and event.button == 1:
                    for rect, recipe in self.craft_rects:
                        if rect.collidepoint(event.pos):
                            self.try_craft(recipe)

                elif event.button == 1:
                    self.attack()

                elif event.button == 3 and not self.craft_open:
                    self.place_selected_block()

        return True

    def place_selected_block(self):
        tile_x, tile_y = self.mouse_tile()
        item = HOTBAR[self.slot]
        player = self.player

        if item not in PLACEABLE:
            return
        if self.inv.get(item, 0) <= 0:
            return
        if self.world.get(tile_x, tile_y) not in (AIR, WATER):
            return

        distance = math.dist(
            player.rect.center,
            (tile_x * TILE + TILE // 2, tile_y * TILE + TILE // 2),
        )
        if distance > REACH:
            return

        block_rect = pygame.Rect(
            tile_x * TILE, tile_y * TILE, TILE, TILE
        )
        if block_rect.colliderect(player.rect.inflate(-7, -5)):
            return

        self.world.set(tile_x, tile_y, item)
        self.inv[item] -= 1

    # ------------------------------------------------------------ main

    def run(self):
        running = True

        while running:
            dt = min(self.clock.tick(60) / 1000.0, 0.05)

            running = self.handle_events()
            self.update(dt)

            self.draw_world()
            self.draw_entities()
            self.draw_foreground()   # water washes over the player
            self.draw_ui()

            pygame.display.flip()

        pygame.quit()