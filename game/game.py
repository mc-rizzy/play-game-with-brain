import math
import random

import pygame

from game.config import (
    TILE, WORLD_W, WORLD_H, DAY_LENGTH, GRAVITY, REACH,
    WINDOWED_SIZE, PLAYER_SPEED, PLAYER_SWIM_SPEED, PLAYER_JUMP,
    MAX_PARTICLES, UI_BASE_W, UI_BASE_H, UI_MIN_SCALE, UI_MAX_SCALE,
)
from game.tiles import *
from game.sprites import make_sprites, make_water_frames, make_entity_sprites
from game.world import World
from game.lighting import Lighting
from game.entities import Player, Enemy, move_collide
from game.utils import lerp_col, sky_and_daylight
from game import ai_control


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
        self.visible = (0, 0, 0, 0)
        self._text_cache = {}
        self._swing_cache = None
        self._heart_cache = {}
        self._prev_ai_buttons = {"attack": False, "place": False,
                                 "torch": False, "craft": False}

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

        self.inv = {
            WOOD: 5,
            DIRT: 0,
            STONE: 0,
            COAL: 0,
            IRON: 0,
            GOLD: 0,
            SAND: 0,
            PLANKS: 4,
            BRICK: 0,
            GLASS: 0,
            TORCH: 3,
            LANTERN: 0,
            FLOWER: 0,
            SWORD: 1,
        }

        self.slot = 6
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
        self.font = pygame.font.SysFont("consolas", max(17, int(17 * s)))
        self.big = pygame.font.SysFont("consolas", max(26, int(26 * s)),
                                       bold=True)
        self._text_cache = {}
        self._heart_cache = {}
        # Scaled hotbar metrics.
        self.slot_size = int(54 * s)
        self.slot_inner = int(50 * s)
        self.icon_size = int(38 * s)
        heart_w = max(20, int(20 * s))
        self.heart_size = (heart_w, max(18, int(heart_w * 0.9)))

    def refresh_icons(self):
        if not hasattr(self, "icon_size"):
            return
        size = self.icon_size
        self.icons = {
            item: pygame.transform.scale(self.sprites[item][0], (size, size))
            for item in HOTBAR
            if item != SWORD
        }
        sword = self.entity.get("sword")
        if sword is not None:
            self.icons[SWORD] = pygame.transform.scale(sword, (size, size))

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

    def invalidate_light(self):
        self.light.mark_dirty()

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
        player = self.player
        player.swing = 0.22

        damage = 3 if HOTBAR[self.slot] == SWORD else 1
        hitbox = pygame.Rect(
            player.rect.centerx + player.face * 24 - 28,
            player.rect.y - 5,
            56,
            54,
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

        # Lighting internally ignores calls unless sunlight changed
        # by one full light level or a block changed.
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

        player_tile_x = player.rect.centerx // TILE
        player_tile_y = player.rect.centery // TILE
        in_water = world.get(player_tile_x, player_tile_y) == WATER

        player.vel.x = move * (
            PLAYER_SWIM_SPEED if in_water else PLAYER_SPEED
        )

        if move:
            player.face = 1 if move > 0 else -1
            player.walk += dt * 9

        if in_water:
            if jump:
                player.vel.y = -240
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

    def quick_torch(self):
        tile_x, tile_y = self.mouse_tile()
        if (
            self.inv.get(TORCH, 0) > 0
            and self.world.get(tile_x, tile_y) == AIR
        ):
            self.world.set(tile_x, tile_y, TORCH)
            self.inv[TORCH] -= 1
            self.invalidate_light()

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

        if (tile_x, tile_y) != self.mine_target:
            self.mine_target = (tile_x, tile_y)
            self.mine_prog = 0.0

        self.mine_prog += dt
        hardness = TILE_DEFS[tile][3]

        if self.mine_prog >= hardness and hardness < 1000:
            self.world.set(tile_x, tile_y, AIR)
            self.inv[tile] = self.inv.get(tile, 0) + 1
            self.invalidate_light()

            self.spawn_particles(
                tile_x * TILE + TILE // 2,
                tile_y * TILE + TILE // 2,
                TILE_DEFS[tile][4],
                10,
            )

            self.mine_target = None
            self.mine_prog = 0.0

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

            self.enemies.append(Enemy(
                tile_x * TILE,
                (world.surface[tile_x] - 2) * TILE,
                "shade" if night else "slime",
            ))

        for enemy in list(self.enemies):
            enemy.update(world, player, dt)

            if enemy.hp <= 0:
                self.enemies.remove(enemy)
                drop = COAL if enemy.kind == "shade" else FLOWER
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

        if self.daylight < 0.45:
            star_colour = (225, 232, 255)
            for sx, sy in self.stars:
                self.screen.fill(
                    star_colour,
                    (sx * self.W, sy * self.H, 3, 3),
                )

        self.draw_sun_and_moon()
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
        soil_back = (48, 40, 48)
        air_gadgets = (TORCH, LANTERN, FLOWER, TALLGRASS)

        for ty in range(y0, y1):
            row_y = ty * TILE - camy
            for tx in range(x0, x1):
                tile = world.g[tx][ty]
                sx = int(tx * TILE - camx)
                sy = int(row_y)

                underground = ty > world.surface[tx]

                if tile == AIR:
                    if underground:
                        screen.fill(soil_back, (sx, sy, TILE, TILE))
                    continue

                if tile == WATER:
                    continue   # drawn later, over entities

                if underground and tile in air_gadgets:
                    screen.fill(soil_back, (sx, sy, TILE, TILE))

                screen.blit(sprites[tile][(tx + ty) & 1], (sx, sy))

        self.draw_mining_cracks()
        self.draw_lighting(x0, y0, x1, y1)

    def draw_foreground(self):
        x0, y0, x1, y1 = self.visible
        frame = (pygame.time.get_ticks() // 220) % 4

        for ty in range(y0, y1):
            for tx in range(x0, x1):
                if self.world.g[tx][ty] == WATER:
                    img = self.water_frames[(frame + tx) % 4]  # phase offset per column
                    self.screen.blit(img, (int(tx * TILE - self.cam.x),
                                        int(ty * TILE - self.cam.y)))

    def draw_sun_and_moon(self):
        angle = (self.tod - 0.25) * 2 * math.pi

        for colour, body_angle, radius in (
            ((255, 222, 110), angle, 34),
            ((224, 230, 244), angle + math.pi, 27),
        ):
            if math.sin(body_angle) <= -0.15:
                continue

            bx = int(self.W * 0.5 + math.cos(body_angle) * self.W * 0.42)
            by = int(self.H * 0.62 - abs(math.sin(body_angle)) * self.H * 0.5)

            # A square-ish pixel sun/moon fits the chunky art direction.
            pygame.draw.rect(
                self.screen,
                colour,
                (
                    bx - radius // 2,
                    by - radius // 2,
                    radius,
                    radius,
                ),
            )

    def draw_clouds(self):
        colour = lerp_col(
            (38, 42, 66),
            (245, 248, 252),
            self.daylight,
        )
        now = pygame.time.get_ticks() * 0.001

        for cloud_x, cloud_y, speed in self.clouds:
            x = int(
                (cloud_x + now * speed - self.cam.x * 0.22)
                % (self.W + 280)
                - 140
            )

            for dx, dy, w, h in (
                (0, 0, 72, 20),
                (18, -10, 38, 16),
                (42, 2, 44, 16),
            ):
                pygame.draw.rect(
                    self.screen,
                    colour,
                    (x + dx, cloud_y + dy, w, h),
                )

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
        stages = int(self.mine_prog / hardness * 4)

        for i in range(stages + 1):
            offset = 6 + i * 9
            pygame.draw.line(
                self.screen,
                (28, 24, 24),
                (sx + offset, sy + 7),
                (sx + offset + 7, sy + 34),
                3,
            )

    def draw_lighting(self, x0, y0, x1, y1):
        width = x1 - x0
        height = y1 - y0

        if (
            self.light_tile_surface is None
            or self.light_tile_surface.get_size() != (width + 2, height + 2)
        ):
            # +2 border so the smooth upscale can bleed one tile past
            # the visible edges without dark seams.
            self.light_tile_surface = pygame.Surface((width + 2, height + 2))
            self.light_screen_surface = pygame.Surface(
                ((width + 2) * TILE, (height + 2) * TILE))
            self._light_cache_key = None

        # Rebuild the overlay only when the light changed or we scrolled
        # to new tiles; camera sub-tile movement just shifts the blit.
        cache_key = (x0, y0, width, height, self.light.version,
                     self.light.display_version)
        if cache_key != self._light_cache_key:
            pixels = pygame.PixelArray(self.light_tile_surface)

            light_r = self.light.dr
            light_g = self.light.dg
            light_b = self.light.db

            # Minimum values prevent pure black; the smooth upscale
            # blends tile-to-tile instead of hard stepping.
            for local_y in range(height + 2):
                world_y = min(WORLD_H - 1, max(0, y0 + local_y - 1))
                row_offset = world_y * WORLD_W

                for local_x in range(width + 2):
                    world_x = min(WORLD_W - 1, max(0, x0 + local_x - 1))
                    i = row_offset + world_x

                    pixels[local_x, local_y] = (
                        18 + (light_r[i] * 237) // 255,
                        18 + (light_g[i] * 237) // 255,
                        22 + (light_b[i] * 233) // 255,
                    )

            del pixels

            # Smooth (bilinear) upscale: soft gradients between tiles
            # while the underlying data stays tile-based.
            pygame.transform.smoothscale(
                self.light_tile_surface,
                self.light_screen_surface.get_size(),
                self.light_screen_surface,
            )
            self._light_cache_key = cache_key

        self.screen.blit(
            self.light_screen_surface,
            (
                int((x0 - 1) * TILE - self.cam.x),
                int((y0 - 1) * TILE - self.cam.y),
            ),
            special_flags=pygame.BLEND_RGBA_MULT,
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
            base = "slime" if e.kind == "slime" else "shade"
            key = f"{base}_{frame}"
            if e.hurt_t > 0:
                key += "_flash"
            img = ent.get(key)
            if img is not None:
                screen.blit(img, (x + (w - img.get_width()) // 2,
                                  y + e.rect.height - img.get_height()))
            else:  # placeholder fallback (assets missing)
                screen.fill((200, 80, 200), (x, y, w, e.rect.height))

            bar = max(4, TILE // 16)
            screen.fill((58, 20, 20), (x, y - bar - 3, w, bar))
            screen.fill((235, 80, 80),
                        (x, y - bar - 3,
                         int(w * max(0, e.hp) / e.max_hp), bar))

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
            screen.blit(img, (x, y))

        # Sword swing: pixel-art blade sweeping an arc.
        if p.swing > 0 and HOTBAR[self.slot] == SWORD:
            sword = ent.get("sword")
            if sword is not None:
                if self._swing_cache is None:
                    size = max(24, int(TILE * 0.7))
                    self._swing_cache = pygame.transform.scale(
                        sword, (size, size))
                progress = 1.0 - p.swing / 0.22  # 0 -> 1
                angle = (-70 + 140 * progress) * p.face
                blade = pygame.transform.rotate(self._swing_cache, angle)
                hx = x + (p.rect.width if p.face > 0 else 0)
                screen.blit(blade, (int(hx - blade.get_width() // 2
                                        + p.face * TILE * 0.25),
                                    int(y + p.rect.height * 0.15)))

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
            cached = pygame.transform.scale(src, self.heart_size)
            self._heart_cache[key] = cached
        return cached

    def draw_ui(self):
        player = self.player
        s = self.ui_scale
        margin = int(20 * s)

        hw, hh = self.heart_size
        gap = max(4, int(7 * s))
        for i in range(player.max_hp):
            self.screen.blit(self.heart_icon(i < player.hp),
                             (margin + i * (hw + gap), int(14 * s)))

        hours = int(((self.tod - 0.25) % 1.0) * 24)
        minutes = int((self.tod * 1440) % 60)
        sun_or_moon = "SUN" if self.daylight > 0.4 else "MOON"

        clock_text = (
            f"Day {self.day}  {sun_or_moon}  "
            f"{hours:02d}:{minutes:02d}"
        )
        self.screen.blit(
            self.text(clock_text, (255, 255, 238)),
            (margin, int(14 * s) + hh + int(6 * s)),
        )

        if self.msg_t > 0:
            text = self.text(self.msg, (255, 248, 220), big=True)
            pad_x, pad_y = int(14 * s), int(8 * s)
            box = pygame.Rect(
                int(self.W / 2 - text.get_width() / 2 - pad_x),
                int(52 * s),
                text.get_width() + pad_x * 2,
                text.get_height() + pad_y * 2,
            )
            pygame.draw.rect(
                self.screen, (18, 22, 36), box,
                border_radius=max(4, int(8 * s)),
            )
            self.screen.blit(
                text, (int(self.W / 2 - text.get_width() / 2),
                       box.y + pad_y),
            )

        self.draw_hotbar()

        if self.craft_open:
            self.draw_craft()

    def draw_hotbar(self):
        s = self.ui_scale
        slot_size = self.slot_size
        inner = self.slot_inner
        total_width = len(HOTBAR) * slot_size
        start_x = int(self.W / 2 - total_width / 2)
        help_h = self.font.get_height() + int(8 * s)
        y = self.H - inner - help_h - int(8 * s)
        pad = (inner - self.icon_size) // 2

        for i, item in enumerate(HOTBAR):
            rect = pygame.Rect(
                start_x + i * slot_size, y, inner, inner
            )

            pygame.draw.rect(
                self.screen, (16, 20, 32), rect,
                border_radius=max(3, int(5 * s)),
            )
            pygame.draw.rect(
                self.screen,
                (255, 220, 120)
                if i == self.slot
                else (115, 124, 146),
                rect,
                max(2, int(3 * s)) if i == self.slot else 1,
                border_radius=max(3, int(5 * s)),
            )

            icon = self.icons.get(item)
            if icon is not None:
                self.screen.blit(icon, (rect.x + pad, rect.y + pad))

            count = self.inv.get(item, 0)
            if item != SWORD and count:
                text = self.text(str(count), (255, 255, 255))
                self.screen.blit(
                    text, (rect.right - text.get_width() - int(5 * s),
                           rect.bottom - text.get_height() - int(3 * s)),
                )

            number = self.text(str(i + 1), (190, 200, 215))
            self.screen.blit(number, (rect.x + int(4 * s),
                                      rect.y + int(3 * s)))

        tile_x, tile_y = self.mouse_tile()
        hovered = self.world.get(tile_x, tile_y)

        selected = (
            TILE_DEFS[hovered][0]
            if hovered != AIR
            else ITEM_NAMES[HOTBAR[self.slot]]
        )

        help_text = (
            f"{selected}  -  E: Craft  F: Torch  "
            f"F11: Fullscreen  LMB: Mine/Attack  RMB: Place"
        )
        rendered = self.text(help_text, (228, 234, 244))
        self.screen.blit(
            rendered,
            (int(self.W / 2 - rendered.get_width() / 2),
             self.H - help_h + int(2 * s)),
        )

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

        pygame.draw.rect(
            self.screen, (22, 26, 42), panel,
            border_radius=max(6, int(12 * s)),
        )
        pygame.draw.rect(
            self.screen, (255, 210, 130), panel, max(2, int(2 * s)),
            border_radius=max(6, int(12 * s)),
        )
        self.screen.blit(
            self.text("Crafting", (255, 225, 160), big=True),
            (panel.x + int(24 * s), panel.y + int(16 * s)),
        )

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
            pygame.draw.rect(
                self.screen,
                (44, 56, 80) if can_craft else (32, 36, 50),
                rect,
                border_radius=max(4, int(7 * s)),
            )

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

                elif event.key == pygame.K_f:
                    self.quick_torch()

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
        self.invalidate_light()

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