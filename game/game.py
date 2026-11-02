import math
import random

import pygame

from config import (
    TILE,
    WORLD_W,
    WORLD_H,
    DAY_LENGTH,
    GRAVITY,
    REACH,
    WINDOWED_SIZE,
    PLAYER_SPEED,
    PLAYER_SWIM_SPEED,
    PLAYER_JUMP,
    MAX_PARTICLES,
)
from tiles import *
from sprites import make_sprites
from world import World
from lighting import Lighting
from entities import Player, Enemy, move_collide
from utils import lerp_col, sky_and_daylight
from sprites import make_sprites, make_water_frames


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Cozy Blocks — a slice-of-life adventure")

        self.fullscreen = True
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.W, self.H = self.screen.get_size()

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 17)
        self.big = pygame.font.SysFont("consolas", 26, bold=True)

        self.sprites = make_sprites()
        self.icons = {
            item: pygame.transform.scale(self.sprites[item][0], (38, 38))
            for item in HOTBAR
            if item != SWORD
        }

        self.world = World()
        self.light = Lighting(self.world)
        self.light.compute(1.0)

        self.water_frames = make_water_frames()
        self.visible = (0, 0, 0, 0)

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

        keys = pygame.key.get_pressed()

        player_tile_x = player.rect.centerx // TILE
        player_tile_y = player.rect.centery // TILE
        in_water = world.get(player_tile_x, player_tile_y) == WATER

        move = (
            (keys[pygame.K_d] or keys[pygame.K_RIGHT])
            - (keys[pygame.K_a] or keys[pygame.K_LEFT])
        )

        player.vel.x = move * (
            PLAYER_SWIM_SPEED if in_water else PLAYER_SPEED
        )

        if move:
            player.face = move
            player.walk += dt * 9

        if in_water:
            if keys[pygame.K_SPACE] or keys[pygame.K_w]:
                player.vel.y = -240
            player.vel.y += GRAVITY * 0.22 * dt
            player.vel.y = min(player.vel.y, 145)
        else:
            player.vel.y += GRAVITY * dt
            if (
                keys[pygame.K_SPACE] or keys[pygame.K_w]
            ) and player.on_ground:
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
        key = (top, bottom, self.W, self.H)

        if self.sky_surface is not None and self.sky_key == key:
            return self.sky_surface

        surface = pygame.Surface((self.W, self.H))
        strip_height = 16

        for y in range(0, self.H, strip_height):
            colour = lerp_col(top, bottom, y / max(1, self.H))
            surface.fill(colour, (0, y, self.W, strip_height))

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

        for ty in range(y0, y1):
            for tx in range(x0, x1):
                tile = world.g[tx][ty]
                sx = int(tx * TILE - self.cam.x)
                sy = int(ty * TILE - self.cam.y)

                underground = ty > world.surface[tx]

                if tile == AIR:
                    if underground:
                        self.screen.fill(
                            (48, 40, 48), (sx, sy, TILE, TILE)
                        )
                    continue

                self.visible = (x0, y0, x1, y1)
                # inside the tile loop:
                if tile == WATER:
                    continue   # drawn later, over entities

                if underground and tile in (
                    TORCH,
                    LANTERN,
                    FLOWER,
                    TALLGRASS,
                ):
                    self.screen.fill(
                        (48, 40, 48), (sx, sy, TILE, TILE)
                    )

                sprite = sprites[tile][(tx + ty) & 1]
                self.screen.blit(sprite, (sx, sy))

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
            or self.light_tile_surface.get_size() != (width, height)
        ):
            self.light_tile_surface = pygame.Surface((width, height))
            self.light_screen_surface = pygame.Surface(
                (width * TILE, height * TILE)
            )

        pixels = pygame.PixelArray(self.light_tile_surface)

        light_r = self.light.r
        light_g = self.light.g
        light_b = self.light.b

        # Minimum values prevent the game becoming pure black while
        # still preserving strong tile-to-tile contrast.
        for local_y in range(height):
            world_y = y0 + local_y
            row_offset = world_y * WORLD_W

            for local_x in range(width):
                i = row_offset + x0 + local_x

                pixels[local_x, local_y] = (
                    18 + (light_r[i] * 237) // 255,
                    18 + (light_g[i] * 237) // 255,
                    22 + (light_b[i] * 233) // 255,
                )

        del pixels

        # Nearest-neighbour scaling: every light pixel becomes exactly
        # one complete game tile. No smooth edges between tiles.
        pygame.transform.scale(
            self.light_tile_surface,
            self.light_screen_surface.get_size(),
            self.light_screen_surface,
        )

        self.screen.blit(
            self.light_screen_surface,
            (
                int(x0 * TILE - self.cam.x),
                int(y0 * TILE - self.cam.y),
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

        for e in self.enemies:
            x, y, w, h = (int(e.rect.x - self.cam.x), int(e.rect.y - self.cam.y),
                        e.rect.width, e.rect.height)
            flash = e.hurt_t > 0

            if e.kind == "slime":
                body = (255, 125, 125) if flash else (92, 210, 104)
                pygame.draw.rect(self.screen, body, (x, y + h // 3, w, h - h // 3))
                pygame.draw.rect(self.screen, body, (x + w // 6, y + h // 8, w * 2 // 3, h // 3))
                eye = max(4, TILE // 16)
                pygame.draw.rect(self.screen, (24, 52, 30), (x + w // 4, y + h // 2, eye, eye))
                pygame.draw.rect(self.screen, (24, 52, 30), (x + w * 2 // 3, y + h // 2, eye, eye))
            else:
                body = (255, 130, 130) if flash else (100, 112, 140)
                pygame.draw.rect(self.screen, body, (x + w // 5, y, w * 3 // 5, h // 2))
                pygame.draw.rect(self.screen, (66, 84, 76), (x + w // 8, y + h // 2, w * 3 // 4, h // 2))
                eye = max(4, TILE // 16)
                pygame.draw.rect(self.screen, (255, 66, 66), (x + w // 3, y + h // 5, eye, eye))
                pygame.draw.rect(self.screen, (255, 66, 66), (x + w * 3 // 5, y + h // 5, eye, eye))

            bar = max(4, TILE // 16)
            pygame.draw.rect(self.screen, (58, 20, 20), (x, y - bar - 3, w, bar))
            pygame.draw.rect(self.screen, (235, 80, 80),
                            (x, y - bar - 3, int(w * max(0, e.hp) / e.max_hp), bar))

        # Player: drawn from rect proportions (0.85 × 1.9 blocks)
        x, y, w, h = (int(p.rect.x - self.cam.x), int(p.rect.y - self.cam.y),
                    p.rect.width, p.rect.height)
        hurt = p.hurt_t > 0 and int(p.hurt_t * 20) % 2 == 0
        skin = (255, 165, 165) if hurt else (243, 192, 145)
        leg = int(math.sin(p.walk) * TILE * 0.06) if abs(p.vel.x) > 5 and p.on_ground else 0
        head_h, body_y = int(h * 0.36), int(h * 0.36)

        pygame.draw.rect(self.screen, (70, 92, 148),  (x + w // 6, y + h * 2 // 3, w // 4, h // 3 + leg))
        pygame.draw.rect(self.screen, (70, 92, 148),  (x + w * 3 // 5, y + h * 2 // 3, w // 4, h // 3 - leg))
        pygame.draw.rect(self.screen, (92, 166, 116), (x + w // 10, body_y, w * 4 // 5, h // 3))
        pygame.draw.rect(self.screen, skin,           (x + w // 7, y, w * 5 // 7, head_h))
        pygame.draw.rect(self.screen, (106, 68, 40),  (x + w // 7, y, w * 5 // 7, head_h // 3))
        eye_x = x + (w * 2 // 3 if p.face > 0 else w // 4)
        pygame.draw.rect(self.screen, (30, 30, 36), (eye_x, y + head_h // 2, max(4, TILE // 20), max(4, TILE // 20)))

        arm_x = x + (w if p.face > 0 else 0)
        if p.swing > 0:
            pygame.draw.line(self.screen, skin, (arm_x, y + h // 2),
                            (arm_x + p.face * TILE // 3, y + h // 4), max(5, TILE // 12))
            if HOTBAR[self.slot] == SWORD:
                pygame.draw.line(self.screen, (222, 228, 238),
                                (arm_x + p.face * TILE // 3, y + h // 4),
                                (arm_x + p.face * TILE * 2 // 3, y), max(5, TILE // 12))
        else:
            pygame.draw.line(self.screen, skin, (arm_x, y + h // 2),
                            (arm_x + p.face * TILE // 10, y + h * 2 // 3), max(5, TILE // 12))

        for pt in self.particles:
            pygame.draw.rect(self.screen, pt[5], (int(pt[0] - self.cam.x), int(pt[1] - self.cam.y), 6, 6))
        for t in self.texts:
            self.screen.blit(self.font.render(t[2], True, t[4]),
                            (int(t[0] - self.cam.x), int(t[1] - self.cam.y)))

    # ------------------------------------------------------------ UI

    def draw_ui(self):
        player = self.player

        for i in range(player.max_hp):
            colour = (235, 70, 90) if i < player.hp else (68, 54, 64)
            cx = 26 + i * 27
            cy = 25

            pygame.draw.rect(self.screen, colour, (cx - 9, cy - 5, 18, 12))
            pygame.draw.rect(self.screen, colour, (cx - 5, cy + 5, 10, 7))

        hours = int(((self.tod - 0.25) % 1.0) * 24)
        minutes = int((self.tod * 1440) % 60)
        sun_or_moon = "SUN" if self.daylight > 0.4 else "MOON"

        clock_text = (
            f"Day {self.day}  {sun_or_moon}  "
            f"{hours:02d}:{minutes:02d}"
        )
        self.screen.blit(
            self.font.render(clock_text, True, (255, 255, 238)),
            (20, 48),
        )

        if self.msg_t > 0:
            text = self.big.render(self.msg, True, (255, 248, 220))
            box = pygame.Rect(
                int(self.W / 2 - text.get_width() / 2 - 14),
                58,
                text.get_width() + 28,
                42,
            )
            pygame.draw.rect(
                self.screen, (18, 22, 36), box, border_radius=8
            )
            self.screen.blit(
                text, (int(self.W / 2 - text.get_width() / 2), 65)
            )

        self.draw_hotbar()

        if self.craft_open:
            self.draw_craft()

    def draw_hotbar(self):
        slot_size = 54
        total_width = len(HOTBAR) * slot_size
        start_x = int(self.W / 2 - total_width / 2)
        y = self.H - 68

        for i, item in enumerate(HOTBAR):
            rect = pygame.Rect(
                start_x + i * slot_size, y, 50, 50
            )

            pygame.draw.rect(
                self.screen, (16, 20, 32), rect, border_radius=5
            )
            pygame.draw.rect(
                self.screen,
                (255, 220, 120)
                if i == self.slot
                else (115, 124, 146),
                rect,
                3 if i == self.slot else 1,
                border_radius=5,
            )

            if item == SWORD:
                pygame.draw.line(
                    self.screen,
                    (226, 231, 240),
                    (rect.x + 14, rect.y + 37),
                    (rect.x + 36, rect.y + 13),
                    6,
                )
                pygame.draw.line(
                    self.screen,
                    (148, 98, 54),
                    (rect.x + 11, rect.y + 40),
                    (rect.x + 20, rect.y + 31),
                    6,
                )
            else:
                self.screen.blit(self.icons[item], (rect.x + 6, rect.y + 6))

            count = self.inv.get(item, 0)
            if item != SWORD and count:
                text = self.font.render(
                    str(count), True, (255, 255, 255)
                )
                self.screen.blit(
                    text, (rect.right - 17, rect.bottom - 19)
                )

            number = self.font.render(
                str(i + 1), True, (190, 200, 215)
            )
            self.screen.blit(number, (rect.x + 4, rect.y + 3))

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
        self.screen.blit(
            self.font.render(help_text, True, (228, 234, 244)),
            (start_x, self.H - 14),
        )

    def draw_craft(self):
        panel = pygame.Rect(
            int(self.W / 2 - 250),
            int(self.H / 2 - 200),
            500,
            400,
        )

        pygame.draw.rect(
            self.screen, (22, 26, 42), panel, border_radius=12
        )
        pygame.draw.rect(
            self.screen, (255, 210, 130), panel, 2, border_radius=12
        )
        self.screen.blit(
            self.big.render("Crafting", True, (255, 225, 160)),
            (panel.x + 24, panel.y + 18),
        )

        self.craft_rects = []
        y = panel.y + 62

        for recipe in RECIPES:
            name, result, count, cost = recipe
            can_craft = all(
                self.inv.get(k, 0) >= v for k, v in cost.items()
            )

            rect = pygame.Rect(
                panel.x + 22, y, panel.w - 44, 44
            )
            pygame.draw.rect(
                self.screen,
                (44, 56, 80) if can_craft else (32, 36, 50),
                rect,
                border_radius=7,
            )

            cost_text = ", ".join(
                f"{ITEM_NAMES[k]} x{v} ({self.inv.get(k, 0)})"
                for k, v in cost.items()
            )
            text = self.font.render(
                f"{name} - {cost_text}",
                True,
                (235, 240, 255)
                if can_craft
                else (128, 136, 154),
            )
            self.screen.blit(text, (rect.x + 14, rect.y + 13))

            self.craft_rects.append((rect, recipe))
            y += 52

        self.screen.blit(
            self.font.render(
                "Click a recipe to craft - E/Esc to close",
                True,
                (180, 190, 205),
            ),
            (panel.x + 24, panel.bottom - 30),
        )

    # ------------------------------------------------------------ events

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

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
                    tile_x, tile_y = self.mouse_tile()

                    if (
                        self.inv.get(TORCH, 0) > 0
                        and self.world.get(tile_x, tile_y) == AIR
                    ):
                        self.world.set(tile_x, tile_y, TORCH)
                        self.inv[TORCH] -= 1
                        self.invalidate_light()

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