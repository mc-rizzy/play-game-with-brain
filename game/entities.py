import math
import random

import pygame

from game.config import TILE, GRAVITY


def move_collide(world, rect, vel, dt):
    on_ground = False

    rect.x += vel.x * dt
    left = int(rect.left // TILE)
    right = int(rect.right // TILE)
    top = int(rect.top // TILE)
    bottom = int(rect.bottom // TILE)

    for tx in range(left, right + 1):
        for ty in range(top, bottom + 1):
            if world.solid(tx, ty):
                tile_rect = pygame.Rect(tx * TILE, ty * TILE, TILE, TILE)
                if rect.colliderect(tile_rect):
                    if vel.x > 0:
                        rect.right = tile_rect.left
                    elif vel.x < 0:
                        rect.left = tile_rect.right
                    vel.x = 0

    rect.y += vel.y * dt
    left = int(rect.left // TILE)
    right = int(rect.right // TILE)
    top = int(rect.top // TILE)
    bottom = int(rect.bottom // TILE)

    for tx in range(left, right + 1):
        for ty in range(top, bottom + 1):
            if world.solid(tx, ty):
                tile_rect = pygame.Rect(tx * TILE, ty * TILE, TILE, TILE)
                if rect.colliderect(tile_rect):
                    if vel.y > 0:
                        rect.bottom = tile_rect.top
                        on_ground = True
                    elif vel.y < 0:
                        rect.top = tile_rect.bottom
                    vel.y = 0

    return on_ground


class Player:
    def __init__(self, x, y):
        self.rect = pygame.Rect(
            x, y,
            int(TILE * 0.80),    # slightly less than 1 block wide
            int(TILE * 1.50),    # 1.5 blocks tall (16x24 art)
        )
        self.vel = pygame.Vector2()
        self.hp, self.max_hp = 10, 10
        self.face, self.walk, self.hurt_t, self.swing = 1, 0.0, 0.0, 0.0
        self.on_ground = False


class Enemy:
    def __init__(self, x, y, kind):
        self.kind = kind
        self.max_hp = {"slime": 4, "bat": 3, "skeleton": 5,
                       "shade": 8}[kind]
        self.hp = self.max_hp

        if kind == "slime":
            w, h = int(TILE * 0.80), int(TILE * 0.80)   # 1×1 block
        elif kind == "bat":
            w, h = int(TILE * 0.70), int(TILE * 0.60)
        elif kind == "skeleton":
            w, h = int(TILE * 0.70), int(TILE * 1.50)
        else:  # shade
            w, h = int(TILE * 0.80), int(TILE * 1.60)

        self.rect = pygame.Rect(x, y, w, h)
        self.vel = pygame.Vector2()
        self.hurt_t, self.touch_t, self.hop = 0.0, 0.0, random.random()
        self.phase = 0.0

    def update(self, world, player, dt):
        distance = player.rect.centerx - self.rect.centerx
        self.hop -= dt
        self.phase += dt * 6.0

        in_water = world.water_at_pixel(self.rect.centerx,
                                       self.rect.centery)

        if abs(distance) < TILE * 9:
            if self.kind == "slime":
                speed = TILE * 1.2
                if self.hop <= 0 and not in_water:
                    self.vel.x = math.copysign(speed, distance)
                    self.vel.y = -TILE * 8.0
                    self.hop = 0.8
            elif self.kind == "bat":
                # Sine pursuit: tracks the player's height with a wing
                # wobble instead of falling.
                speed = TILE * 2.4
                self.vel.x = (
                    math.copysign(speed, distance)
                    if abs(distance) > 24
                    else 0
                )
                dy = player.rect.centery - self.rect.centery
                self.vel.y = (max(-160.0, min(160.0, dy * 3.0))
                              + math.sin(self.phase) * 40.0)
            elif self.kind == "skeleton":
                # Relentless pogo-chaser: fast run, hops whenever the
                # feet touch ground.
                speed = TILE * 2.3
                self.vel.x = (
                    math.copysign(speed, distance)
                    if abs(distance) > 24
                    else 0
                )
                if self.hop <= 0:
                    feet = world.solid(self.rect.centerx // TILE,
                                       (self.rect.bottom + 2) // TILE)
                    if feet:
                        self.vel.y = -TILE * 9.5
                    self.hop = 0.5
            else:  # shade
                speed = TILE * 1.75
                self.vel.x = (
                    math.copysign(speed, distance)
                    if abs(distance) > 32
                    else 0
                )
                if self.hop <= 0 and random.random() < 0.01:
                    self.vel.y = -TILE * 8.5
                    self.hop = 1.0
        else:
            self.vel.x *= 0.8
            if self.kind == "bat":
                self.vel.y = math.sin(self.phase) * 30.0

        if in_water:
            head_dry = not world.water_at_pixel(
                self.rect.centerx, self.rect.top + 4)
            if head_dry:
                # Bob at the surface: settle back down, capped slow.
                self.vel.y = min(self.vel.y + GRAVITY * 0.25 * dt, 40)
            else:
                # Buoyant: rise until the head breaks the surface.
                self.vel.y = max(self.vel.y - GRAVITY * 0.8 * dt, -140)
        elif self.kind == "bat":
            pass  # wings beat gravity
        else:
            self.vel.y += GRAVITY * dt

        self.hurt_t = max(0.0, self.hurt_t - dt)
        self.touch_t = max(0.0, self.touch_t - dt)

        move_collide(world, self.rect, self.vel, dt)