import math
import random

import pygame

from config import TILE, GRAVITY
from tiles import WATER


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
            int(TILE * 0.85),    # slightly less than 1 block wide
            int(TILE * 1.90),    # slightly less than 2 blocks tall
        )
        self.vel = pygame.Vector2()
        self.hp, self.max_hp = 10, 10
        self.face, self.walk, self.hurt_t, self.swing = 1, 0.0, 0.0, 0.0
        self.on_ground = False


class Enemy:
    def __init__(self, x, y, kind):
        self.kind = kind
        self.max_hp = 4 if kind == "slime" else 8
        self.hp = self.max_hp
        
        if kind == "slime":
            w, h = int(TILE * 0.90), int(TILE * 0.80)   # just under 1×1 block
        else:  # shade
            w, h = int(TILE * 0.80), int(TILE * 1.70)
            
        self.rect = pygame.Rect(x, y, w, h)
        self.vel = pygame.Vector2()
        self.hurt_t, self.touch_t, self.hop = 0.0, 0.0, random.random()

    def update(self, world, player, dt):
        distance = player.rect.centerx - self.rect.centerx
        self.hop -= dt

        in_water = (
            world.get(
                self.rect.centerx // TILE,
                self.rect.centery // TILE,
            )
            == WATER
        )

        if abs(distance) < TILE * 9:
            speed = TILE * 1.2 if self.kind == "slime" else TILE * 1.75

            if self.kind == "slime":
                if self.hop <= 0 and not in_water:
                    self.vel.x = math.copysign(95, distance) * speed
                    self.vel.y = -TILE * 8.0
                    self.hop = 0.8
            else:
                self.vel.x = (
                    math.copysign(140, distance) * speed
                    if abs(distance) > 32
                    else 0
                )
                if self.hop <= 0 and random.random() < 0.01:
                    self.vel.y = -TILE * 8.5
                    self.hop = 1.0
        else:
            self.vel.x *= 0.8

        gravity_scale = 0.25 if in_water else 1.0
        self.vel.y += GRAVITY * gravity_scale * dt

        if in_water:
            self.vel.y = min(self.vel.y, 120)

        self.hurt_t = max(0.0, self.hurt_t - dt)
        self.touch_t = max(0.0, self.touch_t - dt)

        move_collide(world, self.rect, self.vel, dt)