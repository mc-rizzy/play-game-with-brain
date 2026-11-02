import math, random
from config import WORLD_W, WORLD_H
from tiles import *

class World:
    def __init__(self):
        self.g = [[AIR] * WORLD_H for _ in range(WORLD_W)]
        self.surface = [30] * WORLD_W
        self.generate()

    def generate(self):
        rng, h = random.Random(20240607), 32
        for x in range(WORLD_W):
            h = max(22, min(48, h + rng.choice((-1, 0, 0, 1))))
            height = int(h + 5 * math.sin(x * 0.045) + 2 * math.sin(x * 0.15))
            self.surface[x] = height
            beach = height > 40
            for y in range(height, WORLD_H):
                d = y - height
                if y == WORLD_H - 1:   t = BEDROCK
                elif d == 0:           t = SAND if beach else GRASS
                elif d < 4:            t = SAND if beach else DIRT
                elif rng.random() < .012: t = GOLD if d > 34 else (IRON if d > 16 else COAL)
                elif rng.random() < .012 and d > 8: t = COAL
                else:                  t = STONE
                self.g[x][y] = t
            if beach:
                for y in range(height - 2, height):
                    if y >= 0: self.g[x][y] = WATER
            elif rng.random() < .10:
                self.g[x][height - 1] = FLOWER if rng.random() < .4 else TALLGRASS
            if 8 < x < WORLD_W - 8 and rng.random() < .055 and not beach:
                self.tree(x, height, rng)

    def tree(self, x, y, rng):
        trunk = rng.randint(4, 6)
        for i in range(1, trunk + 1): self.g[x][y - i] = WOOD
        for dx in range(-2, 3):
            for dy in range(-trunk - 2, -trunk + 1):
                if abs(dx) == 2 and dy == -trunk + 1: continue
                if 0 <= x + dx < WORLD_W and 0 <= y + dy: self.g[x + dx][y + dy] = LEAVES

    def get(self, x, y):
        return self.g[x][y] if 0 <= x < WORLD_W and 0 <= y < WORLD_H else BEDROCK
    def solid(self, x, y):
        t = self.get(x, y)
        return TILE_DEFS[t][1] if t != AIR else False
    def opaque(self, x, y):
        t = self.get(x, y)
        return TILE_DEFS[t][2] if t != AIR else False
    def set(self, x, y, t):
        if 0 <= x < WORLD_W and 0 <= y < WORLD_H: self.g[x][y] = t