from collections import deque

from config import WORLD_W, WORLD_H, LIGHT_SPREAD_LOSS
from tiles import AIR, TILE_DEFS, light_factor


class Lighting:
    """
    Tile-based coloured lighting.

    - Every tile has its own R/G/B light value.
    - Light spreads tile-to-tile.
    - Entering a tile multiplies the light by that tile's factor.
    - Ordinary solid blocks use SOLID_LIGHT_FACTOR from config.py.
    - Glass, leaves, water, stone, etc. can override that factor
      in TILE_LIGHT_FACTOR inside tiles.py.
    """

    def __init__(self, world):
        self.world = world
        self.count = WORLD_W * WORLD_H

        self.r = bytearray(self.count)
        self.g = bytearray(self.count)
        self.b = bytearray(self.count)

        self.dirty = True
        self.cached_sun = -1

    def mark_dirty(self):
        self.dirty = True

    def light_at(self, x, y):
        if not (0 <= x < WORLD_W and 0 <= y < WORLD_H):
            return (0, 0, 0)

        i = y * WORLD_W + x
        return (self.r[i], self.g[i], self.b[i])

    def brightness_at(self, x, y):
        r, g, b = self.light_at(x, y)
        return max(r, g, b)

    def compute(self, daylight):
        sun = int(max(0.0, min(1.0, daylight)) * 255)

        if not self.dirty and sun == self.cached_sun:
            return

        self.cached_sun = sun
        self.dirty = False

        grid = self.world.g
        queue = deque()

        self.r = bytearray(self.count)
        self.g = bytearray(self.count)
        self.b = bytearray(self.count)

        def add_source(x, y, red, green, blue):
            i = y * WORLD_W + x
            changed = False

            if red > self.r[i]:
                self.r[i] = red
                changed = True
            if green > self.g[i]:
                self.g[i] = green
                changed = True
            if blue > self.b[i]:
                self.b[i] = blue
                changed = True

            if changed:
                queue.append((x, y))

        # ----------------------------------------------------
        # Direct sunlight falling down each column.
        # It is weakened by every tile it passes through,
        # including solid blocks.
        # ----------------------------------------------------
        for x in range(WORLD_W):
            level = float(sun)

            for y in range(WORLD_H):
                tile = grid[x][y]

                if y > 0:
                    level = level * light_factor(tile) - 3

                if level <= 2:
                    # Still need to place emissive blocks below
                    # this point, so scan the rest cheaply.
                    for yy in range(y, WORLD_H):
                        t = grid[x][yy]
                        if t != AIR and TILE_DEFS[t][5] != (0, 0, 0):
                            er, eg, eb = TILE_DEFS[t][5]
                            add_source(x, yy, er * 17, eg * 17, eb * 17)
                    break

                add_source(
                    x, y,
                    int(level),
                    int(level),
                    int(level * 0.96),
                )

                if tile != AIR and TILE_DEFS[tile][5] != (0, 0, 0):
                    er, eg, eb = TILE_DEFS[tile][5]
                    add_source(x, y, er * 17, eg * 17, eb * 17)

        # ----------------------------------------------------
        # Spread light tile-to-tile.
        # ----------------------------------------------------
        while queue:
            x, y = queue.popleft()
            i = y * WORLD_W + x

            cr = self.r[i]
            cg = self.g[i]
            cb = self.b[i]

            if cr <= 2 and cg <= 2 and cb <= 2:
                continue

            for nx, ny in ((x - 1, y), (x + 1, y),
                           (x, y - 1), (x, y + 1)):
                if not (0 <= nx < WORLD_W and 0 <= ny < WORLD_H):
                    continue

                next_tile = grid[nx][ny]
                factor = light_factor(next_tile)

                nr = int(cr * factor) - LIGHT_SPREAD_LOSS
                ng = int(cg * factor) - LIGHT_SPREAD_LOSS
                nb = int(cb * factor) - LIGHT_SPREAD_LOSS

                if nr <= 2 and ng <= 2 and nb <= 2:
                    continue

                ni = ny * WORLD_W + nx
                changed = False

                # The +2 threshold prevents endless tiny updates.
                if nr > self.r[ni] + 2:
                    self.r[ni] = max(0, nr)
                    changed = True
                if ng > self.g[ni] + 2:
                    self.g[ni] = max(0, ng)
                    changed = True
                if nb > self.b[ni] + 2:
                    self.b[ni] = max(0, nb)
                    changed = True

                if changed:
                    queue.append((nx, ny))