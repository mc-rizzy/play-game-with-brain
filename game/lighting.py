from collections import deque

from game.config import (
    WORLD_W, WORLD_H, LIGHT_SPREAD_LOSS,
    LIGHT_DIAGONAL_LOSS, LIGHT_DIAGONAL_FACTOR, LIGHT_DISPLAY_LERP,
)
from game.tiles import AIR, TILE_DEFS, light_factor, is_opaque


class Lighting:
    """
    Tile-based coloured lighting.

    - Every tile has its own R/G/B light value.
    - Light spreads tile-to-tile, orthogonally and diagonally.
    - Entering a tile multiplies the light by that tile's factor.
    - Ordinary solid blocks use SOLID_LIGHT_FACTOR from config.py.
    - Glass, leaves, water, stone, etc. can override that factor
      in TILE_LIGHT_FACTOR inside tiles.py.
    - Display buffers lerp towards the computed target each frame so
      day/night and torch changes fade smoothly instead of popping.
    """

    _ORTHO = ((-1, 0), (1, 0), (0, -1), (0, 1))
    _DIAG = ((-1, -1), (1, -1), (-1, 1), (1, 1))

    def __init__(self, world):
        self.world = world
        self.count = WORLD_W * WORLD_H

        self.r = bytearray(self.count)
        self.g = bytearray(self.count)
        self.b = bytearray(self.count)

        # Smoothed display copies (rendering reads these).
        self.dr = bytearray(self.count)
        self.dg = bytearray(self.count)
        self.db = bytearray(self.count)
        self._display_init = False

        self.dirty = True
        self.cached_sun = -1
        self.version = 0
        self.display_version = 0

    def mark_dirty(self):
        self.dirty = True

    def _idx(self, x, y):
        return y * WORLD_W + x

    def light_at(self, x, y):
        if not (0 <= x < WORLD_W and 0 <= y < WORLD_H):
            return (0, 0, 0)
        i = self._idx(x, y)
        return (self.r[i], self.g[i], self.b[i])

    def light_at_display(self, x, y):
        if not (0 <= x < WORLD_W and 0 <= y < WORLD_H):
            return (0, 0, 0)
        i = self._idx(x, y)
        return (self.dr[i], self.dg[i], self.db[i])

    def brightness_at(self, x, y):
        r, g, b = self.light_at(x, y)
        return max(r, g, b)

    def update_display(self, dt):
        """Ease display buffers towards the computed target."""
        if not self._display_init:
            self.dr[:] = self.r
            self.dg[:] = self.g
            self.db[:] = self.b
            self._display_init = True
            self.display_version += 1
            return True
        t = min(1.0, dt * LIGHT_DISPLAY_LERP)
        if t >= 1.0:
            changed = (self.dr != self.r or self.dg != self.g
                       or self.db != self.b)
            if changed:
                self.dr[:] = self.r
                self.dg[:] = self.g
                self.db[:] = self.b
                self.display_version += 1
            return changed
        # Integer lerp in place; fast enough for ~25k tiles.
        r, g, b = self.r, self.g, self.b
        dr, dg, db = self.dr, self.dg, self.db
        changed = False
        for i in range(self.count):
            for src, dst in ((r, dr), (g, dg), (b, db)):
                d = src[i] - dst[i]
                if d:
                    step = int(d * t)
                    if step == 0:
                        step = 1 if d > 0 else -1
                    dst[i] = max(0, min(255, dst[i] + step))
                    changed = True
        if changed:
            self.display_version += 1
        return changed

    def compute(self, daylight):
        sun = int(max(0.0, min(1.0, daylight)) * 255)

        if not self.dirty and sun == self.cached_sun:
            return False

        self.cached_sun = sun
        self.dirty = False
        self.version += 1

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
        # ----------------------------------------------------
        for x in range(WORLD_W):
            level = float(sun)

            for y in range(WORLD_H):
                tile = grid[x][y]

                if y > 0:
                    level = level * light_factor(tile) - 3

                if level <= 2:
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
        # Spread light tile-to-tile (orthogonal + diagonal).
        # ----------------------------------------------------
        while queue:
            x, y = queue.popleft()
            i = y * WORLD_W + x

            cr = self.r[i]
            cg = self.g[i]
            cb = self.b[i]

            if cr <= 2 and cg <= 2 and cb <= 2:
                continue

            for dx, dy in self._ORTHO:
                self._spread(grid, queue, x, y, x + dx, y + dy,
                             cr, cg, cb, diagonal=False)
            for dx, dy in self._DIAG:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < WORLD_W and 0 <= ny < WORLD_H):
                    continue
                # Don't leak diagonally through solid corners: if both
                # orthogonal neighbours are opaque, the diagonal is sealed.
                if (is_opaque(grid[x + dx][y])
                        and is_opaque(grid[x][y + dy])):
                    continue
                self._spread(grid, queue, x, y, nx, ny, cr, cg, cb,
                             diagonal=True)
        return True

    def _spread(self, grid, queue, x, y, nx, ny, cr, cg, cb, diagonal):
        if not (0 <= nx < WORLD_W and 0 <= ny < WORLD_H):
            return
        next_tile = grid[nx][ny]
        factor = light_factor(next_tile)
        if diagonal:
            factor *= LIGHT_DIAGONAL_FACTOR
            loss = LIGHT_SPREAD_LOSS + LIGHT_DIAGONAL_LOSS
        else:
            loss = LIGHT_SPREAD_LOSS

        nr = int(cr * factor) - loss
        ng = int(cg * factor) - loss
        nb = int(cb * factor) - loss

        if nr <= 2 and ng <= 2 and nb <= 2:
            return

        ni = ny * WORLD_W + nx
        changed = False

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
