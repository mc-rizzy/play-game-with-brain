import math, random
from array import array
from game.config import (
    WORLD_W, WORLD_H, TILE, SUBCELLS, DROPS_PER_SUB,
    FLOW_DIFF_MIN, FLOW_MAX, FLOW_STRIDES, FLOW_TRIM_MAX,
    POUR_MIN, POUR_AMOUNT, DRIP_MAX,
)
from game.tiles import *

# Sub-grid dimensions: SUBCELLS x SUBCELLS water/light cells per block.
W2, H2 = WORLD_W * SUBCELLS, WORLD_H * SUBCELLS
SUB_TILE = TILE // SUBCELLS
TILE_DROPS = SUBCELLS * SUBCELLS * DROPS_PER_SUB


def _tile_solid(t):
    return t != AIR and TILE_DEFS[t][1]


class World:
    def __init__(self):
        # Terrain grid holds terrain only, never WATER: water lives on
        # the sub-grid below and get() reports wet tiles as WATER.
        self.g = [[AIR] * WORLD_H for _ in range(WORLD_W)]
        self.surface = [30] * WORLD_W
        # Drops of water per sub-cell (0..255) + per-tile totals
        # (0..1020, needs uint16). Fine quantum so leveling rests
        # flat to sub-pixel precision instead of stairstepping.
        self.ws = bytearray(W2 * H2)
        self.wtc = array('H', [0]) * (WORLD_W * WORLD_H)
        # Sub-columns holding water: the tick scans these instead of
        # the whole grid (spatial partition; physics still runs
        # unconditionally every tick).
        self.wet_cols = set()
        # Tiles holding water (exact write-through set, no laziness).
        self.wet_tiles = set()
        self._tick_n = 0
        # Edit plumbing for lighting: Lighting registers itself as the
        # listener and watches edit_counter for changes.
        self.edit_listener = None
        self.edit_counter = 0
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
                    if y >= 0:
                        self._fill_water_tile(x, y)
            elif rng.random() < .10:
                self.g[x][height - 1] = FLOWER if rng.random() < .4 else TALLGRASS
            if 8 < x < WORLD_W - 8 and rng.random() < .055 and not beach:
                self.tree(x, height, rng)
        self._build_houses()

    def _build_houses(self):
        sx = WORLD_W // 2
        self._house(sx - 19, door_side=1)
        self._house(sx + 11, door_side=-1)
        for x in (sx - 1, sx, sx + 1):
            for y in range(self.surface[x] - 8, self.surface[x]):
                if y >= 0 and self.g[x][y] in (WOOD, LEAVES):
                    self.set(x, y, AIR)

    def _house(self, x0, door_side):
        """Plank house with windows, stepped roof, chimney, torches.

        door_side faces spawn (+1: door on the right wall). All edits
        go through set() so beach water is displaced, not trapped.
        """
        W = 9
        fy = min(self.surface[x] for x in range(x0, x0 + W))
        for x in range(x0 - 1, x0 + W + 1):
            for y in range(fy - 11, fy):
                if y >= 0:
                    self.set(x, y, AIR)
        for x in range(x0, x0 + W):
            self.set(x, fy, PLANKS)
            for y in range(fy + 1, self.surface[x]):
                self.set(x, y, DIRT)
        for y in range(fy - 4, fy):
            self.set(x0, y, WOOD)
            self.set(x0 + W - 1, y, WOOD)
        door_x = x0 + W - 1 if door_side > 0 else x0
        back_x = x0 if door_side > 0 else x0 + W - 1
        for y in range(fy - 3, fy):
            self.set(door_x, y, AIR)
        self.set(door_x, fy - 4, GLASS)
        self.set(back_x, fy - 2, GLASS)
        spans = [(x0 - 1, x0 + W), (x0, x0 + W - 1),
                 (x0 + 1, x0 + W - 2), (x0 + 3, x0 + W - 4)]
        for i, (lo, hi) in enumerate(spans):
            for x in range(lo, hi + 1):
                self.set(x, fy - 5 - i,
                         PLANKS if i < 3 else WOOD)
        self.set(x0 + W - 3, fy - 9, BRICK)
        self.set(x0 + W - 3, fy - 8, BRICK)
        self.set(door_x + door_side, fy - 2, TORCH)
        bx_in = back_x - 1 if door_side > 0 else back_x + 1
        self.set(bx_in, fy - 2, TORCH)

    def tree(self, x, y, rng):
        trunk = rng.randint(4, 6)
        for i in range(1, trunk + 1): self.g[x][y - i] = WOOD
        for dx in range(-2, 3):
            for dy in range(-trunk - 2, -trunk + 1):
                if abs(dx) == 2 and dy == -trunk + 1: continue
                if 0 <= x + dx < WORLD_W and 0 <= y + dy: self.g[x + dx][y + dy] = LEAVES

    def get(self, x, y):
        if not (0 <= x < WORLD_W and 0 <= y < WORLD_H):
            return BEDROCK
        if self.wtc[y * WORLD_W + x]:
            return WATER
        return self.g[x][y]

    def solid(self, x, y):
        t = self.get(x, y)
        return TILE_DEFS[t][1] if t != AIR else False

    def opaque(self, x, y):
        t = self.get(x, y)
        return TILE_DEFS[t][2] if t != AIR else False

    def set(self, x, y, t):
        if not (0 <= x < WORLD_W and 0 <= y < WORLD_H):
            return
        old = self.g[x][y]
        if t == WATER:
            if _tile_solid(old):
                return
            if self.wtc[y * WORLD_W + x] == TILE_DROPS:
                return
            self._fill_water_tile(x, y)
            self.g[x][y] = AIR
            self.edit_counter += 1
            if self.edit_listener is not None:
                self.edit_listener(x, y, old, AIR)
            return
        if old == t and self.wtc[y * WORLD_W + x] == 0:
            return
        self._clear_water_tile(x, y)
        self.g[x][y] = t
        self.edit_counter += 1
        if self.edit_listener is not None:
            self.edit_listener(x, y, old, t)

    # ------------------------------------------------------------ water

    def _sync_tile(self, x, y):
        if self.wtc[y * WORLD_W + x]:
            self.wet_tiles.add((x, y))
        else:
            self.wet_tiles.discard((x, y))

    def _fill_water_tile(self, x, y):
        base = y * WORLD_W + x
        for dx in range(SUBCELLS):
            for dy in range(SUBCELLS):
                sx, sy = x * SUBCELLS + dx, y * SUBCELLS + dy
                i = sy * W2 + sx
                if self.ws[i] < DROPS_PER_SUB:
                    self.wtc[base] += DROPS_PER_SUB - self.ws[i]
                    self.ws[i] = DROPS_PER_SUB
                    self.wet_cols.add(sx)
        self._sync_tile(x, y)

    def _clear_water_tile(self, x, y):
        base = y * WORLD_W + x
        if not self.wtc[base]:
            return
        for dx in range(SUBCELLS):
            for dy in range(SUBCELLS):
                self.ws[(y * SUBCELLS + dy) * W2 + x * SUBCELLS + dx] = 0
        self.wtc[base] = 0
        self._sync_tile(x, y)

    def _move_count(self, ax, ay, bx, by, k):
        if ax == bx and ay == by:
            return
        self.wtc[ay * WORLD_W + ax] -= k
        self.wtc[by * WORLD_W + bx] += k
        self._sync_tile(ax, ay)
        self._sync_tile(bx, by)

    def water_at_pixel(self, px, py):
        """Exact sub-cell water test (half-tile swimming precision)."""
        sx, sy = int(px) // SUB_TILE, int(py) // SUB_TILE
        if 0 <= sx < W2 and 0 <= sy < H2:
            return self.ws[sy * W2 + sx] > 0
        return False

    def water_tick(self):
        """One unconditional flow step in three passes.

        1. GRAVITY (sub-cell): anything with empty space below falls.
        2. PRESSURE (tile pairs): supported water flows sideways toward
           lower neighbours until surfaces flatten to within one
           sub-unit, filling container shapes like Terraria/Noita.
        3. DRIP: lone sub-units at cliff edges step off and fall.
        Volume-conserving. Returns True if any water moved.
        """
        moved = False
        if self._gravity_pass():
            moved = True
        if self._pressure_pass():
            moved = True
        if self._drip_pass():
            moved = True
        self._tick_n += 1
        if self._tick_n % 8 == 0:
            self._prune_cols()
        return moved

    def _gravity_pass(self):
        moved = False
        g = self.g
        ws = self.ws
        cols = list(self.wet_cols)
        random.shuffle(cols)
        for sx in cols:
            tx = sx // SUBCELLS
            for sy in range(H2 - 2, -1, -1):
                i = sy * W2 + sx
                m = ws[i]
                if not m:
                    continue
                bi = i + W2
                room = DROPS_PER_SUB - ws[bi]
                bty = (sy + 1) // SUBCELLS
                if room > 0 and not _tile_solid(g[tx][bty]):
                    k = m if m < room else room
                    ws[i] = m - k
                    ws[bi] = ws[bi] + k
                    self._move_count(tx, sy // SUBCELLS, tx, bty, k)
                    moved = True
        return moved

    def _supported(self, ax, ay):
        """True if the wet run below rests on solid with no air gap."""
        y = ay
        while y < WORLD_H and self.wtc[y * WORLD_W + ax] > 0:
            y += 1
        if y >= WORLD_H:
            return True
        return _tile_solid(self.g[ax][y])

    def _transfer_drops(self, ax, ay, bx, by, k):
        """Move k drops from A's top to B's bottom. Returns drops moved."""
        if _tile_solid(self.g[bx][by]):
            return 0
        ws = self.ws
        ax0, ay0 = ax * SUBCELLS, ay * SUBCELLS
        bx0, by0 = bx * SUBCELLS, by * SUBCELLS
        room = 0
        for dy in range(SUBCELLS):
            for dx in range(SUBCELLS):
                room += DROPS_PER_SUB - ws[(by0 + dy) * W2 + bx0 + dx]
        k = min(k, room)
        if k <= 0:
            return 0
        # Take top-down (drain the surface first).
        remaining = k
        for dy in range(SUBCELLS):
            for dx in range(SUBCELLS):
                if not remaining:
                    break
                i = (ay0 + dy) * W2 + ax0 + dx
                take = ws[i] if ws[i] < remaining else remaining
                ws[i] -= take
                remaining -= take
        moved = k - remaining
        # Place bottom-up (stack from the floor).
        remaining = moved
        for dy in range(SUBCELLS - 1, -1, -1):
            for dx in range(SUBCELLS):
                if not remaining:
                    break
                i = (by0 + dy) * W2 + bx0 + dx
                space = DROPS_PER_SUB - ws[i]
                put = space if space < remaining else remaining
                ws[i] += put
                self.wet_cols.add(bx0 + dx)
                remaining -= put
        placed = moved - remaining
        self._move_count(ax, ay, bx, by, placed)
        return placed

    def _pressure_pass(self):
        moved = False
        tiles = list(self.wet_tiles)
        random.shuffle(tiles)
        for stride in FLOW_STRIDES:
            # Bulk flows adjacent tiles; longer strides gently trim
            # slopes that adjacent pairs can't see (multigrid-style).
            cap = FLOW_MAX if stride == 1 else FLOW_TRIM_MAX
            done = set()
            for (ax, ay) in tiles:
                for bx in (ax - stride, ax + stride):
                    if not (0 <= bx < WORLD_W):
                        continue
                    lo, hi = (ax, bx) if ax < bx else (bx, ax)
                    if (lo, ay) in done:
                        continue
                    done.add((lo, ay))
                    a = self.wtc[ay * WORLD_W + lo]
                    b = self.wtc[ay * WORLD_W + hi]
                    if not a and not b:
                        continue
                    if stride > 1:
                        # Long-range pressure only transmits through
                        # connected water (never through dams or air).
                        connected = True
                        for x in range(lo + 1, hi):
                            if not self.wtc[ay * WORLD_W + x]:
                                connected = False
                                break
                        if not connected:
                            continue
                    if a and b:
                        if a > b + FLOW_DIFF_MIN - 1:
                            src, tgt = (lo, ay), (hi, ay)
                        elif b > a + FLOW_DIFF_MIN - 1:
                            src, tgt = (hi, ay), (lo, ay)
                        else:
                            continue
                        if not self._supported(*src):
                            continue
                        diff = abs(a - b)
                        # Never more than half the difference: levels
                        # can't overshoot and slosh back.
                        if self._transfer_drops(*src, *tgt, min(cap, diff // 2)):
                            moved = True
                    elif stride == 1 and a and not _tile_solid(self.g[hi][ay]):
                        # Pour onto dry ground (needs a real puddle).
                        if a >= POUR_MIN and self._supported(lo, ay):
                            if self._transfer_drops(lo, ay, hi, ay, POUR_AMOUNT):
                                moved = True
                    elif stride == 1 and b and not _tile_solid(self.g[lo][ay]):
                        if b >= POUR_MIN and self._supported(hi, ay):
                            if self._transfer_drops(hi, ay, lo, ay, POUR_AMOUNT):
                                moved = True
        return moved

    def _drip_pass(self):
        # Small blobs at cliff edges tip off whole and fall.
        moved = False
        for (x, y) in list(self.wet_tiles):
            if self.wtc[y * WORLD_W + x] > DRIP_MAX:
                continue
            # Locate the blob (topmost wet sub-cell).
            ws = self.ws
            found = None
            for dy in range(SUBCELLS):
                for dx in range(SUBCELLS):
                    sx, sy = x * SUBCELLS + dx, y * SUBCELLS + dy
                    if ws[sy * W2 + sx]:
                        found = (sx, sy)
                        break
                if found is not None:
                    break
            if found is None:
                continue
            sx, sy = found
            # Only drip when supported (resting, not falling).
            below_solid_or_wet = (
                sy + 1 >= H2
                or ws[(sy + 1) * W2 + sx]
                or _tile_solid(self.g[sx // SUBCELLS][(sy + 1) // SUBCELLS])
            )
            if not below_solid_or_wet:
                continue
            dirs = (-1, 1) if random.random() < 0.5 else (1, -1)
            for dx in dirs:
                nx = sx + dx
                if not (0 <= nx < W2) or sy + 1 >= H2:
                    continue
                ni = sy * W2 + nx
                ntx = nx // SUBCELLS
                if ws[ni] or _tile_solid(self.g[ntx][sy // SUBCELLS]):
                    continue
                bi = ni + W2
                if ws[bi] or _tile_solid(self.g[ntx][(sy + 1) // SUBCELLS]):
                    continue
                k = ws[sy * W2 + sx]
                ws[sy * W2 + sx] = 0
                ws[ni] = k
                self._move_count(x, y, ntx, sy // SUBCELLS, k)
                self.wet_cols.add(nx)
                moved = True
                break
        return moved

    def _prune_cols(self):
        dead = [c for c in self.wet_cols
                if not any(self.ws[y * W2 + c] for y in range(H2))]
        for c in dead:
            self.wet_cols.discard(c)
