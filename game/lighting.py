"""Sub-tile sun + torch lighting with continuous physics.

Resolution: 2x2 sub-cells per block (each 1/4 block in area). No dirty
flags, no lazy updates, no full-world recomputes in the hot path:

- SUN is analytic: each half-column is marched from the sky down every
  time a block in it changes, so shadows are exact and update in the
  same tick as the edit. Daylight intensity (sunval) scales at render
  time, so dawn/dusk are smooth and free.
- TORCH light is a continuous flood: every frame drains a budgeted
  propagation queue (light visibly sweeps in after edits instead of
  popping). Emissive blocks are registered in a source set; darkening
  edits reset the affected region so stale glow can't persist.
- Water is optically clear to light (it never dims or shadows), which
  keeps the water sim and the light sim independent.
"""

import math
from array import array
from collections import deque

from game.config import (
    WORLD_W, WORLD_H, SUBCELLS, LIGHT_MAX,
    TORCH_LOSS, RELAX_FRAME_BUDGET, RELAX_EDIT_TOPUP,
    RESET_R_TORCH,
)
from game.tiles import AIR, TILE_DEFS, light_factor
from game.world import W2, H2

# Per-sub-cell transmission: sqrt of the tile factor, so two sub-steps
# equal one old tile step (0.55^2 ~= 0.3). Fixed-point 0..256 for the
# torch flood, float for the analytic sun march.
_SUBF = {}
_SUBF_F = {}


def _sub_factor(tile_id):
    f = _SUBF.get(tile_id)
    if f is None:
        f = int(math.sqrt(max(0.0, min(1.0, light_factor(tile_id)))) * 256)
        _SUBF[tile_id] = f
    return f


def _sub_factor_f(tile_id):
    f = _SUBF_F.get(tile_id)
    if f is None:
        f = math.sqrt(max(0.0, min(1.0, light_factor(tile_id))))
        _SUBF_F[tile_id] = f
    return f


def _emission(tile_id):
    if tile_id == AIR:
        return (0, 0, 0)
    e = TILE_DEFS[tile_id][5]
    return (e[0] * 17, e[1] * 17, e[2] * 17)


def _emits(tile_id):
    if tile_id == AIR:
        return False
    return TILE_DEFS[tile_id][5] != (0, 0, 0)


class Lighting:
    def __init__(self, world):
        self.world = world
        self.count = W2 * H2

        # Full-sun solution (geometry only; scaled by sunval at read).
        self.sun = bytearray(self.count)
        # Torch/emission flood.
        self.r = bytearray(self.count)
        self.g = bytearray(self.count)
        self.b = bytearray(self.count)
        # Fixed-point transmission per sub-cell, 0..256 (mirrors
        # terrain; needs uint16 since clear air is exactly 256).
        self.subf = array('H', [0]) * self.count

        self.sunval = LIGHT_MAX
        self.sources = {}
        self._scan_sources()
        self._build_subf()
        self._solve_sun_full()

        self._queue = deque()
        self.version = 0
        self.display_version = 0
        self._solved_edits = -1
        # Self-register: every tile edit flows here, no flags needed.
        world.edit_listener = self.on_world_edit

    # ------------------------------------------------------------ setup

    def _scan_sources(self):
        g = self.world.g
        for x in range(WORLD_W):
            col = g[x]
            for y in range(WORLD_H):
                if _emits(col[y]):
                    self.sources[(x, y)] = _emission(col[y])

    def _build_subf(self):
        g = self.world.g
        subf = self.subf
        for tx in range(WORLD_W):
            col = g[tx]
            for ty in range(WORLD_H):
                f = _sub_factor(col[ty])
                base = (ty * SUBCELLS) * W2 + tx * SUBCELLS
                for dy in range(SUBCELLS):
                    row = base + dy * W2
                    for dx in range(SUBCELLS):
                        subf[row + dx] = f

    # --------------------------------------------------------------- sun

    def _solve_sun_column(self, tx):
        """March one tile column from the sky (exact, ~30us)."""
        g = self.world.g
        sun = self.sun
        for dx in range(SUBCELLS):
            sx = tx * SUBCELLS + dx
            if not (0 <= sx < W2):
                continue
            level = 255.0
            for sy in range(H2):
                if sy > 0:
                    level *= _sub_factor_f(g[tx][sy // SUBCELLS])
                if level < 1.0:
                    for yy in range(sy, H2):
                        sun[yy * W2 + sx] = 0
                    break
                sun[sy * W2 + sx] = int(level)

    def _solve_sun_full(self):
        for tx in range(WORLD_W):
            self._solve_sun_column(tx)

    # ------------------------------------------------------- torch flood

    def _seed_torch_tile(self, tx, ty):
        e = self.sources.get((tx, ty))
        if e is None:
            return
        er, eg, eb = e
        r, g, b = self.r, self.g, self.b
        q = self._queue
        for dx in range(SUBCELLS):
            for dy in range(SUBCELLS):
                i = (ty * SUBCELLS + dy) * W2 + tx * SUBCELLS + dx
                pushed = False
                if er > r[i]:
                    r[i] = er
                    pushed = True
                if eg > g[i]:
                    g[i] = eg
                    pushed = True
                if eb > b[i]:
                    b[i] = eb
                    pushed = True
                if pushed:
                    q.append(i)

    def _relax(self, budget):
        """Drain up to `budget` propagation steps. Returns cells raised."""
        r, g, b = self.r, self.g, self.b
        subf = self.subf
        q = self._queue
        loss = TORCH_LOSS
        changed = 0
        pops = 0
        while q and pops < budget:
            i = q.popleft()
            pops += 1
            cr = r[i]
            cg = g[i]
            cb = b[i]
            if cr < 2 and cg < 2 and cb < 2:
                continue
            x = i % W2
            # Left / right / up / down sub-neighbours.
            if x > 0:
                ni = i - 1
                f = subf[ni]
                nr = ((cr * f) >> 8) - loss
                ng = ((cg * f) >> 8) - loss
                nb = ((cb * f) >> 8) - loss
                pushed = False
                if nr > r[ni] + 1:
                    r[ni] = nr
                    pushed = True
                if ng > g[ni] + 1:
                    g[ni] = ng
                    pushed = True
                if nb > b[ni] + 1:
                    b[ni] = nb
                    pushed = True
                if pushed:
                    q.append(ni)
                    changed += 1
            if x < W2 - 1:
                ni = i + 1
                f = subf[ni]
                nr = ((cr * f) >> 8) - loss
                ng = ((cg * f) >> 8) - loss
                nb = ((cb * f) >> 8) - loss
                pushed = False
                if nr > r[ni] + 1:
                    r[ni] = nr
                    pushed = True
                if ng > g[ni] + 1:
                    g[ni] = ng
                    pushed = True
                if nb > b[ni] + 1:
                    b[ni] = nb
                    pushed = True
                if pushed:
                    q.append(ni)
                    changed += 1
            if i >= W2:
                ni = i - W2
                f = subf[ni]
                nr = ((cr * f) >> 8) - loss
                ng = ((cg * f) >> 8) - loss
                nb = ((cb * f) >> 8) - loss
                pushed = False
                if nr > r[ni] + 1:
                    r[ni] = nr
                    pushed = True
                if ng > g[ni] + 1:
                    g[ni] = ng
                    pushed = True
                if nb > b[ni] + 1:
                    b[ni] = nb
                    pushed = True
                if pushed:
                    q.append(ni)
                    changed += 1
            if i < W2 * (H2 - 1):
                ni = i + W2
                f = subf[ni]
                nr = ((cr * f) >> 8) - loss
                ng = ((cg * f) >> 8) - loss
                nb = ((cb * f) >> 8) - loss
                pushed = False
                if nr > r[ni] + 1:
                    r[ni] = nr
                    pushed = True
                if ng > g[ni] + 1:
                    g[ni] = ng
                    pushed = True
                if nb > b[ni] + 1:
                    b[ni] = nb
                    pushed = True
                if pushed:
                    q.append(ni)
                    changed += 1
        return changed

    # -------------------------------------------------------------- edits

    def on_world_edit(self, tx, ty, old, new):
        """Every tile edit flows here (registered as world's listener)."""
        # Sun is column-local: recompute the edited column eagerly so
        # shadows update in the same tick as the edit.
        self._solve_sun_column(tx)
        # Transmission mirror follows the terrain.
        f = _sub_factor(new)
        base = ty * SUBCELLS * W2 + tx * SUBCELLS
        for dy in range(SUBCELLS):
            row = base + dy * W2
            for dx in range(SUBCELLS):
                self.subf[row + dx] = f
        # Source registry follows emissive blocks.
        if _emits(new):
            self.sources[(tx, ty)] = _emission(new)
        else:
            self.sources.pop((tx, ty), None)

        removed_light = sum(_emission(new)) < sum(_emission(old))
        if removed_light or self._sources_near(tx, ty, 50):
            self._torch_edit(tx, ty, old, new, removed_light)
        self.version += 1
        self.display_version = self.version

    def _sources_near(self, tx, ty, radius):
        for (sx, sy) in self.sources:
            if abs(sx - tx) <= radius and abs(sy - ty) <= radius:
                return True
        return False

    def _torch_edit(self, tx, ty, old, new, removed_light):
        darkens = (removed_light
                   or _sub_factor(new) < _sub_factor(old))
        if darkens:
            # Stale-bright torch cells can't un-relax, so reset the
            # affected region and let continuous physics refill it.
            self._reset_box(tx, ty, RESET_R_TORCH)
            self._seed_region(tx, ty, RESET_R_TORCH)
        else:
            # Brightening only: seed the boundary, raising does the rest.
            self._push_ring(tx, ty, RESET_R_TORCH)
        if (tx, ty) in self.sources:
            self._seed_torch_tile(tx, ty)
        self._relax(RELAX_EDIT_TOPUP)

    def _reset_box(self, tx, ty, radius):
        x0 = max(0, (tx - radius) * SUBCELLS)
        x1 = min(W2, (tx + radius + 1) * SUBCELLS)
        y0 = max(0, (ty - radius) * SUBCELLS)
        y1 = min(H2, (ty + radius + 1) * SUBCELLS)
        width = x1 - x0
        if width <= 0 or y0 >= y1:
            return
        blank = bytes(width)
        for grid in (self.r, self.g, self.b):
            for sy in range(y0, y1):
                a = sy * W2 + x0
                grid[a:a + width] = blank

    def _seed_region(self, tx, ty, radius):
        for (sx, syy) in self.sources:
            if abs(sx - tx) <= radius and abs(syy - ty) <= radius:
                self._seed_torch_tile(sx, syy)
        self._push_ring(tx, ty, radius)

    def _push_ring(self, tx, ty, radius):
        """Push the correct-valued ring around a box so flow continues."""
        x0 = max(0, (tx - radius) * SUBCELLS)
        x1 = min(W2, (tx + radius + 1) * SUBCELLS)
        y0 = max(0, (ty - radius) * SUBCELLS)
        y1 = min(H2, (ty + radius + 1) * SUBCELLS)
        r, g, b = self.r, self.g, self.b
        q = self._queue

        def push(i):
            if r[i] > 1 or g[i] > 1 or b[i] > 1:
                q.append(i)

        if x0 > 0:
            for sy in range(y0, y1):
                push(sy * W2 + x0 - 1)
        if x1 < W2:
            for sy in range(y0, y1):
                push(sy * W2 + x1)
        if y0 > 0:
            base = (y0 - 1) * W2
            for sx in range(x0, x1):
                push(base + sx)
        if y1 < H2:
            base = y1 * W2
            for sx in range(x0, x1):
                push(base + sx)

    # --------------------------------------------------------------- API

    def tick(self, budget=RELAX_FRAME_BUDGET):
        """Advance torch physics one budgeted step. Returns changed."""
        if self._relax(budget):
            self.version += 1
            self.display_version = self.version
            return True
        return False

    def update_display(self, dt):
        """Advance the light sim (called every frame)."""
        return self.tick()

    def compute(self, daylight):
        """Set sun intensity; drain torch physics to convergence if the
        world changed since the last solve. Returns True if solved."""
        self.sunval = int(max(0.0, min(1.0, daylight)) * LIGHT_MAX)
        if self.world.edit_counter == self._solved_edits and not self._queue:
            return False
        for (sx, sy) in self.sources:
            self._seed_torch_tile(sx, sy)
        pops = 0
        while self._queue and pops < 1000000:
            self._relax(50000)
            pops += 50000
        self._solved_edits = self.world.edit_counter
        self.version += 1
        self.display_version = self.version
        return True

    def sun_smooth(self, sx, sy):
        """Sun value blended with horizontal neighbours ([1,2,1]).

        Each sub-cell observes the cells left and right of it, so
        light softens laterally around shadow edges instead of forming
        hard vertical stripes. Analytic (a pure function of the exact
        column grid), so edits still read instantly with no lag state.
        """
        row = sy * W2
        c = self.sun[row + sx]
        left = self.sun[row + sx - 1] if sx > 0 else c
        right = self.sun[row + sx + 1] if sx < W2 - 1 else c
        return (left + (c << 1) + right) >> 2

    def _combined_sub(self, sx, sy):
        s = (self.sun_smooth(sx, sy) * self.sunval) // LIGHT_MAX
        i = sy * W2 + sx
        r = s + self.r[i]
        g = s + self.g[i]
        b = (s * 96) // 100 + self.b[i]
        return (r if r <= 255 else 255,
                g if g <= 255 else 255,
                b if b <= 255 else 255)

    def light_at(self, x, y):
        if not (0 <= x < WORLD_W and 0 <= y < WORLD_H):
            return (0, 0, 0)
        best_r = best_g = best_b = 0
        for dx in range(SUBCELLS):
            for dy in range(SUBCELLS):
                r, g, b = self._combined_sub(x * SUBCELLS + dx,
                                             y * SUBCELLS + dy)
                if r > best_r:
                    best_r = r
                if g > best_g:
                    best_g = g
                if b > best_b:
                    best_b = b
        return (best_r, best_g, best_b)

    def light_at_display(self, x, y):
        # The display reads the live sim directly: no lag buffers.
        return self.light_at(x, y)

    def brightness_at(self, x, y):
        r, g, b = self.light_at(x, y)
        return max(r, g, b)
