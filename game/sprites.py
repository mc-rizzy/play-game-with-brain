import math
import os
import random
import pygame

from game.config import TILE, ASSETS_DIR
from game.tiles import *

ART = 16
PIXEL = TILE // ART          # 80 // 16 = 5


def px(s, x, y, w, h, colour):
    s.fill(colour, (x, y, w, h))


def cluster(s, points, colour):
    """Draw a group of connected pixels — avoids lonely-pixel noise."""
    for x, y, w, h in points:
        px(s, x, y, w, h, colour)


def make_small_tile(tid, rng):
    """Author every tile on a strict 16x16 pixel grid (Terraria-style)."""
    s = pygame.Surface((ART, ART), pygame.SRCALPHA)

    if tid == GRASS:
        px(s, 0, 0, 16, 16, (122, 79, 45))
        cluster(s, [(1,8,3,2),(9,10,4,2),(4,13,3,2),(11,6,3,2)], (104, 65, 36))
        cluster(s, [(3,6,2,1),(11,13,2,1),(1,11,2,1)], (142, 98, 58))
        px(s, 0, 0, 16, 4, (88, 184, 64))
        px(s, 0, 0, 16, 2, (126, 214, 88))
        px(s, 0, 4, 16, 1, (58, 142, 44))
        for x in (1, 4, 7, 10, 13):
            px(s, x, 5, 2, rng.choice((2, 3)), (58, 142, 44))
        cluster(s, [(2,1,2,1),(9,2,3,1),(13,0,2,1)], (150, 226, 104))

    elif tid == DIRT:
        px(s, 0, 0, 16, 16, (128, 83, 49))
        cluster(s, [(1,2,4,2),(10,1,3,2),(4,8,5,3),(11,10,3,2),(1,13,3,2)], (108, 66, 38))
        cluster(s, [(3,5,2,1),(12,5,2,1),(7,12,2,1)], (150, 104, 64))
        px(s, 6, 4, 2, 2, (168, 124, 82)); px(s, 13, 12, 1, 1, (168, 124, 82))

    elif tid == STONE:
        px(s, 0, 0, 16, 16, (124, 126, 134))
        cluster(s, [(1,6,6,2),(9,5,6,2),(3,12,7,2)], (98, 100, 110))
        cluster(s, [(1,1,5,2),(10,2,5,1),(2,9,4,1)], (146, 148, 156))
        px(s, 0, 15, 16, 1, (88, 90, 100))

    elif tid == WOOD:
        px(s, 0, 0, 16, 16, (128, 85, 44))
        for x in (3, 8, 13):
            px(s, x, 0, 1, 16, (94, 58, 28))
        cluster(s, [(0,2,2,4),(5,7,2,5),(10,1,2,4)], (154, 110, 58))
        px(s, 0, 0, 16, 1, (168, 124, 70))

    elif tid == LEAVES:
        px(s, 0, 0, 16, 16, (34, 96, 34))
        blobs = [(0,0),(6,0),(11,1),(2,5),(8,4),(12,6),(0,10),(6,9),(11,11),(4,13)]
        for x, y in blobs:
            px(s, x, y, 4, 3, (58, 148, 50))
        for x, y in ((2,1),(9,0),(13,7),(5,10),(10,12)):
            px(s, x, y, 2, 1, (92, 188, 70))
        for x, y in ((5,3),(14,3),(1,8),(10,8),(14,12)):
            s.fill((0, 0, 0, 0), (x, y, 1, 1))

    elif tid == SAND:
        px(s, 0, 0, 16, 16, (228, 204, 138))
        cluster(s, [(1,3,3,1),(9,2,4,1),(4,8,3,1),(11,10,3,1),(2,13,4,1)], (206, 178, 110))
        cluster(s, [(5,1,2,1),(12,6,2,1),(7,11,2,1)], (244, 222, 166))

    elif tid in (COAL, IRON, GOLD):
        px(s, 0, 0, 16, 16, (118, 120, 128))
        cluster(s, [(1,6,6,2),(9,10,6,2)], (94, 96, 106))
        cluster(s, [(1,1,5,2),(10,2,4,1)], (142, 144, 152))
        gem, gem_hi = {
            COAL: ((30, 30, 36), (62, 62, 72)),
            IRON: ((216, 168, 128), (246, 204, 164)),
            GOLD: ((240, 196, 60), (255, 232, 120)),
        }[tid]
        for x, y in ((2, 3), (10, 2), (5, 9), (11, 11)):
            px(s, x, y, 3, 3, gem)
            px(s, x, y, 2, 1, gem_hi)

    elif tid == PLANKS:
        px(s, 0, 0, 16, 16, (186, 144, 88))
        for y in (5, 11):
            px(s, 0, y, 16, 1, (138, 100, 56))
        px(s, 8, 0, 1, 5, (138, 100, 56)); px(s, 4, 6, 1, 5, (138, 100, 56))
        px(s, 11, 12, 1, 4, (138, 100, 56))
        cluster(s, [(2,2,3,1),(10,8,3,1),(5,13,4,1)], (206, 168, 108))
        px(s, 1, 1, 1, 1, (110, 76, 42)); px(s, 14, 8, 1, 1, (110, 76, 42))

    elif tid == BRICK:
        px(s, 0, 0, 16, 16, (176, 80, 64))
        for y in (0, 5, 10, 15):
            px(s, 0, y, 16, 1, (216, 200, 184))
        px(s, 8, 1, 1, 4, (216, 200, 184)); px(s, 4, 6, 1, 4, (216, 200, 184))
        px(s, 12, 6, 1, 4, (216, 200, 184)); px(s, 8, 11, 1, 4, (216, 200, 184))
        cluster(s, [(1,2,4,1),(10,7,3,1),(2,12,3,1)], (198, 104, 82))

    elif tid == GLASS:
        s.fill((172, 220, 232, 90))
        pygame.draw.rect(s, (238, 250, 252), (0, 0, 16, 16), 1)
        px(s, 4, 10, 2, 4, (255, 255, 255)); px(s, 6, 8, 2, 2, (255, 255, 255))

    elif tid == TORCH:
        px(s, 7, 8, 2, 8, (110, 72, 36))
        px(s, 7, 8, 1, 8, (84, 54, 26))
        px(s, 6, 3, 4, 5, (255, 166, 48))
        px(s, 7, 1, 2, 4, (255, 226, 130))

    elif tid == LANTERN:
        px(s, 6, 1, 4, 1, (58, 50, 40))
        px(s, 5, 3, 6, 10, (255, 198, 82))
        px(s, 6, 5, 4, 6, (255, 244, 186))
        pygame.draw.rect(s, (58, 50, 40), (5, 3, 6, 10), 1)
        px(s, 5, 13, 6, 2, (58, 50, 40))

    elif tid == FLOWER:
        px(s, 7, 8, 2, 8, (54, 140, 52))
        px(s, 4, 11, 3, 2, (72, 166, 62))
        for x, y, w, h in ((6, 0, 4, 3), (3, 3, 4, 3), (9, 3, 4, 3), (6, 6, 4, 2)):
            px(s, x, y, w, h, (240, 100, 150))
        px(s, 7, 3, 2, 2, (255, 228, 94))

    elif tid == TALLGRASS:
        for x in (1, 5, 9, 12):
            top = rng.randint(4, 6)
            px(s, x, top, 2, 16 - top, (74, 168, 64))
            px(s, x, top, 2, 2, (104, 198, 84))

    elif tid == BEDROCK:
        px(s, 0, 0, 16, 16, (46, 46, 54))
        cluster(s, [(1,1,5,3),(10,1,5,4),(2,9,6,4),(10,10,5,3)], (74, 74, 84))
        cluster(s, [(2,2,2,1),(4,10,2,1)], (96, 96, 108))

    else:
        px(s, 0, 0, 16, 16, TILE_DEFS[tid][4])

    return s


def make_still_water():
    """Flat blue 16x16 for water the player isn't disturbing."""
    s = pygame.Surface((ART, ART), pygame.SRCALPHA)
    s.fill((44, 104, 196, 165))
    return s


# Celestial art lives on the same 16x16 pixel grid as tiles.
# Sun: Y edge, M mid, L hot core ('.' transparent).
SUN_MAP = (
    ".....YYYYYY.....",
    "...YYMMMMMMYY...",
    "..YMMMMMMMMMMY..",
    ".YMMMMMMMMMMMMY.",
    ".YMMMLLLLLLMMMY.",
    "YMMMLLLLLLLLMMMY",
    "YMMMLLLLLLLLMMMY",
    "YMMMLLLLLLLLMMMY",
    "YMMMLLLLLLLLMMMY",
    "YMMMLLLLLLLLMMMY",
    "YMMMLLLLLLLLMMMY",
    ".YMMMLLLLLLMMMY.",
    ".YMMMMMMMMMMMMY.",
    "..YMMMMMMMMMMY..",
    "...YYMMMMMMYY...",
    ".....YYYYYY.....",
)

# Clouds: H highlight caps, W body, S shaded underside and shadow
# edge ('.' transparent). Rows need not be equal length.
CLOUD_MAPS = (
    (
        "............HHHHHH............",
        "..........HHWWWWWWHH..........",
        "........HWWWWWWWWWWWH.........",
        "......HWWWWWWWWWWWWWWHH.......",
        "....HWWWWWWWWWWWWWWWWWWS......",
        "...HWWWWWWWWWWWWWWWWWWWSS.....",
        "..HWWWWWWWWWWWWWWWWWWWWWSS....",
        "..WWWWWWWWWWWWWWWWWWWWWWWSS...",
        "..SSSSSSSSSSSSSSSSSSSSSSSSSS..",
    ),
    (
        ".........HHHHHH.........",
        ".......HHWWWWWWHH.......",
        ".....HWWWWWWWWWWHH......",
        "...HWWWWWWWWWWWWWWS.....",
        "..HWWWWWWWWWWWWWWWSS....",
        "..WWWWWWWWWWWWWWWWSS....",
        "..SSSSSSSSSSSSSSSSSS....",
    ),
    (
        "...............HHHHHHH.............",
        "...........HHHWWWWWWWWWHHH.........",
        "......HHHWWWWWWWWWWWWWWWWWHHH......",
        "...HHWWWWWWWWWWWWWWWWWWWWWWWWHH....",
        "..HWWWWWWWWWWWWWWWWWWWWWWWWWWWWS...",
        "..WWWWWWWWWWWWWWWWWWWWWWWWWWWWWSS..",
        "..SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS..",
    ),
)


def paint_map(rows, palette):
    """Paint glyph rows to a grid surface ('.' stays transparent)."""
    w = max(len(r) for r in rows)
    s = pygame.Surface((w, len(rows)), pygame.SRCALPHA)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != ".":
                s.fill(palette[ch], (x, y, 1, 1))
    return s


def moon_surface():
    """Crescent moon on the 16x16 grid, three fixed shades."""
    pale, mid, crater = (230, 236, 250), (208, 216, 234), (186, 196, 218)
    s = pygame.Surface((ART, ART), pygame.SRCALPHA)
    for y in range(ART):
        for x in range(ART):
            d1 = math.hypot(x + 0.5 - 7.5, y + 0.5 - 7.5)
            d2 = math.hypot(x + 0.5 - 10.5, y + 0.5 - 5.5)
            if d1 <= 7.2 and d2 > 6.0:
                col = mid if d2 <= 7.4 else pale
                s.fill(col, (x, y, 1, 1))
    for x, y in ((4, 5), (6, 9), (3, 10)):
        if s.get_at((x, y))[3]:
            s.fill(crater, (x, y, 1, 1))
    return s


def _scale(small):
    # transform.scale = hard nearest-neighbour edges, no smoothing
    return pygame.transform.scale(small, (TILE, TILE))


def _convert(surface):
    """Convert to display format for faster blits (no-op headless)."""
    try:
        if pygame.display.get_surface() is None:
            return surface
        if surface.get_flags() & pygame.SRCALPHA:
            return surface.convert_alpha()
        return surface.convert()
    except pygame.error:
        return surface


def make_sprites():
    sprites = {}

    for tid in TILE_DEFS:
        variants = []
        for variant in range(2):
            rng = random.Random(tid * 1000 + variant * 97)
            img = _scale(make_small_tile(tid, rng))
            if tid in TILE_ALPHA:
                img.set_alpha(TILE_ALPHA[tid])
            variants.append(_convert(img))
        sprites[tid] = variants

    # Water is drawn from its own animated frames
    sprites[WATER] = [make_water_frames()[0]]
    return sprites


def make_water_frames():
    """4-frame looping water on a 16x16 grid: waves + rising bubble."""
    frames = []

    for frame in range(4):
        s = pygame.Surface((ART, ART), pygame.SRCALPHA)
        s.fill((44, 104, 196, 165))

        # Darker depths at the bottom
        px(s, 0, 12, 16, 4, (34, 82, 168, 165))

        # Two wave rows, shifted each frame
        for row, y in enumerate((3, 8)):
            offset = (frame * 2 + row * 6) % 16
            for start in (offset - 16, offset):
                px(s, start + 1, y, 5, 1, (116, 182, 240, 190))
                px(s, start + 2, y - 1, 2, 1, (170, 216, 250, 200))

        # Bubble rises one step per frame
        px(s, 11, 11 - frame * 2, 1, 1, (180, 224, 255, 180))

        frames.append(_convert(_scale(s)))

    return frames


def _load_png(name):
    """Load one PNG from game/assets. Returns None when missing."""
    path = os.path.join(ASSETS_DIR, name)
    try:
        if not os.path.isfile(path):
            return None
        return pygame.image.load(path).convert_alpha()
    except (pygame.error, OSError):
        return None


def _flash_variant(surface):
    """Red hurt-flash copy of an entity frame."""
    flashed = surface.copy()
    flashed.fill((255, 90, 90), special_flags=pygame.BLEND_RGB_ADD)
    return flashed


def make_entity_sprites():
    """Load character/entity pixel art from game/assets/*.png.

    Returns a dict of display-sized frames (nearest-neighbour scaled):
    player_idle/walk1/walk2 (+ flipped + flash variants), slime_0/1,
    shade_0/1, bat_0/1, skeleton_0/1 (+ flash variants), sword + hearts
    at native size. Missing files fall back to a solid placeholder so
    the game still runs.
    """
    def placeholder(w, h, colour=(200, 80, 200)):
        s = pygame.Surface((max(1, w), max(1, h)), pygame.SRCALPHA)
        s.fill(colour)
        pygame.draw.rect(s, (30, 30, 36), s.get_rect(), 1)
        return s

    def sized(name, w, h):
        small = _load_png(name)
        if small is None:
            return _convert(placeholder(w, h))
        return _convert(pygame.transform.scale(small, (max(1, w), max(1, h))))

    # Integer scales of the 16px art so pixels stay uniform: player
    # 16x24 at 5x (same pixel size as tiles), slime 16x16 and shade
    # 16x32 at 4x, bat 16x16 at 4x, skeleton 16x24 at 5x.
    # Blits are bottom-aligned so feet touch the ground.
    pw, ph = 16 * 5, 24 * 5
    sw, sh = 16 * 4, 16 * 4
    hw, hh = 16 * 4, 32 * 4
    bw, bh = 16 * 4, 16 * 4
    kw, kh = 16 * 5, 24 * 5

    out = {}
    for key in ("player_idle", "player_walk1", "player_walk2"):
        img = sized(f"{key}.png", pw, ph)
        out[key] = img
        out[f"{key}_left"] = _convert(pygame.transform.flip(img, True, False))
        out[f"{key}_flash"] = _convert(_flash_variant(img))
        out[f"{key}_left_flash"] = _convert(
            pygame.transform.flip(out[f"{key}_flash"], True, False))
    for key, w, h in (("slime_0", sw, sh), ("slime_1", sw, sh),
                      ("shade_0", hw, hh), ("shade_1", hw, hh),
                      ("bat_0", bw, bh), ("bat_1", bw, bh),
                      ("skeleton_0", kw, kh), ("skeleton_1", kw, kh)):
        img = sized(f"{key}.png", w, h)
        out[key] = img
        out[f"{key}_flash"] = _convert(_flash_variant(img))

    out["sword"] = _load_png("sword.png") or placeholder(14, 14)
    out["heart_full"] = _load_png("heart_full.png") or placeholder(10, 9)
    out["heart_empty"] = _load_png("heart_empty.png") or placeholder(10, 9)
    return out