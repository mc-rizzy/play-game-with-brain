import random
import pygame

from config import TILE
from tiles import *

ART = 20
PIXEL = TILE // ART          # 80 // 20 = 4


def px(s, x, y, w, h, colour):
    s.fill(colour, (x, y, w, h))


def cluster(s, points, colour):
    """Draw a group of connected pixels — avoids lonely-pixel noise."""
    for x, y, w, h in points:
        px(s, x, y, w, h, colour)


def make_small_tile(tid, rng):
    s = pygame.Surface((ART, ART), pygame.SRCALPHA)

    if tid == GRASS:
        # Dirt body with cool shadow, warm midtone
        px(s, 0, 0, 20, 20, (122, 79, 45))
        cluster(s, [(2,10,4,2),(12,13,5,2),(6,16,4,2),(15,8,3,2)], (104, 65, 36))
        cluster(s, [(4,8,3,1),(14,16,3,1),(1,14,2,1)], (142, 98, 58))
        # Grass cap: bright top, darker jagged overhang
        px(s, 0, 0, 20, 5, (88, 184, 64))
        px(s, 0, 0, 20, 2, (126, 214, 88))
        px(s, 0, 5, 20, 1, (58, 142, 44))
        for x in (1, 5, 9, 13, 17):
            px(s, x, 6, 2, rng.choice((2, 3)), (58, 142, 44))
        cluster(s, [(3,2,3,1),(11,3,4,1),(16,1,2,1)], (150, 226, 104))

    elif tid == DIRT:
        px(s, 0, 0, 20, 20, (128, 83, 49))
        cluster(s, [(2,3,5,3),(12,2,4,2),(5,11,6,3),(14,12,4,3),(2,16,4,2)], (108, 66, 38))
        cluster(s, [(4,7,3,1),(15,7,2,1),(9,15,3,1)], (150, 104, 64))
        px(s, 8, 5, 2, 2, (168, 124, 82)); px(s, 16, 15, 2, 1, (168, 124, 82))

    elif tid == STONE:
        px(s, 0, 0, 20, 20, (124, 126, 134))
        # Large rock slabs rather than noise
        cluster(s, [(1,8,8,2),(11,7,8,2),(4,15,9,2)], (98, 100, 110))
        cluster(s, [(1,1,7,2),(12,2,6,2),(3,11,5,2)], (146, 148, 156))
        px(s, 0, 19, 20, 1, (88, 90, 100))

    elif tid == WOOD:
        px(s, 0, 0, 20, 20, (128, 85, 44))
        # Bark columns with shadow grooves
        for x in (4, 10, 16):
            px(s, x, 0, 2, 20, (94, 58, 28))
        cluster(s, [(1,3,2,5),(7,9,2,6),(13,2,2,5)], (154, 110, 58))
        px(s, 0, 0, 20, 1, (168, 124, 70))

    elif tid == LEAVES:
        # Dark base showing through gaps, clustered leaf blobs on top
        px(s, 0, 0, 20, 20, (34, 96, 34))
        blobs = [(1,1),(7,0),(13,2),(3,7),(10,6),(16,8),(1,13),(8,12),(14,14),(5,17),(15,17)]
        for x, y in blobs:
            px(s, x, y, 5, 4, (58, 148, 50))
        for x, y in ((3,2),(11,1),(16,9),(6,13),(13,15)):
            px(s, x, y, 3, 2, (92, 188, 70))
        # A few transparent holes so leaves don't look like a solid slab
        for x, y in ((6,5),(17,4),(2,10),(12,10),(18,16)):
            s.fill((0, 0, 0, 0), (x, y, 2, 2))

    elif tid == SAND:
        px(s, 0, 0, 20, 20, (228, 204, 138))
        cluster(s, [(2,4,4,1),(11,3,5,1),(5,10,4,1),(14,12,4,1),(3,16,5,1)], (206, 178, 110))
        cluster(s, [(6,2,3,1),(15,7,3,1),(9,14,3,1)], (244, 222, 166))

    elif tid in (COAL, IRON, GOLD):
        px(s, 0, 0, 20, 20, (118, 120, 128))
        cluster(s, [(1,8,8,2),(11,13,8,2)], (94, 96, 106))
        cluster(s, [(2,2,6,2),(13,3,5,2)], (142, 144, 152))
        gem, gem_hi = {
            COAL: ((30, 30, 36), (62, 62, 72)),
            IRON: ((216, 168, 128), (246, 204, 164)),
            GOLD: ((240, 196, 60), (255, 232, 120)),
        }[tid]
        # Chunky gem clusters with a highlight corner
        for x, y in ((3, 4), (13, 3), (7, 12), (15, 14)):
            px(s, x, y, 4, 3, gem)
            px(s, x, y, 2, 1, gem_hi)

    elif tid == PLANKS:
        px(s, 0, 0, 20, 20, (186, 144, 88))
        for y in (6, 13):
            px(s, 0, y, 20, 2, (138, 100, 56))
        px(s, 10, 0, 2, 6, (138, 100, 56)); px(s, 5, 8, 2, 5, (138, 100, 56))
        px(s, 14, 15, 2, 5, (138, 100, 56))
        cluster(s, [(3,3,4,1),(13,10,4,1),(7,17,5,1)], (206, 168, 108))
        px(s, 2, 2, 1, 1, (110, 76, 42)); px(s, 17, 10, 1, 1, (110, 76, 42))  # nails

    elif tid == BRICK:
        px(s, 0, 0, 20, 20, (176, 80, 64))
        for y in (0, 6, 13):
            px(s, 0, y, 20, 2, (216, 200, 184))
        px(s, 10, 2, 2, 4, (216, 200, 184)); px(s, 5, 8, 2, 5, (216, 200, 184))
        px(s, 15, 15, 2, 5, (216, 200, 184))
        cluster(s, [(2,3,5,1),(13,9,4,1),(3,16,4,1)], (198, 104, 82))

    elif tid == GLASS:
        s.fill((172, 220, 232, 90))
        pygame.draw.rect(s, (238, 250, 252), (0, 0, 20, 20), 2)
        # Diagonal shine — two chunky steps
        px(s, 5, 13, 2, 5, (255, 255, 255)); px(s, 7, 11, 2, 2, (255, 255, 255))
        px(s, 9, 9, 2, 2, (255, 255, 255))

    elif tid == TORCH:
        px(s, 9, 10, 3, 9, (110, 72, 36))
        px(s, 8, 4, 5, 6, (255, 166, 48))
        px(s, 9, 2, 3, 4, (255, 226, 130))

    elif tid == LANTERN:
        px(s, 7, 1, 6, 2, (58, 50, 40))
        px(s, 6, 4, 8, 12, (255, 198, 82))
        px(s, 8, 7, 4, 6, (255, 244, 186))
        pygame.draw.rect(s, (58, 50, 40), (6, 4, 8, 12), 1)
        px(s, 6, 16, 8, 2, (58, 50, 40))

    elif tid == FLOWER:
        px(s, 9, 10, 2, 10, (54, 140, 52))
        px(s, 5, 14, 4, 2, (72, 166, 62))
        for x, y in ((8, 1), (4, 5), (13, 5), (8, 7)):
            px(s, x, y, 5, 4, (240, 100, 150))
        px(s, 9, 4, 3, 2, (255, 228, 94))

    elif tid == TALLGRASS:
        for x in (2, 7, 12, 16):
            top = rng.randint(5, 8)
            px(s, x, top, 2, 20 - top, (74, 168, 64))
            px(s, x, top, 2, 2, (104, 198, 84))

    elif tid == BEDROCK:
        px(s, 0, 0, 20, 20, (46, 46, 54))
        cluster(s, [(1,2,7,4),(12,1,6,5),(3,11,8,5),(13,12,6,4)], (74, 74, 84))
        cluster(s, [(3,3,3,1),(5,13,3,1)], (96, 96, 108))

    else:
        px(s, 0, 0, 20, 20, TILE_DEFS[tid][4])

    return s


def _scale(small):
    # transform.scale = hard nearest-neighbour edges, no smoothing
    return pygame.transform.scale(small, (TILE, TILE))


def make_sprites():
    sprites = {}

    for tid in TILE_DEFS:
        variants = []
        for variant in range(2):
            rng = random.Random(tid * 1000 + variant * 97)
            img = _scale(make_small_tile(tid, rng))
            if tid in TILE_ALPHA:
                img.set_alpha(TILE_ALPHA[tid])
            variants.append(img)
        sprites[tid] = variants

    # Water is drawn from its own animated frames
    sprites[WATER] = [make_water_frames()[0]]
    return sprites


def make_water_frames():
    """4-frame looping water: moving wave crests + rising bubble."""
    frames = []

    for frame in range(4):
        s = pygame.Surface((ART, ART), pygame.SRCALPHA)
        s.fill((44, 104, 196, 165))

        # Darker depths at the bottom
        px(s, 0, 15, 20, 5, (34, 82, 168, 165))

        # Two wave rows, shifted each frame
        for row, y in enumerate((4, 10)):
            offset = (frame * 3 + row * 5) % 20
            for start in (offset - 20, offset):
                px(s, start + 1, y, 6, 2, (116, 182, 240, 190))
                px(s, start + 2, y - 1, 3, 1, (170, 216, 250, 200))

        # Bubble rises one step per frame
        px(s, 14, 16 - frame * 3, 2, 2, (180, 224, 255, 180))

        frames.append(_scale(s))

    return frames