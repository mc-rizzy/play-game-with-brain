from config import SOLID_LIGHT_FACTOR

AIR, GRASS, DIRT, STONE, WOOD, LEAVES, SAND, WATER = range(8)
COAL, IRON, GOLD, PLANKS, BRICK, GLASS, TORCH, LANTERN = range(8, 16)
FLOWER, TALLGRASS, BEDROCK = 16, 17, 18
SWORD = 100

# name, solid, opaque, hardness(sec), base colour, emitted light (r,g,b 0..15)
TILE_DEFS = {
    GRASS:    ("Grass",    True,  True,  .35, (96,176,72),   (0,0,0)),
    DIRT:     ("Dirt",     True,  True,  .30, (134,88,54),   (0,0,0)),
    STONE:    ("Stone",    True,  True,  .90, (128,128,136), (0,0,0)),
    WOOD:     ("Wood Log", True,  True,  .60, (120,82,46),   (0,0,0)),
    LEAVES:   ("Leaves",   True,  False, .15, (62,148,58),   (0,0,0)),
    SAND:     ("Sand",     True,  True,  .30, (226,204,140), (0,0,0)),
    WATER:    ("Water",    False, False, 9e9, (52,110,200),  (0,0,0)),
    COAL:     ("Coal Ore", True,  True,  1.1, (90,90,96),    (0,0,0)),
    IRON:     ("Iron Ore", True,  True,  1.4, (150,130,118), (0,0,0)),
    GOLD:     ("Gold Ore", True,  True,  1.7, (196,164,70),  (0,0,0)),
    PLANKS:   ("Planks",   True,  True,  .50, (188,148,92),  (0,0,0)),
    BRICK:    ("Brick",    True,  True,  1.0, (176,84,68),   (0,0,0)),
    GLASS:    ("Glass",    True,  False, .20, (190,225,235), (0,0,0)),
    TORCH:   ("Torch",   False, False, .05, (255, 170, 60), (14, 9, 4)),
    LANTERN: ("Lantern", False, False, .20, (255, 200, 90), (15, 12, 6)),
    FLOWER:   ("Flower",   False, False, .05, (230,90,140),  (0,0,0)),
    TALLGRASS:("Tall Grass",False,False, .05, (90,180,80),   (0,0,0)),
    BEDROCK:  ("Bedrock",  True,  True,  9e9, (50,50,56),    (0,0,0)),
}
PLACEABLE = [DIRT, STONE, PLANKS, WOOD, BRICK, GLASS, TORCH, LANTERN, FLOWER]
HOTBAR    = [DIRT, STONE, PLANKS, WOOD, BRICK, GLASS, TORCH, LANTERN, SWORD]
ITEM_NAMES = {**{k: v[0] for k, v in TILE_DEFS.items()}, SWORD: "Wooden Sword"}

RECIPES = [
    ("Planks x4",    PLANKS, 4, {WOOD: 1}),
    ("Torch x4",     TORCH,  4, {WOOD: 1, COAL: 1}),
    ("Lantern",      LANTERN,1, {WOOD: 2, TORCH: 1, IRON: 1}),
    ("Brick x2",     BRICK,  2, {STONE: 2}),
    ("Glass x2",     GLASS,  2, {SAND: 2}),
    ("Wooden Sword", SWORD,  1, {WOOD: 3}),
]


# How much light is lost when entering this tile.
# Opaque tiles are handled specially by the lighting engine:
# light may illuminate their exposed face, but cannot pass through.
def light_attenuation(tile_id):
    if tile_id == LEAVES:
        return 3
    if tile_id == WATER:
        return 2
    return 1


def is_opaque(tile_id):
    if tile_id == AIR:
        return False
    return TILE_DEFS[tile_id][2]


def emitted_light(tile_id):
    if tile_id == AIR:
        return (0, 0, 0)
    return TILE_DEFS[tile_id][5]

# ---------------------------------------------------------------
# Lighting factors
#
# 1.0  = almost no extra loss
# 0.5  = keeps roughly half the light
# 0.1  = keeps very little light
#
# These override SOLID_LIGHT_FACTOR for specific blocks.
# ---------------------------------------------------------------

TILE_LIGHT_FACTOR = {
    GLASS:     0.94,
    LEAVES:    0.68,
    WATER:     0.78,

    GRASS:     0.45,
    DIRT:      0.46,
    SAND:      0.50,
    PLANKS:    0.50,
    WOOD:      0.40,

    STONE:     0.30,
    COAL:      0.30,
    IRON:      0.30,
    GOLD:      0.30,
    BRICK:     0.32,
    BEDROCK:   0.05,
}


def light_factor(tile_id):
    """How much light remains after entering this tile."""
    if tile_id in TILE_LIGHT_FACTOR:
        return TILE_LIGHT_FACTOR[tile_id]

    if tile_id != AIR and TILE_DEFS[tile_id][1]:  # solid
        return SOLID_LIGHT_FACTOR

    return 1.0


# ---------------------------------------------------------------
# Rendering transparency
#
# This is visual opacity, separate from light blocking.
# 255 = fully opaque, 0 = invisible.
# ---------------------------------------------------------------

TILE_ALPHA = {
    WATER: 165,
    GLASS: 210,
    LEAVES: 255,
}

# Tiles drawn after the player/enemies.
FOREGROUND_TILES = {
    WATER,
}