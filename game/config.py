TILE = 80

WORLD_W, WORLD_H = 260, 95
DAY_LENGTH = 300.0

# Physics scales with tile size.
GRAVITY = TILE * 30.0          # 2400 at TILE=80
REACH = TILE * 5.2

PLAYER_SPEED = TILE * 2.9
PLAYER_SWIM_SPEED = TILE * 1.8
PLAYER_JUMP = -TILE * 10.5

WINDOWED_SIZE = (1280, 720)
MAX_PARTICLES = 280

# Lighting -------------------------------------------------------

LIGHT_MAX = 255

# Applied whenever light spreads one tile, before tile factors.
LIGHT_SPREAD_LOSS = 9

# Default multiplier when light enters an ordinary solid block.
# 0.45 means the block keeps about 45% of the incoming light,
# before LIGHT_SPREAD_LOSS is also applied.
#
# Individual blocks can override this in tiles.py.
SOLID_LIGHT_FACTOR = 0.45