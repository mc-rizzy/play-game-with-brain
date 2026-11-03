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

import os as _os
ASSETS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "assets")

# GUI scaling: UI is authored at 1280x720 and scaled up on larger screens.
UI_BASE_W, UI_BASE_H = 1280, 720
UI_MIN_SCALE = 1.0
UI_MAX_SCALE = 2.25

# Lighting -------------------------------------------------------

LIGHT_MAX = 255

# Applied whenever light spreads one tile, before tile factors.
LIGHT_SPREAD_LOSS = 9

# Diagonal spread costs extra (diagonal ~= 1.41 tiles away).
LIGHT_DIAGONAL_LOSS = 5
LIGHT_DIAGONAL_FACTOR = 0.72

# Per-second lerp rate of the displayed light towards the computed target.
# Higher = snappier, lower = smoother day/night and torch transitions.
LIGHT_DISPLAY_LERP = 7.0

# Default multiplier when light enters an ordinary solid block.
# 0.45 means the block keeps about 45% of the incoming light,
# before LIGHT_SPREAD_LOSS is also applied.
#
# Individual blocks can override this in tiles.py.
SOLID_LIGHT_FACTOR = 0.45