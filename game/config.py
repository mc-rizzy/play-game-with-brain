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

# Sub-tile simulation -------------------------------------------
# Water and lighting run at 2x2 sub-cells per block (each 1/4 block in
# area): fine enough for smooth-looking flow and shadows, still hard
# pixel edges with no blending.

SUBCELLS = 2

# Water quantum: drops per sub-cell (tile holds 4x255 = 1020). Fine
# enough that pressure leveling rests flat to sub-pixel precision
# instead of stairstepping, coarse enough to stay fast.
DROPS_PER_SUB = 255

# Pressure flow: wet neighbours differing by at least this many drops
# exchange up to FLOW_MAX drops per tick (never more than half the
# difference, so levels can't overshoot and slosh).
FLOW_DIFF_MIN = 2
FLOW_MAX = 128
# Longer strides trim gentle slopes adjacent pairs can't see.
FLOW_STRIDES = (1, 2, 4, 8)
FLOW_TRIM_MAX = 8

# Spreading: a puddle of at least POUR_MIN drops pours POUR_AMOUNT
# drops per tick into dry ground (thinner films hold by tension).
POUR_MIN = 128
POUR_AMOUNT = 64

# Drip: blobs up to DRIP_MAX drops tip off cliff edges whole.
DRIP_MAX = 64

# Lighting -------------------------------------------------------
# Retro pixel lighting: analytic sun columns (exact crisp shadows)
# plus a continuously-relaxing torch flood, rendered in hard
# LIGHT_LEVELS bands per half-tile cell.

LIGHT_MAX = 255
LIGHT_LEVELS = 64

# Torch flood loss per sub-cell step (~10 per tile, as before).
TORCH_LOSS = 5

# Continuous-physics budgets (propagation steps). The frame budget
# keeps edits flowing into light sweeps; the edit top-up runs
# synchronously so breaks/places read instantly.
RELAX_FRAME_BUDGET = 1500
RELAX_EDIT_TOPUP = 2500

# Reset radius (tiles) around darkening edits: covers torch reach.
RESET_R_TORCH = 26

# Water physics tick: unconditional, always running.
WATER_TICK_DT = 0.08

# Applied whenever light spreads one tile, before tile factors.
LIGHT_SPREAD_LOSS = 9

# Diagonal spread costs extra (diagonal ~= 1.41 tiles away).
LIGHT_DIAGONAL_LOSS = 5
LIGHT_DIAGONAL_FACTOR = 0.72

# Kept for API compatibility. Display now snaps to the computed target
# (retro hard transitions) instead of lerping every tile each frame.
LIGHT_DISPLAY_LERP = 7.0

# Default multiplier when light enters an ordinary solid block.
# 0.45 means the block keeps about 45% of the incoming light,
# before LIGHT_SPREAD_LOSS is also applied.
#
# Individual blocks can override this in tiles.py.
SOLID_LIGHT_FACTOR = 0.45