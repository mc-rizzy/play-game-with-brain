"""Generate pixel-art PNGs for characters/entities (stdlib only, no deps).

Run once from the workspace root::

    python game/generate_assets.py

or from inside ``game/``::

    python generate_assets.py

Output: ``game/assets/*.png`` — small pixel-art images loaded by
``sprites.py`` and scaled up with nearest-neighbour at runtime.
"""

import os
import struct
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

PAL = {
    ".": None,
    "O": (30, 30, 36, 255),      # outline
    "H": (106, 68, 40, 255),     # hair
    "h": (150, 105, 65, 255),    # hair highlight
    "S": (243, 192, 145, 255),   # skin
    "s": (214, 160, 120, 255),   # skin shadow
    "E": (30, 30, 36, 255),      # eye dark
    "W": (255, 255, 255, 255),   # eye white / shine
    "T": (92, 166, 116, 255),    # shirt
    "t": (70, 132, 90, 255),     # shirt dark
    "P": (70, 92, 148, 255),     # pants
    "p": (54, 72, 116, 255),     # pants dark
    "B": (88, 64, 44, 255),      # boots
    "G": (92, 210, 104, 255),    # slime body
    "g": (62, 160, 78, 255),     # slime dark
    "L": (170, 240, 180, 255),   # slime highlight
    "e": (24, 52, 30, 255),      # slime eye
    "C": (100, 112, 140, 255),   # shade cloak
    "c": (70, 80, 105, 255),     # shade cloak dark
    "D": (52, 62, 82, 255),      # shade lower
    "R": (255, 66, 66, 255),     # shade eyes
    "r": (180, 40, 40, 255),     # shade eye dark
    "V": (226, 231, 240, 255),   # sword blade
    "v": (170, 180, 196, 255),   # sword blade dark
    "N": (148, 98, 54, 255),     # sword hilt
    "n": (110, 70, 38, 255),     # sword hilt dark
    "F": (235, 70, 90, 255),     # heart full
    "f": (255, 150, 160, 255),   # heart highlight
    "M": (68, 54, 64, 255),      # heart empty
    "m": (48, 38, 46, 255),      # heart empty dark
}


def _chunk(ctype, data):
    out = struct.pack(">I", len(data)) + ctype + data
    out += struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
    return out


def write_png(path, rows):
    """Write RGBA rows (list of list of (r,g,b,a) or None) as PNG."""
    h = len(rows)
    w = len(rows[0]) if h else 0
    raw = bytearray()
    for row in rows:
        raw.append(0)  # filter: none
        for px in row:
            if px is None:
                raw.extend((0, 0, 0, 0))
            else:
                raw.extend(px)
    data = (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + _chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def compile_map(name, lines):
    w = max(len(r) for r in lines)
    rows = []
    for line in lines:
        line = line.ljust(w, ".")[:w]
        row = []
        for ch in line:
            if ch not in PAL:
                raise ValueError(f"{name}: unknown glyph {ch!r}")
            row.append(PAL[ch])
        rows.append(row)
    return rows


# ---------------------------------------------------------------- player
# 14 wide x 30 tall, facing right (runtime flips for left).

PLAYER_TOP = [
    "....OOOOOO....",
    "..OOHHHHHHOO..",
    "..OHhhhhhhHO..",
    "..OHHHHHHHHO..",
    "..OSSSSSSSSO..",
    "..OSSSSSSSESO.",
    "..OSSSSSSSSO..",
    "..OSssSSSSSO..",
    "...OSSSSOO....",
    "..OOTTTTTTTO..",
    ".OOTTTTTTTTTO.",
    ".OTtTTTTTTtTO.",
    ".OTtTTTTTTtTO.",
    ".OStTTTTTTtSO.",
    ".OStTTTTTTtSO.",
    ".OStTttttTtSO.",
    "..OTTTTTTTTO..",
]

PLAYER_IDLE_LEGS = [
    "...OPPPPPPO...",
    "...OPpPPpPO...",
    "...OPpPPpPO...",
    "...OPpPPpPO...",
    "...OPpPPpPO...",
    "...OPpPPpPO...",
    "...OPpPPpPO...",
    "...OPpPPpPO...",
    "..OBBP..PPBB..",
    "..OBBO..OBBO..",
    "..OOOO..OOOO..",
    "...OO....OO...",
    ".............",
]

PLAYER_WALK1_LEGS = [  # left leg forward
    "...OPPPPPPO...",
    "...OPpPPpPO...",
    "..OOPpPPpPO...",
    "..OPOpPPpPO...",
    ".OPO.PPpPO....",
    ".OPO.PPpPO....",
    ".OBO..PpPO....",
    ".OBO..PpPOO...",
    ".OBB...PpPBO..",
    ".OOO...OBBBO..",
    "..O.....OOOO..",
    ".............",
    ".............",
]

PLAYER_WALK2_LEGS = [  # right leg forward
    "...OPPPPPPO...",
    "...OPpPPpPO...",
    "...OPpPPpPOO..",
    "...OPpPPpOPO..",
    "....OPpPP.OPO.",
    "....OPpPP.OPO.",
    "....OPpP..OBO.",
    "...OOPpP..OBO.",
    "..OBPpP...BBO.",
    "..OBBBO...OOO.",
    "..OOOO.....O..",
    ".............",
    ".............",
]

# ---------------------------------------------------------------- slime
# 16 wide x 14 tall.

SLIME_0 = [
    "................",
    ".....OOOOOO.....",
    "...OOGGGGGGOO...",
    "..OGGLLGGGGEeO..",
    ".OGGLLLGGGGGeOO.",
    ".OGLLGGGGGeEeO..",
    "OGGLLGGGGGGeEeO.",
    "OGLLGGGGGGGeEeO.",
    "OGgGGGGGGGGGeEO.",
    "OGgGGGGGGGGGGEO.",
    ".OggGGGGGGGGGO..",
    ".OOOOOOOOOOOO...",
    "................",
    "................",
]

SLIME_1 = [  # squashed
    "................",
    "................",
    "................",
    "...OOOOOOOOOO...",
    ".OOGGGGGGGGGGOO.",
    "OGGLLLGGGGGeEeO.",
    "OGLLLGGGGGeEeEO.",
    "OGgGGGGGGGGeEEO.",
    "OGggGGGGGGGGGEO.",
    ".OggGGGGGGGGGO..",
    "..OOOOOOOOOOO...",
    "................",
    "................",
    "................",
]

# ---------------------------------------------------------------- shade
# 14 wide x 30 tall.

SHADE_0 = [
    "....OOOOOO....",
    "..OOCCCCCCOO..",
    ".OCCccccccCCO.",
    ".OCccccccccCO.",
    ".OCcRRccRRcCO.",
    ".OCcRRccRRcCO.",
    ".OCccrccrccCO.",
    ".OCccccccccCO.",
    ".OCccccccccCO.",
    ".ODDccccccDDO.",
    ".ODDccccccDDO.",
    "..ODDccccDDO..",
    "..ODDccccDDO..",
    "..ODDccccDDO..",
    "..ODDccccDDO..",
    "..ODDccccDDO..",
    "..ODDccccDDO..",
    "...DDccccDD...",
    "...DDccccDD...",
    "...DDccccDD...",
    "...DDc..cDD...",
    "...DDc..cDD...",
    "...OO....OO...",
    ".............",
    ".............",
    ".............",
    ".............",
    ".............",
    ".............",
    ".............",
]

SHADE_1 = [  # cloak sway + blink
    "....OOOOOO....",
    "..OOCCCCCCOO..",
    ".OCCccccccCCO.",
    ".OCccccccccCO.",
    ".OCcRrccrRcCO.",
    ".OCccccccccCO.",
    ".OCccccccccCO.",
    ".OCccccccccCO.",
    "..CccccccccC..",
    "..DDccccccDD..",
    "..DDccccccDD..",
    "...DDccccDD...",
    "...DDccccDD...",
    "...DDccccDD...",
    "...DDccccDD...",
    "...DDccccDDD..",
    "...DDccccDDD..",
    "....DccccDD...",
    "....DccccDD...",
    "....Dc..cDD...",
    "....Dc...DD...",
    "....OO...OO...",
    ".............",
    ".............",
    ".............",
    ".............",
    ".............",
    ".............",
    ".............",
    ".............",
]

# ---------------------------------------------------------------- sword
# 14 x 14 diagonal sword for hotbar + swing.

SWORD = [
    "..........OO..",
    ".........OVVO.",
    "........OVVVO.",
    ".......OVVvVO.",
    "......OVVvVO..",
    ".....OVVvVO...",
    "....OVVvVO....",
    "...OVVvVO.....",
    "..OVVvVO......",
    ".OOVvVO.......",
    ".NNNVO........",
    "..NNN.........",
    "..NnN.........",
    "...n..........",
]

# ---------------------------------------------------------------- hearts
# 10 x 9 GUI hearts.

HEART_FULL = [
    ".OO....OO.",
    "OFfO..OFfO",
    "OFfFOOOFFO",
    "OFFFFFfFFO",
    "OFFFFFfFFO",
    ".OFFFFFFO.",
    "..OFFFFO..",
    "...OFFO...",
    "....OO....",
]

HEART_EMPTY = [
    ".OO....OO.",
    "OMmO..OMmO",
    "OMmMOOMMMO",
    "OMMMMMMMMO",
    "OMMMMMMMMO",
    ".OMMMMMMO.",
    "..OMMMMO..",
    "...OMMO...",
    "....OO....",
]

MAPS = {
    "player_idle": PLAYER_TOP + PLAYER_IDLE_LEGS,
    "player_walk1": PLAYER_TOP + PLAYER_WALK1_LEGS,
    "player_walk2": PLAYER_TOP + PLAYER_WALK2_LEGS,
    "slime_0": SLIME_0,
    "slime_1": SLIME_1,
    "shade_0": SHADE_0,
    "shade_1": SHADE_1,
    "sword": SWORD,
    "heart_full": HEART_FULL,
    "heart_empty": HEART_EMPTY,
}


def main():
    os.makedirs(ASSETS, exist_ok=True)
    for name, lines in MAPS.items():
        rows = compile_map(name, lines)
        path = write_png(os.path.join(ASSETS, f"{name}.png"), rows)
        print(f"wrote {path} ({len(rows[0])}x{len(rows)})")


if __name__ == "__main__":
    main()
