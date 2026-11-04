"""Generate pixel-art PNGs for characters/entities (stdlib only, no deps).

Run once from the workspace root::

    python game/generate_assets.py

Output: ``game/assets/*.png`` — 16x16 pixel-art images (tall characters
are 16x32, two stacked 16x16 cells) loaded by ``sprites.py`` and scaled
up with nearest-neighbour at runtime.
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
    "K": (60, 42, 28, 255),      # belt leather
    "k": (240, 200, 90, 255),    # belt buckle
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
    "a": (124, 112, 168, 255),   # bat body
    "b": (72, 62, 116, 255),     # bat wing
    "d": (228, 228, 238, 255),   # bone
    "o": (158, 160, 182, 255),   # bone shadow
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
# 16 wide x 24 tall (1 x 1.5 blocks), facing right. Boot soles sit on
# the last row so the character touches the ground.

PLAYER_TOP = [
    ".....OOOOOO.....",
    "...OOHHHHHHOO...",
    "..OHHhhhhhhHHO..",
    "..OHHHHHHHHHHO..",
    "..OHSSSSSSSSO...",
    "..OHSSSSSSESO...",
    "..OHSSSSSSSSO...",
    "..OHSSSssSSSO...",
    "...OSSSSSSO.....",
    "..OOTTTTTTTTO...",
    "..OTtTTTTTTtTO..",
    "..OStTTTTTTtSO..",
    "..OKKKKKKKKKKO..",
    "..OKKKKkkKKKKO..",
    "...OPPPPPPPO....",
]

PLAYER_IDLE_LEGS = [
    "...OPPPPPPPO....",
    "...OPpPPPpPO....",
    "...OPpPPPpPO....",
    "...OPpPPPpPO....",
    "...OPpP..PpPO...",
    "...OPpP..PpPO...",
    "...OBBP..PPBO...",
    "...OBBO...OBBO..",
    "...OOOO...OOOO..",
]

PLAYER_WALK1_LEGS = [  # stride left
    "...OPPPPPPPO....",
    "...OPpPPPpPO....",
    "...OPpPPPpPO....",
    "..OOPpPPPpPO....",
    "..OPpP..PpPO....",
    ".OPpP....PpPO...",
    ".OBBP....PPBO...",
    ".OBO......OBO...",
    ".OOO......OOO...",
]

PLAYER_WALK2_LEGS = [  # stride right
    "...OPPPPPPPO....",
    "...OPpPPPpPO....",
    "...OPpPPPpPO....",
    "....OPpPPPpPO...",
    "....OPpP..PpPO..",
    "...OPpP....PpPO.",
    "...OBBP....PPBO.",
    "...OBO......OBO.",
    "...OOO......OOO.",
]

# ---------------------------------------------------------------- slime
# 16 wide x 16 tall.

SLIME_0 = [  # base sits on the last row
    "................",
    "................",
    "................",
    "................",
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
]

SLIME_1 = [  # squashed, base sits on the last row
    "................",
    "................",
    "................",
    "................",
    "................",
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
]

# ---------------------------------------------------------------- shade
# 16 wide x 32 tall (two stacked 16x16 cells).

SHADE_0 = [  # feet sit on the last row
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    ".....OOOOOO.....",
    "...OOCCCCCCOO...",
    "..OCCccccccCCO..",
    "..OCccccccccCO..",
    "..OCcRRccRRcCO..",
    "..OCcRRccRRcCO..",
    "..OCccrccrccCO..",
    "..OCccccccccCO..",
    "..OCccccccccCO..",
    "..ODDccccccDDO..",
    "..ODDccccccDDO..",
    "...ODDccccDDO...",
    "...ODDccccDDO...",
    "...ODDccccDDO...",
    "...ODDccccDDO...",
    "...ODDccccDDO...",
    "...ODDccccDDO...",
    "....DDccccDD....",
    "....DDccccDD....",
    "....DDccccDD....",
    "....DDc..cDD....",
    "....DDc..cDD....",
    "....OO....OO....",
]

SHADE_1 = [  # cloak sway + blink, feet on the last row
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    ".....OOOOOO.....",
    "...OOCCCCCCOO...",
    "..OCCccccccCCO..",
    "..OCccccccccCO..",
    "..OCcRrccrRcCO..",
    "..OCccccccccCO..",
    "..OCccccccccCO..",
    "..OCccccccccCO..",
    "...CccccccccC...",
    "...DDccccccDD...",
    "...DDccccccDD...",
    "....DDccccDD....",
    "....DDccccDD....",
    "....DDccccDD....",
    "....DDccccDD....",
    "....DDccccDDD...",
    "....DDccccDDD...",
    ".....DccccDD....",
    ".....DccccDD....",
    ".....Dc..cDD....",
    ".....Dc...DD....",
    ".....OO...OO....",
]

# ---------------------------------------------------------------- sword
# 16 x 16 diagonal sword for hotbar + swing.

SWORD = [
    "................",
    "...........OO...",
    "..........OVVO..",
    ".........OVVVO..",
    "........OVVvVO..",
    ".......OVVvVO...",
    "......OVVvVO....",
    ".....OVVvVO.....",
    "....OVVvVO......",
    "...OVVvVO.......",
    "..OOVvVO........",
    "..NNNVO.........",
    "...NNN..........",
    "...NnN..........",
    "....n...........",
    "................",
]

# ---------------------------------------------------------------- hearts
# 16 x 16 GUI hearts (10x9 heart centred on transparent padding).

_HEART_FULL_CORE = [
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

_HEART_EMPTY_CORE = [
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

HEART_FULL = (
    ["................"] * 3
    + ["..." + row + "..." for row in _HEART_FULL_CORE]
    + ["................"] * 4
)

HEART_EMPTY = (
    ["................"] * 3
    + ["..." + row + "..." for row in _HEART_EMPTY_CORE]
    + ["................"] * 4
)

BAT_0 = (
    "OO............OO",
    "ObbO........ObbO",
    "ObbbO......ObbbO",
    ".ObbbO....ObbbO.",
    "..ObbbO..ObbbO..",
    "....ObbaaabbO...",
    ".....OaaaaO.....",
    ".....ORaaRO.....",
    ".....OaaaaO.....",
    "......OaaO......",
    "......O..O......",
    ".....OO..OO.....",
    "................",
    "................",
    "................",
    "................",
)

BAT_1 = (
    "................",
    "......O..O......",
    ".....OaaaaO.....",
    ".O...ORaaRO...O.",
    ".Ob..OaaaaO..bO.",
    ".Obb.OaaaaO.bbO.",
    "..ObbOaaaaObbO..",
    "..ObbOa..aObbO..",
    "...ObO....ObO...",
    "...ObO....ObO...",
    "....OO......OO..",
    "................",
    "................",
    "................",
    "................",
    "................",
)

SKELETON_0 = (
    "....OOOOOOOO....",
    "...OddddddddO...",
    "...OdEEddEEdo...",
    "...Oddddddddo...",
    "...Oddooodddo...",
    "....OOooooOO....",
    "....oooddooo....",
    "...ooooddoooo...",
    "....oooddooo....",
    ".....ooddoo.....",
    ".....ooddoo.....",
    "......OddO......",
    ".....OooooO.....",
    ".....OooooO.....",
    ".....Od..dO.....",
    ".....Od..dO.....",
    ".....Od..dO.....",
    ".....Od..dO.....",
    ".....Od..dO.....",
    ".....Od..dO.....",
    ".....Od..dO.....",
    ".....Od.dOOO....",
    ".....Od.........",
    ".....OOO........",
)

MAPS = {
    "player_idle": PLAYER_TOP + PLAYER_IDLE_LEGS,
    "player_walk1": PLAYER_TOP + PLAYER_WALK1_LEGS,
    "player_walk2": PLAYER_TOP + PLAYER_WALK2_LEGS,
    "slime_0": SLIME_0,
    "slime_1": SLIME_1,
    "shade_0": SHADE_0,
    "shade_1": SHADE_1,
    "bat_0": BAT_0,
    "bat_1": BAT_1,
    "skeleton_0": SKELETON_0,
    "skeleton_1": (SKELETON_0[:14]
                   + tuple(row[::-1] for row in SKELETON_0[14:])),
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
