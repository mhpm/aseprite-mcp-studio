"""Built-in bitmap fonts, so small text (damage numbers, labels, UI) is crisp at 1x without
depending on whatever TrueType fonts the machine happens to have.

* "tiny"  3x5: digits, A-Z (lower case maps to upper) and common punctuation.
* "bold"  5x7 with 2 px strokes: digits and + - ! %, made for damage and score numbers.

Text renders to an RGBA image with optional outline, drop shadow and a two-tone fill (top
colour over bottom colour), the usual look of a game damage number.
"""

from __future__ import annotations

import numpy as np

from .document import StudioError

_TINY_SRC = {
    "0": "111 101 101 101 111",
    "1": "010 110 010 010 111",
    "2": "111 001 111 100 111",
    "3": "111 001 011 001 111",
    "4": "101 101 111 001 001",
    "5": "111 100 111 001 111",
    "6": "111 100 111 101 111",
    "7": "111 001 001 010 010",
    "8": "111 101 111 101 111",
    "9": "111 101 111 001 111",
    "A": "010 101 111 101 101",
    "B": "110 101 110 101 110",
    "C": "011 100 100 100 011",
    "D": "110 101 101 101 110",
    "E": "111 100 110 100 111",
    "F": "111 100 110 100 100",
    "G": "011 100 101 101 011",
    "H": "101 101 111 101 101",
    "I": "111 010 010 010 111",
    "J": "001 001 001 101 010",
    "K": "101 101 110 101 101",
    "L": "100 100 100 100 111",
    "M": "101 111 111 101 101",
    "N": "110 101 101 101 101",
    "O": "010 101 101 101 010",
    "P": "110 101 110 100 100",
    "Q": "010 101 101 110 011",
    "R": "110 101 110 101 101",
    "S": "011 100 010 001 110",
    "T": "111 010 010 010 010",
    "U": "101 101 101 101 111",
    "V": "101 101 101 101 010",
    "W": "101 101 111 111 101",
    "X": "101 101 010 101 101",
    "Y": "101 101 010 010 010",
    "Z": "111 001 010 100 111",
    " ": "000 000 000 000 000",
    "-": "000 000 111 000 000",
    "+": "000 010 111 010 000",
    "!": "010 010 010 000 010",
    "?": "110 001 010 000 010",
    ".": "000 000 000 000 010",
    ",": "000 000 000 010 100",
    ":": "000 010 000 010 000",
    "/": "001 001 010 100 100",
    "%": "101 001 010 100 101",
    "(": "010 100 100 100 010",
    ")": "010 001 001 001 010",
    "=": "000 111 000 111 000",
    "'": "010 010 000 000 000",
    "*": "000 101 010 101 000",
    "#": "101 111 101 111 101",
    "<": "001 010 100 010 001",
    ">": "100 010 001 010 100",
    "_": "000 000 000 000 111",
}

_BOLD_SRC = {
    "0": [".###.", "##.##", "##.##", "##.##", "##.##", "##.##", ".###."],
    "1": ["..##.", ".###.", "..##.", "..##.", "..##.", "..##.", ".####"],
    "2": ["####.", "...##", "...##", ".###.", "##...", "##...", "#####"],
    "3": ["####.", "...##", "...##", ".###.", "...##", "...##", "####."],
    "4": ["##.##", "##.##", "##.##", "#####", "...##", "...##", "...##"],
    "5": ["#####", "##...", "##...", "####.", "...##", "...##", "####."],
    "6": [".###.", "##...", "##...", "####.", "##.##", "##.##", ".###."],
    "7": ["#####", "...##", "...##", "..##.", "..##.", ".##..", ".##.."],
    "8": [".###.", "##.##", "##.##", ".###.", "##.##", "##.##", ".###."],
    "9": [".###.", "##.##", "##.##", ".####", "...##", "...##", ".###."],
    "-": ["....", "....", "####", "####", "....", "....", "...."],
    "+": ["......", "..##..", "######", "######", "..##..", "......", "......"],
    "!": ["##", "##", "##", "##", "##", "..", "##"],
    "%": ["##..#", "##.##", "..##.", ".##..", "##...", "#..##", "...##"],
    " ": ["...", "...", "...", "...", "...", "...", "..."],
}


def _tiny(ch: str) -> np.ndarray:
    rows = _TINY_SRC.get(ch.upper())
    if rows is None:
        raise StudioError(f"the tiny font has no glyph for {ch!r}")
    return np.array([[c == "1" for c in r] for r in rows.split()], bool)


def _bold(ch: str) -> np.ndarray:
    rows = _BOLD_SRC.get(ch)
    if rows is None:
        raise StudioError(f"the bold font has digits and + - ! % only, not {ch!r}; use font='tiny'")
    return np.array([[c == "#" for c in r] for r in rows], bool)


FONTS = {"tiny": (_tiny, 5), "bold": (_bold, 7)}


def text_mask(text: str, font: str = "tiny", spacing: int = 1, line_gap: int = 1) -> np.ndarray:
    if font not in FONTS:
        raise StudioError(f"unknown font {font!r}; built-in fonts are {list(FONTS)}")
    glyph, height = FONTS[font]
    lines = []
    for line in text.split("\n"):
        gl = [glyph(c) for c in line] or [np.zeros((height, 1), bool)]
        w = sum(g.shape[1] for g in gl) + spacing * (len(gl) - 1)
        m = np.zeros((height, w), bool)
        x = 0
        for g in gl:
            m[:, x : x + g.shape[1]] = g
            x += g.shape[1] + spacing
        lines.append(m)
    W = max(m.shape[1] for m in lines)
    H = len(lines) * height + (len(lines) - 1) * line_gap
    out = np.zeros((H, W), bool)
    for i, m in enumerate(lines):
        y = i * (height + line_gap)
        out[y : y + height, : m.shape[1]] = m
    return out


def render_text(
    text: str,
    color,
    font: str = "tiny",
    bottom_color=None,
    outline=None,
    outline_diagonal: bool = True,
    shadow=None,
    scale: int = 1,
    spacing: int = 1,
) -> np.ndarray:
    """RGBA image of `text`. `bottom_color` fills the lower half of each glyph (two-tone);
    `outline` adds a 1 px keyline; `shadow` a 1 px drop shadow down-right."""
    m = text_mask(text, font, spacing)
    if scale > 1:
        m = m.repeat(scale, 0).repeat(scale, 1)
    pad = (1 if outline is not None else 0) + (1 if shadow is not None else 0)
    H, W = m.shape
    img = np.zeros((H + 2 * pad, W + 2 * pad, 4), np.uint8)
    o = pad - (1 if shadow is not None else 0)
    if shadow is not None:
        sm = np.zeros(img.shape[:2], bool)
        sm[o + 1 : o + 1 + H, o + 1 : o + 1 + W] = m
        if outline is not None:
            sm = _dilate(sm, outline_diagonal)
        img[sm] = shadow
    if outline is not None:
        om = np.zeros(img.shape[:2], bool)
        om[o : o + H, o : o + W] = m
        img[_dilate(om, outline_diagonal)] = outline
    fm = np.zeros(img.shape[:2], bool)
    fm[o : o + H, o : o + W] = m
    img[fm] = color
    if bottom_color is not None:
        glyph_h = FONTS[font][1] * scale
        rows = np.arange(img.shape[0])[:, None] - o
        lower = ((rows % (glyph_h + scale)) >= (glyph_h + 1) // 2) & fm
        img[lower] = bottom_color
    return img


def _dilate(m: np.ndarray, diagonal: bool) -> np.ndarray:
    p = np.pad(m, 1)
    out = m | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    if diagonal:
        out |= p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
    return out
