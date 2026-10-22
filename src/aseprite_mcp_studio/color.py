"""Colour parsing and perceptual colour maths (OKLab).

OKLab is used wherever colours are compared or mixed: nearest-palette mapping, palette
reduction, ramps. Euclidean distance in OKLab tracks perceived difference far better than
RGB distance, which over-separates greens and under-separates blues and dark tones.
"""

from __future__ import annotations

import numpy as np

from .document import RGBA, StudioError

NAMED = {
    "transparent": (0, 0, 0, 0),
    "black": (0, 0, 0, 255),
    "white": (255, 255, 255, 255),
    "red": (255, 0, 0, 255),
    "green": (0, 255, 0, 255),
    "blue": (0, 0, 255, 255),
    "yellow": (255, 255, 0, 255),
    "cyan": (0, 255, 255, 255),
    "magenta": (255, 0, 255, 255),
    "gray": (128, 128, 128, 255),
    "grey": (128, 128, 128, 255),
}


def parse(c: str | list | tuple) -> RGBA:
    """'#rgb', '#rrggbb', '#rrggbbaa', 'r,g,b[,a]', [r,g,b(,a)] or a name -> (r, g, b, a)."""
    if isinstance(c, (list, tuple)):
        vals = [int(v) for v in c]
    else:
        s = str(c).strip().lower()
        if s in NAMED:
            return NAMED[s]
        if s.startswith("#"):
            h = s[1:]
            if len(h) in (3, 4):
                h = "".join(ch * 2 for ch in h)
            if len(h) not in (6, 8) or any(ch not in "0123456789abcdef" for ch in h):
                raise StudioError(f"bad colour {c!r}; use #rrggbb or #rrggbbaa")
            vals = [int(h[i : i + 2], 16) for i in range(0, len(h), 2)]
        elif "," in s:
            vals = [int(v) for v in s.split(",")]
        else:
            raise StudioError(f"bad colour {c!r}; use #rrggbb, #rrggbbaa, 'r,g,b' or a name")
    if len(vals) == 3:
        vals.append(255)
    if len(vals) != 4 or any(not 0 <= v <= 255 for v in vals):
        raise StudioError(f"bad colour {c!r}")
    return tuple(vals)  # type: ignore[return-value]


def to_hex(c) -> str:
    c = [int(v) for v in c]
    return "#" + "".join(f"{v:02x}" for v in (c if len(c) == 4 and c[3] != 255 else c[:3]))


# ---------------------------------------------------------------------------- OKLab
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055) * 255.0


_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) sRGB 0-255 -> (..., 3) OKLab."""
    lms = _srgb_to_linear(np.asarray(rgb, float)) @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lms = np.asarray(lab, float) @ np.linalg.inv(_M2).T
    return _linear_to_srgb((lms**3) @ np.linalg.inv(_M1).T)


def nearest(colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Index into `palette` (k, 3 RGB) of the perceptually nearest entry for each of
    `colors` (n, 3 RGB). Chunked so large images stay within memory."""
    pl = rgb_to_oklab(palette)
    out = np.empty(len(colors), np.int64)
    for i in range(0, len(colors), 65536):
        cl = rgb_to_oklab(colors[i : i + 65536])
        d = ((cl[:, None, :] - pl[None, :, :]) ** 2).sum(-1)
        out[i : i + 65536] = d.argmin(1)
    return out


def delta_e(a, b) -> float:
    """Perceptual distance between two RGB colours (OKLab Euclidean, x100 so ~1 is a JND)."""
    la, lb = rgb_to_oklab(np.array(a[:3], float)), rgb_to_oklab(np.array(b[:3], float))
    return float(np.linalg.norm(la - lb) * 100)


def luminance(rgb) -> float:
    lin = _srgb_to_linear(np.array(rgb[:3], float))
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def ramp(base: str, steps: int = 5, hue_shift: float = 20.0, contrast: float = 0.55) -> list[str]:
    """A shading ramp around `base`, darkest first, built in OKLCh: lightness spreads evenly,
    hue rotates toward warm in the lights and cool in the shadows (`hue_shift` degrees total),
    and chroma tapers at both ends. This is the classic hand-made pixel-art ramp rather than
    a straight RGB blend toward black and white, which goes muddy."""
    r, g, b, _ = parse(base)
    L, A, B = rgb_to_oklab(np.array([r, g, b], float))
    C, h = float(np.hypot(A, B)), float(np.degrees(np.arctan2(B, A)))
    out = []
    for i in range(steps):
        t = i / (steps - 1) - 0.5 if steps > 1 else 0.0  # -0.5 dark .. +0.5 light
        li = float(np.clip(L + t * contrast * 2 * min(L, 1 - L) * 1.6, 0.04, 0.98))
        warm = 60.0  # hue of warm light in OKLab degrees (orange-yellow)
        dh = ((warm - h + 180) % 360) - 180
        hi = h + np.sign(dh) * min(abs(dh), hue_shift) * t * 2 * 0.5
        ci = C * (1 - 0.6 * abs(t) ** 1.5)
        lab = np.array([li, ci * np.cos(np.radians(hi)), ci * np.sin(np.radians(hi))])
        out.append(to_hex(np.clip(np.round(oklab_to_rgb(lab)), 0, 255).astype(int)))
    return out
