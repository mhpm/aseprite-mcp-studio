"""Geometric transforms that respect pixel art.

Rotation uses RotSprite (Xenowhirl's method): upscale 8x with three Scale2x passes, which
reconstructs edges instead of blowing pixels up into blocks, rotate that with nearest
neighbour, then sample back down. Unlike a plain nearest-neighbour rotation it keeps lines
continuous and does not chew holes in 1 px outlines, and unlike bilinear it never invents
colours, so the palette survives.
"""

from __future__ import annotations

import math

import numpy as np

from .raster import pack


def scale2x(img: np.ndarray) -> np.ndarray:
    """EPX / Scale2x on an RGBA image (exact colour equality)."""
    p = pack(img)
    h, w = p.shape
    pad = np.pad(p, 1, mode="edge")
    a, b, c, d = pad[:-2, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:], pad[2:, 1:-1]  # up, left, right, down
    e = p
    cond = (a != d) & (b != c)
    e0 = np.where(cond & (b == a), a, e)
    e1 = np.where(cond & (a == c), c, e)
    e2 = np.where(cond & (b == d), b, e)
    e3 = np.where(cond & (d == c), d, e)
    out = np.empty((h * 2, w * 2), np.uint32)
    out[0::2, 0::2], out[0::2, 1::2], out[1::2, 0::2], out[1::2, 1::2] = e0, e1, e2, e3
    return out.view(np.uint8).reshape(h * 2, w * 2, 4)


def rotate(
    img: np.ndarray, degrees: float, pivot: tuple[float, float] | None = None, method: str = "rotsprite"
) -> tuple[np.ndarray, int, int]:
    """Rotate clockwise (screen coordinates) about `pivot` (image coords, default centre).

    Returns (image, dx, dy): the new image and where its top-left lands relative to the
    old top-left, so the pivot stays put on the canvas."""
    h, w = img.shape[:2]
    if pivot is None:
        pivot = ((w - 1) / 2, (h - 1) / 2)
    deg = degrees % 360
    if abs(deg) < 1e-9:
        return img.copy(), 0, 0
    if abs(deg - 90) < 1e-9 or abs(deg - 180) < 1e-9 or abs(deg - 270) < 1e-9:
        k = int(round(deg / 90))  # exact quarter turns: no resampling at all
        out = np.rot90(img, -k).copy()
        px, py = pivot
        if k == 1:
            npx, npy = h - 1 - py, px
        elif k == 2:
            npx, npy = w - 1 - px, h - 1 - py
        else:
            npx, npy = py, w - 1 - px
        return out, int(round(px - npx)), int(round(py - npy))
    s = 8 if method == "rotsprite" else 1
    src = img
    if s == 8:
        for _ in range(3):
            src = scale2x(src)
    th = math.radians(deg)
    cos, sin = math.cos(th), math.sin(th)
    px, py = pivot
    # bounding box of the rotated image in original pixel units
    corners = np.array([[-0.5, -0.5], [w - 0.5, -0.5], [-0.5, h - 0.5], [w - 0.5, h - 0.5]]) - (px, py)
    rx = corners[:, 0] * cos - corners[:, 1] * sin
    ry = corners[:, 0] * sin + corners[:, 1] * cos
    x0, x1 = math.floor(rx.min() + px + 0.5), math.ceil(rx.max() + px - 0.5)
    y0, y1 = math.floor(ry.min() + py + 0.5), math.ceil(ry.max() + py - 0.5)
    ow, oh = x1 - x0 + 1, y1 - y0 + 1
    yy, xx = np.mgrid[0:oh, 0:ow]
    # destination pixel centres -> source coordinates (inverse rotation)
    dx = xx + x0 - px
    dy = yy + y0 - py
    sx = dx * cos + dy * sin + px
    sy = -dx * sin + dy * cos + py
    # sample the 8x image at the matching sub-pixel (centre of the 8x8 block)
    ix = np.floor((sx + 0.5) * s).astype(np.int64)
    iy = np.floor((sy + 0.5) * s).astype(np.int64)
    H, W = src.shape[:2]
    ok = (ix >= 0) & (iy >= 0) & (ix < W) & (iy < H)
    out = np.zeros((oh, ow, 4), np.uint8)
    out[ok] = src[iy[ok], ix[ok]]
    return out, x0, y0


def scale_nearest(img: np.ndarray, sx: float, sy: float | None = None) -> np.ndarray:
    sy = sx if sy is None else sy
    h, w = img.shape[:2]
    nh, nw = max(1, round(h * sy)), max(1, round(w * sx))
    ys = np.minimum((np.arange(nh) + 0.5) * h / nh, h - 1).astype(int)
    xs = np.minimum((np.arange(nw) + 0.5) * w / nw, w - 1).astype(int)
    return img[ys][:, xs].copy()


def squash(img: np.ndarray, amount: float, anchor: str = "bottom") -> tuple[np.ndarray, int, int]:
    """Volume-preserving squash (amount > 0) or stretch (< 0) keeping the anchor edge fixed.
    Returns (image, dx, dy) offsets of the new top-left."""
    h, w = img.shape[:2]
    vert = anchor in ("bottom", "top")
    k = 1 - amount
    if vert:
        nh, nw = max(1, round(h * k)), max(1, round(w / k))
    else:
        nw, nh = max(1, round(w * k)), max(1, round(h / k))
    out = scale_nearest(img, nw / w, nh / h)
    dx = -(nw - w) // 2 if vert else (0 if anchor == "left" else w - nw)
    dy = (h - nh if anchor == "bottom" else 0) if vert else -(nh - h) // 2
    return out, dx, dy
