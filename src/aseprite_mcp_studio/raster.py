"""Pixel primitives: an editable surface over one cel, and the shapes drawn on it.

Every primitive produces integer coordinates and the Surface writes them. Lines and curves
are "pixel perfect": after rasterising, any pixel that only forms an L-shaped corner between
two diagonal neighbours is removed, which is what a pixel artist does by hand and what keeps
1 px outlines from looking doubled at the turns.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .document import Cel, Layer, Sprite, trim


class Surface:
    """A growable RGBA buffer for one (layer, frame), in canvas coordinates.

    It starts as the union of the canvas and the existing cel, and grows when something is
    drawn outside it, so drawing past the canvas edge keeps those pixels in the cel instead
    of cutting them off. `commit` trims it back to its content."""

    def __init__(self, spr: Sprite, layer: Layer, frame: int):
        self.spr, self.layer, self.frame = spr, layer, frame
        self.x0, self.y0, x1, y1 = 0, 0, spr.width, spr.height
        cel = layer.cels.get(frame)
        if cel is not None:
            cx, cy, cw, ch = cel.bounds
            self.x0, self.y0 = min(0, cx), min(0, cy)
            x1, y1 = max(x1, cx + cw), max(y1, cy + ch)
        self.buf = np.zeros((y1 - self.y0, x1 - self.x0, 4), np.uint8)
        if cel is not None:
            cx, cy, cw, ch = cel.bounds
            self.buf[cy - self.y0 : cy - self.y0 + ch, cx - self.x0 : cx - self.x0 + cw] = cel.image
        self.written = 0
        self.off_canvas = 0
        self.lock_alpha = False

    def ensure(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Grow so that [x0, x1) x [y0, y1) is inside the buffer."""
        h, w = self.buf.shape[:2]
        nx0, ny0 = min(self.x0, x0), min(self.y0, y0)
        nx1, ny1 = max(self.x0 + w, x1), max(self.y0 + h, y1)
        if (nx0, ny0, nx1, ny1) == (self.x0, self.y0, self.x0 + w, self.y0 + h):
            return
        nb = np.zeros((ny1 - ny0, nx1 - nx0, 4), np.uint8)
        nb[self.y0 - ny0 : self.y0 - ny0 + h, self.x0 - nx0 : self.x0 - nx0 + w] = self.buf
        self.buf, self.x0, self.y0 = nb, nx0, ny0

    def view(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        self.ensure(x, y, x + w, y + h)
        return self.buf[y - self.y0 : y - self.y0 + h, x - self.x0 : x - self.x0 + w]

    def canvas(self) -> np.ndarray:
        """The canvas-sized region (a view: edits write through)."""
        return self.view(0, 0, self.spr.width, self.spr.height)

    def plot(self, xs, ys, color, blend: bool = False) -> int:
        """Write one colour at many points. Returns how many pixels were set."""
        xs = np.asarray(xs, np.int64).ravel()
        ys = np.asarray(ys, np.int64).ravel()
        if xs.size == 0:
            return 0
        self.ensure(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        bx, by = xs - self.x0, ys - self.y0
        if self.lock_alpha:
            keep = self.buf[by, bx, 3] > 0
            bx, by, xs, ys = bx[keep], by[keep], xs[keep], ys[keep]
        col = np.array(color, np.uint8)
        if blend and col[3] < 255:
            dst = self.buf[by, bx].astype(np.float64)
            self.buf[by, bx] = blend_over(dst, col.astype(np.float64))
        else:
            self.buf[by, bx] = col
        self._count(xs, ys)
        return int(xs.size)

    def put_image(self, img: np.ndarray, x: int, y: int, blend: bool = True, skip_transparent: bool = True) -> int:
        """Stamp an RGBA image at (x, y)."""
        h, w = img.shape[:2]
        region = self.view(x, y, w, h)
        mask = img[..., 3] > 0 if skip_transparent else np.ones((h, w), bool)
        if self.lock_alpha:
            mask &= region[..., 3] > 0
        if blend:
            src = img.astype(np.float64)
            out = blend_over(region.astype(np.float64), src)
            region[mask] = out[mask]
        else:
            region[mask] = img[mask]
        ys, xs = np.nonzero(mask)
        self._count(xs + x, ys + y)
        return int(mask.sum())

    def _count(self, xs: np.ndarray, ys: np.ndarray) -> None:
        self.written += int(xs.size)
        inside = (xs >= 0) & (ys >= 0) & (xs < self.spr.width) & (ys < self.spr.height)
        self.off_canvas += int((~inside).sum())

    def commit(self) -> dict:
        t = trim(self.buf, self.x0, self.y0)
        old = self.layer.cels.get(self.frame)
        if t is None:
            self.layer.cels.pop(self.frame, None)
        else:
            img, x, y = t
            if old is not None:  # keep the Cel object so linked frames stay linked
                old.image, old.x, old.y = img, x, y
            else:
                self.layer.cels[self.frame] = Cel(img, x, y)
        res = {"pixels_written": self.written}
        if self.off_canvas:
            res["pixels_off_canvas"] = self.off_canvas
            res["note"] = "pixels past the canvas edge were kept in the cel, not clipped"
        return res


def blend_over(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Straight-alpha 'over' compositing on float arrays (..., 4) in 0-255."""
    sa = src[..., 3:4] / 255.0
    da = dst[..., 3:4] / 255.0
    oa = sa + da * (1 - sa)
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb = np.where(oa > 0, (src[..., :3] * sa + dst[..., :3] * da * (1 - sa)) / np.maximum(oa, 1e-9), 0)
    return np.concatenate([np.round(rgb), np.round(oa * 255)], -1).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------- shapes
def line_points(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Bresenham, all octants."""
    pts = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        pts.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return pts
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def pixel_perfect(pts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop the middle pixel of every L-corner (a, b, c) where a and c touch diagonally."""
    out: list[tuple[int, int]] = []
    for p in pts:
        if out and out[-1] == p:
            continue
        out.append(p)
        if len(out) >= 3:
            a, b, c = out[-3], out[-2], out[-1]
            if abs(a[0] - c[0]) == 1 and abs(a[1] - c[1]) == 1 and (b[0] == a[0] or b[1] == a[1]):
                out.pop(-2)
    return out


def polyline_points(points: list[tuple[int, int]], closed: bool = False, perfect: bool = True) -> list:
    pts: list[tuple[int, int]] = []
    seq = list(points) + ([points[0]] if closed and len(points) > 2 else [])
    for (ax, ay), (bx, by) in zip(seq, seq[1:], strict=False):
        seg = line_points(ax, ay, bx, by)
        pts.extend(seg if not pts else seg[1:])
    return pixel_perfect(pts) if perfect else pts


def bezier_points(ctrl: list[tuple[float, float]], perfect: bool = True) -> list[tuple[int, int]]:
    """Quadratic or cubic Bezier (3 or 4 control points), sampled densely then stitched with
    Bresenham so there are no gaps, then cleaned to a pixel-perfect stroke."""
    n = len(ctrl)
    c = np.array(ctrl, float)
    length = sum(np.hypot(*(c[i + 1] - c[i])) for i in range(n - 1))
    ts = np.linspace(0, 1, max(8, int(length * 3)))
    if n == 3:
        pts = ((1 - ts) ** 2)[:, None] * c[0] + (2 * (1 - ts) * ts)[:, None] * c[1] + (ts**2)[:, None] * c[2]
    elif n == 4:
        pts = (
            ((1 - ts) ** 3)[:, None] * c[0]
            + (3 * (1 - ts) ** 2 * ts)[:, None] * c[1]
            + (3 * (1 - ts) * ts**2)[:, None] * c[2]
            + (ts**3)[:, None] * c[3]
        )
    else:
        raise ValueError("a curve takes 3 (quadratic) or 4 (cubic) control points")
    ipts = [(int(round(x)), int(round(y))) for x, y in pts]
    return polyline_points(ipts, perfect=perfect)


def ellipse_mask(w: int, h: int, filled: bool = True) -> np.ndarray:
    """An ellipse filling a w x h box exactly (even sizes work, unlike centre+radius)."""
    yy, xx = np.mgrid[0:h, 0:w]
    rx, ry = w / 2, h / 2
    inside = ((xx + 0.5 - rx) / rx) ** 2 + ((yy + 0.5 - ry) / ry) ** 2 <= 1.0 + 1e-9
    if filled:
        return inside
    pad = np.pad(inside, 1)
    interior = pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return inside & ~interior


def flood_region(img: np.ndarray, x: int, y: int, tolerance: int = 0, diagonal: bool = False) -> np.ndarray:
    """Mask of the connected region of pixels matching img[y, x] (scanline fill)."""
    h, w = img.shape[:2]
    target = img[y, x].astype(np.int16)
    match = np.abs(img.astype(np.int16) - target).max(-1) <= tolerance
    if target[3] == 0:  # all transparent pixels match each other whatever their RGB
        match = img[..., 3] == 0 if tolerance == 0 else match | (img[..., 3] <= tolerance)
    out = np.zeros((h, w), bool)
    q = deque([(x, y)])
    while q:
        sx, sy = q.popleft()
        if out[sy, sx] or not match[sy, sx]:
            continue
        lx = sx
        while lx > 0 and match[sy, lx - 1] and not out[sy, lx - 1]:
            lx -= 1
        rx = sx
        while rx < w - 1 and match[sy, rx + 1] and not out[sy, rx + 1]:
            rx += 1
        out[sy, lx : rx + 1] = True
        a, b = (max(0, lx - 1), min(w - 1, rx + 1)) if diagonal else (lx, rx)
        for ny in (sy - 1, sy + 1):
            if 0 <= ny < h:
                row = match[ny, a : b + 1] & ~out[ny, a : b + 1]
                starts = np.flatnonzero(row & ~np.concatenate([[False], row[:-1]]))
                q.extend((a + int(s), ny) for s in starts)
    return out


def components(mask: np.ndarray, diagonal: bool = True) -> tuple[np.ndarray, int]:
    """Label connected components of a boolean mask. Returns (labels, count); 0 = background."""
    h, w = mask.shape
    labels = np.zeros((h, w), np.int32)
    n = 0
    nbrs = (
        [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        if diagonal
        else [(0, -1), (-1, 0), (1, 0), (0, 1)]
    )
    for y, x in zip(*np.nonzero(mask), strict=False):
        if labels[y, x]:
            continue
        n += 1
        labels[y, x] = n
        stack = [(y, x)]
        while stack:
            cy, cx = stack.pop()
            for dx, dy in nbrs:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = n
                    stack.append((ny, nx))
    return labels, n


def outline_mask(alpha: np.ndarray, diagonal: bool = True, inside: bool = False) -> np.ndarray:
    """Pixels just outside (or the border pixels just inside) an opaque shape."""
    o = alpha > 0
    p = np.pad(o, 1)
    sides = [p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]]
    diag = [p[:-2, :-2], p[:-2, 2:], p[2:, :-2], p[2:, 2:]] if diagonal else []
    if inside:
        empty_near = np.zeros_like(o)
        for s in sides + diag:
            empty_near |= ~s
        return o & empty_near
    near = np.zeros_like(o)
    for s in sides + diag:
        near |= s
    return near & ~o


def pack(img: np.ndarray) -> np.ndarray:
    """RGBA -> one uint32 per pixel, for fast colour equality tests."""
    a = np.ascontiguousarray(img, np.uint8)
    return a.view(np.uint32).reshape(a.shape[:2])
