"""Compositing, exports and the review views.

Compositing follows Aseprite: layers bottom to top, a group is flattened first and then
blended with its own opacity and mode, cel opacity multiplies layer opacity, hidden layers
(and everything inside a hidden group) are skipped.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .color import rgb_to_oklab
from .document import Layer, Sprite, StudioError
from .font import render_text


# ---------------------------------------------------------------------------- blending
def _blend_rgb(mode: str, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Separable blend modes on 0..1 RGB (b = backdrop, s = source)."""
    if mode == "multiply":
        return b * s
    if mode == "screen":
        return b + s - b * s
    if mode == "overlay":
        return np.where(b <= 0.5, 2 * b * s, 1 - 2 * (1 - b) * (1 - s))
    if mode == "darken":
        return np.minimum(b, s)
    if mode == "lighten":
        return np.maximum(b, s)
    if mode == "difference":
        return np.abs(b - s)
    if mode == "exclusion":
        return b + s - 2 * b * s
    if mode == "addition":
        return np.minimum(1, b + s)
    if mode == "subtract":
        return np.maximum(0, b - s)
    if mode == "hard_light":
        return np.where(s <= 0.5, 2 * b * s, 1 - 2 * (1 - b) * (1 - s))
    return s  # normal and the non-separable modes fall back to normal


def composite_onto(dst: np.ndarray, src: np.ndarray, opacity: float = 1.0, mode: str = "normal") -> None:
    """Blend src (h, w, 4 uint8) over dst (same shape, float 0..1 RGBA, straight) in place."""
    sa = (src[..., 3:4] / 255.0) * opacity
    if not sa.any():
        return
    s = src[..., :3] / 255.0
    b, ba = dst[..., :3], dst[..., 3:4]
    mixed = _blend_rgb(mode, b, s) if mode != "normal" else s
    s_eff = (1 - ba) * s + ba * mixed  # where the backdrop is empty the plain source shows
    oa = sa + ba * (1 - sa)
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb = np.where(oa > 0, (s_eff * sa + b * ba * (1 - sa)) / np.maximum(oa, 1e-12), 0)
    dst[..., :3] = rgb
    dst[..., 3:4] = oa


def _render_layers(spr: Sprite, frame: int, parent: Layer | None, layers_filter) -> np.ndarray:
    acc = np.zeros((spr.height, spr.width, 4), float)
    for la in spr.children(parent):
        if not la.visible or (layers_filter and not layers_filter(la)):
            continue
        if la.kind == "group":
            sub = _render_layers(spr, frame, la, layers_filter)
            composite_onto(acc, to_uint8(sub), la.opacity / 255, la.blend)
            continue
        c = la.cels.get(frame)
        if c is None:
            continue
        x, y, w, h = c.bounds
        x0, y0, x1, y1 = max(0, x), max(0, y), min(spr.width, x + w), min(spr.height, y + h)
        if x0 >= x1 or y0 >= y1:
            continue
        composite_onto(
            acc[y0:y1, x0:x1],
            c.image[y0 - y : y1 - y, x0 - x : x1 - x],
            (c.opacity / 255) * (la.opacity / 255),
            la.blend,
        )
    return acc


def to_uint8(acc: np.ndarray) -> np.ndarray:
    out = np.empty(acc.shape, np.uint8)
    out[..., :3] = np.round(acc[..., :3] * 255).clip(0, 255)
    out[..., 3] = np.round(acc[..., 3] * 255).clip(0, 255)
    out[out[..., 3] == 0] = 0
    return out


def render_frame(
    spr: Sprite,
    frame: int,
    layers: list[str] | None = None,
    exclude: list[str] | None = None,
    include_reference: bool = False,
) -> np.ndarray:
    """Flattened RGBA (uint8) of one 0-based frame."""
    want = {spr.layer(n).path for n in layers} if layers else None
    skip = {spr.layer(n).path for n in exclude} if exclude else set()

    def filt(la: Layer) -> bool:
        if la.reference and not include_reference:
            return False
        if la.path in skip:
            return False
        if want is None or la.kind == "group":
            return True
        return la.path in want or any(la.path.startswith(w + "/") for w in want)

    return to_uint8(_render_layers(spr, frame, None, filt))


def upscale(img: np.ndarray, s: int) -> np.ndarray:
    return img if s == 1 else img.repeat(s, 0).repeat(s, 1)


def on_background(img: np.ndarray, color) -> np.ndarray:
    acc = np.zeros(img.shape, float)
    acc[..., :3] = np.array(color[:3]) / 255
    acc[..., 3] = 1
    composite_onto(acc, img)
    return to_uint8(acc)


# ---------------------------------------------------------------------------- exports
def frames_range(spr: Sprite, frames: list[int] | None, tag: str | None) -> list[int]:
    if tag:
        t = next((t for t in spr.tags if t.name == tag), None)
        if t is None:
            raise StudioError(f"no tag {tag!r}; tags are {[t.name for t in spr.tags]}")
        return list(range(t.start, t.end + 1))
    if frames:
        return [spr.check_frame(f) for f in frames]
    return list(range(spr.frame_count))


def save_png(img: np.ndarray, path: Path, scale: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(upscale(img, scale), "RGBA").save(path)


def save_gif(
    frames: list[np.ndarray], durations: list[int], path: Path, scale: int = 1, background=None, loop: bool = True
) -> None:
    """GIF with real 1-bit transparency (or a solid background), one global palette so
    colours do not flicker between frames, and per-frame durations."""
    imgs = [upscale(on_background(f, background) if background else f, scale) for f in frames]
    stack = np.concatenate([i.reshape(-1, 4) for i in imgs])
    opaque = stack[stack[:, 3] >= 128][:, :3]
    cols = np.unique(opaque, axis=0) if len(opaque) else np.zeros((1, 3), np.uint8)
    if len(cols) > 255:  # reduce perceptually, never dither pixel art
        q = Image.fromarray(cols.reshape(1, -1, 3), "RGB").quantize(255, dither=Image.Dither.NONE)
        cols = np.array(q.convert("RGB")).reshape(-1, 3)
        cols = np.unique(cols, axis=0)
    pal = np.zeros((256, 3), np.uint8)
    pal[1 : len(cols) + 1] = cols
    lut = {tuple(c): i + 1 for i, c in enumerate(cols)}
    out = []
    for im in imgs:
        flat = im.reshape(-1, 4)
        idx = np.zeros(len(flat), np.uint8)
        on = flat[:, 3] >= 128
        keys = flat[on][:, :3]
        if len(keys):
            uniq, inv = np.unique(keys, axis=0, return_inverse=True)
            mapped = np.array([lut.get(tuple(u), 0) for u in uniq])
            missing = mapped == 0
            if missing.any():  # colours lost in reduction: nearest kept colour
                from .color import nearest

                mapped[missing] = nearest(uniq[missing].astype(float), cols.astype(float)) + 1
            idx[on] = mapped[inv.ravel()]
        p = Image.fromarray(idx.reshape(im.shape[:2]), "P")
        p.putpalette(pal.ravel().tolist())
        out.append(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    out[0].save(
        path,
        save_all=True,
        append_images=out[1:],
        duration=durations,
        loop=0 if loop else None,
        transparency=0,
        disposal=2,
        optimize=False,
    )


def save_sheet(
    spr: Sprite,
    frames: list[int],
    path: Path,
    layout: str = "horizontal",
    columns: int | None = None,
    padding: int = 0,
    scale: int = 1,
    data_path: Path | None = None,
    layers=None,
) -> dict:
    n = len(frames)
    if layout == "horizontal":
        cols = n
    elif layout == "vertical":
        cols = 1
    else:
        cols = columns or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fw, fh = spr.width * scale, spr.height * scale
    W, H = cols * fw + (cols + 1) * padding, rows * fh + (rows + 1) * padding
    sheet = np.zeros((H, W, 4), np.uint8)
    meta = {
        "frames": [],
        "meta": {
            "size": {"w": W, "h": H},
            "scale": scale,
            "frameTags": [{"name": t.name, "from": t.start, "to": t.end, "direction": t.direction} for t in spr.tags],
        },
    }
    for i, f in enumerate(frames):
        x = padding + (i % cols) * (fw + padding)
        y = padding + (i // cols) * (fh + padding)
        sheet[y : y + fh, x : x + fw] = upscale(render_frame(spr, f, layers), scale)
        meta["frames"].append(
            {"frame": {"x": x, "y": y, "w": fw, "h": fh}, "duration": spr.durations[f], "index": f + 1}
        )
    save_png(sheet, path)
    if data_path:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(json.dumps(meta, indent=1))
    return {"size": [W, H], "columns": cols, "rows": rows}


# ---------------------------------------------------------------------------- views
def _checker(h: int, w: int, cell: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    c = (((yy // cell) + (xx // cell)) % 2).astype(bool)
    out = np.empty((h, w, 4), np.uint8)
    out[...] = (238, 238, 242, 255)
    out[c] = (214, 214, 222, 255)
    return out


def _value(img: np.ndarray) -> np.ndarray:
    lab = rgb_to_oklab(img[..., :3].astype(float))
    v = (np.clip(lab[..., 0], 0, 1) * 255).astype(np.uint8)
    out = img.copy()
    out[..., 0] = out[..., 1] = out[..., 2] = v
    return out


def _silhouette(img: np.ndarray) -> np.ndarray:
    out = np.zeros_like(img)
    out[img[..., 3] > 0] = (20, 20, 28, 255)
    return out


def view_image(
    spr: Sprite,
    frames: list[int],
    scale: int = 0,
    region=None,
    mode: str = "color",
    grid: bool = True,
    rulers: bool = True,
    onion: bool = False,
    layers=None,
    max_side: int = 1600,
) -> bytes:
    """One PNG laying out frames side by side, each over a checkerboard with an optional
    pixel grid (darker every 8 px), coordinate rulers in sprite pixels and a title."""
    x0, y0, w, h = region if region else (0, 0, spr.width, spr.height)
    cols = min(4, len(frames))
    rows_n = int(np.ceil(len(frames) / cols))
    if scale <= 0:
        scale = max(1, min(16, 384 // max(w, h)))
        while scale > 1 and (cols * w * scale > max_side or rows_n * h * scale > max_side):
            scale -= 1
    ruler = 14 if rulers else 0
    title = 10
    cw, ch = w * scale + ruler, h * scale + ruler + title
    gap = 8
    W, H = cols * cw + (cols - 1) * gap, rows_n * ch + (rows_n - 1) * gap
    page = np.zeros((H, W, 4), np.uint8)
    page[...] = (250, 250, 252, 255)
    ink = (40, 40, 56, 255)
    for k, f in enumerate(frames):
        ox = (k % cols) * (cw + gap)
        oy = (k // cols) * (ch + gap)
        img = render_frame(spr, f, layers)
        if onion:
            under = np.zeros_like(img)
            for of, tint in ((f - 1, (220, 60, 60)), (f + 1, (60, 160, 220))):
                if 0 <= of < spr.frame_count:
                    o = render_frame(spr, of, layers)
                    t = o.copy()
                    t[..., :3] = tint
                    t[..., 3] = (o[..., 3] > 0) * 90
                    acc = under.astype(float) / 255
                    composite_onto(acc, t)
                    under = to_uint8(acc)
            acc = under.astype(float) / 255
            composite_onto(acc, img)
            img = to_uint8(acc)
        img = img[y0 : y0 + h, x0 : x0 + w] if region else img
        if mode == "value":
            img = _value(img)
        elif mode == "silhouette":
            img = _silhouette(img)
        tile = _checker(h * scale, w * scale, max(4, scale * 2) if scale > 2 else 8)
        acc = tile.astype(float) / 255
        composite_onto(acc, upscale(img, scale))
        tile = to_uint8(acc)
        if grid and scale >= 6:  # thin translucent lines, so the art stays readable
            g = tile.astype(np.int16)
            for gx in range(0, w):
                k = 0.45 if (x0 + gx) % 8 == 0 else 0.18
                col = g[:, gx * scale]
                g[:, gx * scale] = col + (np.array([60, 60, 90, 255]) - col) * k
            for gy in range(0, h):
                k = 0.45 if (y0 + gy) % 8 == 0 else 0.18
                row = g[gy * scale, :]
                g[gy * scale, :] = row + (np.array([60, 60, 90, 255]) - row) * k
            tile = g.clip(0, 255).astype(np.uint8)
        page[oy + title + ruler : oy + title + ruler + h * scale, ox + ruler : ox + ruler + w * scale] = tile
        _stamp(page, render_text(f"F{f + 1} {spr.durations[f]}MS", ink), ox + ruler, oy + 1)
        if rulers:
            step = 1
            while step * scale < 14:
                step *= 2
            step = max(step, 4 if scale < 6 else step)
            for gx in range(0, w, step):
                _stamp(page, render_text(str(x0 + gx), (90, 90, 110, 255)), ox + ruler + gx * scale + 1, oy + title + 4)
            for gy in range(0, h, step):
                _stamp(page, render_text(str(y0 + gy), (90, 90, 110, 255)), ox + 1, oy + title + ruler + gy * scale + 1)
    buf = io.BytesIO()
    Image.fromarray(page, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def _stamp(page: np.ndarray, img: np.ndarray, x: int, y: int) -> None:
    h, w = img.shape[:2]
    H, W = page.shape[:2]
    if x >= W or y >= H:
        return
    w2, h2 = min(w, W - x), min(h, H - y)
    m = img[:h2, :w2, 3] > 0
    page[y : y + h2, x : x + w2][m] = img[:h2, :w2][m]
