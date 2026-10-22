"""Every editing operation, as plain functions on a Sprite, registered by name.

The `apply` tool runs a list of them as one transaction. Conventions shared by all ops:

* Frames are 1-based. Ops that draw or change cels take `frame` (one) or `frames`: a list,
  "all", a range "3-7", or a mix "1,3,5-8". Default: frame 1.
* `layer` is a name, a group path ("arms/left") or omitted for the top image layer.
* Colours: "#rgb", "#rrggbb", "#rrggbbaa", "r,g,b[,a]" or a name. A transparent colour erases.
* Coordinates are canvas pixels, (0, 0) top-left, y down. Drawing past the canvas edge keeps
  the pixels in the cel (reported as pixels_off_canvas).
* `lock_alpha: true` on a drawing op paints only over pixels that are already opaque.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from . import ase_format
from .color import nearest, parse
from .document import BLEND_MODES, TAG_DIRECTIONS, Cel, Slice, SliceKey, Sprite, StudioError, Tag, trim
from .font import render_text
from .raster import (
    Surface,
    bezier_points,
    ellipse_mask,
    flood_region,
    line_points,
    outline_mask,
    pixel_perfect,
    polyline_points,
)
from .transform import rotate as rotate_img
from .transform import scale_nearest, squash

OPS: dict[str, Callable] = {}
OP_DOCS: dict[str, str] = {}


def op(signature_doc: str):
    def deco(fn):
        OPS[fn.__name__.rstrip("_")] = fn
        OP_DOCS[fn.__name__.rstrip("_")] = signature_doc
        return fn

    return deco


class Ctx:
    """What an op may need beyond the sprite: reading other files through the path policy."""

    def __init__(self, resolve: Callable[[str], Path] | None = None, load_sprite=None):
        self.resolve = resolve or (lambda p: Path(p))
        self.load_sprite = load_sprite or (lambda p: ase_format.read(self.resolve(p)))


def run(spr: Sprite, ctx: Ctx, name: str, args: dict) -> dict:
    fn = OPS.get(name)
    if fn is None:
        raise StudioError(f"unknown op {name!r}; ops are: {', '.join(sorted(OPS))}")
    kwargs = {}
    params = inspect.signature(fn).parameters
    for k, v in (args or {}).items():
        key = k + "_" if (k + "_") in params and k not in params else k
        if key not in params or key in ("spr", "ctx"):
            raise StudioError(f"{name}: unknown argument {k!r}; it takes {OP_DOCS[name]}")
        kwargs[key] = v
    missing = [
        p for p, d in params.items() if d.default is inspect._empty and p not in ("spr", "ctx") and p not in kwargs
    ]
    if missing:
        raise StudioError(f"{name}: missing {missing}; it takes {OP_DOCS[name]}")
    return fn(spr, ctx, **kwargs) or {}


# ---------------------------------------------------------------------------- helpers
def frames_of(spr: Sprite, frame=None, frames=None) -> list[int]:
    spec = frames if frames is not None else (frame if frame is not None else 1)
    if isinstance(spec, int):
        return [spr.check_frame(spec)]
    if isinstance(spec, list):
        return [spr.check_frame(int(f)) for f in spec]
    s = str(spec).strip().lower()
    if s == "all":
        return list(range(spr.frame_count))
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = (int(v) for v in part.split("-", 1))
            out += [spr.check_frame(f) for f in range(a, b + 1)]
        elif part:
            out.append(spr.check_frame(int(part)))
    return out


def _draw(spr: Sprite, layer, frames: list[int], lock_alpha: bool, paint: Callable[[Surface, int], None]) -> dict:
    la = spr.image_layer(layer)
    total, off = 0, 0
    linked: set[int] = set()
    for f in frames:
        s = Surface(spr, la, f)
        s.lock_alpha = lock_alpha
        paint(s, f)
        r = s.commit()
        total += r["pixels_written"]
        off += r.get("pixels_off_canvas", 0)
        linked.update(x + 1 for x in spr.linked_frames(la, f))
    res: dict = {"layer": la.path, "frames": [f + 1 for f in frames], "pixels_written": total}
    if off:
        res["pixels_off_canvas"] = off
    if linked - {f + 1 for f in frames}:
        res["also_changed_linked_frames"] = sorted(linked - {f + 1 for f in frames})
    return res


def _xy(points) -> tuple[np.ndarray, np.ndarray]:
    a = np.array(points, np.int64).reshape(-1, 2)
    return a[:, 0], a[:, 1]


# ---------------------------------------------------------------------------- layers
@op("name, group?, opacity?=255, blend?=normal, visible?=true, reference?=false")
def add_layer(
    spr,
    ctx,
    name: str,
    group: str | None = None,
    opacity: int = 255,
    blend: str = "normal",
    visible: bool = True,
    reference: bool = False,
):
    if blend not in BLEND_MODES:
        raise StudioError(f"blend must be one of {BLEND_MODES}")
    la = spr.add_layer(name, group, opacity=opacity, blend=blend, visible=visible, reference=reference)
    return {"added_layer": la.path}


@op("name, group?")
def add_group(spr, ctx, name: str, group: str | None = None):
    return {"added_group": spr.add_layer(name, group, kind="group").path}


@op("layer")
def remove_layer(spr, ctx, layer: str):
    spr.remove_layer(layer)
    return {"removed_layer": layer}


@op("layer, name?, opacity?, blend?, visible?, editable?")
def set_layer(
    spr,
    ctx,
    layer: str,
    name: str | None = None,
    opacity: int | None = None,
    blend: str | None = None,
    visible: bool | None = None,
    editable: bool | None = None,
):
    la = spr.layer(layer)
    if name is not None:
        la.name = name
    if opacity is not None:
        la.opacity = int(opacity)
    if blend is not None:
        if blend not in BLEND_MODES:
            raise StudioError(f"blend must be one of {BLEND_MODES}")
        la.blend = blend
    if visible is not None:
        la.visible = bool(visible)
    if editable is not None:
        la.editable = bool(editable)
    return {"layer": la.path}


@op("layer, to (1 = bottom among its siblings)")
def move_layer(spr, ctx, layer: str, to: int):
    spr.move_layer(layer, int(to))
    return {"order": [la.path for la in spr.layers]}


@op("layer, name")
def duplicate_layer(spr, ctx, layer: str, name: str):
    src = spr.image_layer(layer)
    la = spr.add_layer(name, src.parent.path if src.parent else None, opacity=src.opacity, blend=src.blend)
    copies: dict[int, Cel] = {}
    for f, c in src.cels.items():
        copies.setdefault(id(c), c.clone())
        la.cels[f] = copies[id(c)]
    spr.layers.remove(la)
    spr.layers.insert(spr.layers.index(src) + 1, la)
    return {"added_layer": la.path}


# ---------------------------------------------------------------------------- frames
@op("count?=1, at? (1-based position to insert before; default: append), duration?=100, copy_from?")
def add_frames(spr, ctx, count: int = 1, at: int | None = None, duration: int = 100, copy_from: int | None = None):
    pos = spr.frame_count if at is None else spr.check_frame(at)
    src = spr.check_frame(copy_from) if copy_from is not None else None
    spr.insert_frames(pos, count, duration)
    if src is not None:
        src_i = src + count if src >= pos else src
        for la in spr.layers:
            c = la.cels.get(src_i)
            if c is not None:
                for k in range(count):
                    la.cels[pos + k] = c.clone()
    return {"frame_count": spr.frame_count, "new_frames": list(range(pos + 1, pos + count + 1))}


@op("frames")
def remove_frames(spr, ctx, frames):
    for f in sorted(set(frames_of(spr, frames=frames)), reverse=True):
        spr.remove_frame(f)
    return {"frame_count": spr.frame_count}


@op("frame, linked?=false")
def duplicate_frame(spr, ctx, frame: int, linked: bool = False):
    return {"new_frame": spr.duplicate_frame(spr.check_frame(frame), linked) + 1, "frame_count": spr.frame_count}


@op("frame, to")
def move_frame(spr, ctx, frame: int, to: int):
    spr.move_frame(spr.check_frame(frame), spr.check_frame(to))
    return {}


@op("ms, frames?=all")
def set_duration(spr, ctx, ms: int, frames="all"):
    fs = frames_of(spr, frames=frames)
    for f in fs:
        spr.durations[f] = max(1, int(ms))
    return {"durations": spr.durations}


@op("durations (list of ms, one per frame from frame 1)")
def set_durations(spr, ctx, durations: list):
    if len(durations) != spr.frame_count:
        raise StudioError(f"give {spr.frame_count} durations, one per frame")
    spr.durations = [max(1, int(d)) for d in durations]
    return {"durations": spr.durations}


@op("frames (an unbroken range)")
def reverse_frames(spr, ctx, frames):
    fs = sorted(frames_of(spr, frames=frames))
    if fs != list(range(fs[0], fs[-1] + 1)):
        raise StudioError("reverse_frames needs an unbroken range such as '3-8'")
    order = list(range(spr.frame_count))
    order[fs[0] : fs[-1] + 1] = reversed(order[fs[0] : fs[-1] + 1])
    spr.reorder_frames(order)
    return {}


# ---------------------------------------------------------------------------- tags & slices
@op("name, from, to, direction?=forward, repeat?=0 (0 = loop forever), color?")
def add_tag(
    spr, ctx, name: str, from_: int, to: int, direction: str = "forward", repeat: int = 0, color: str | None = None
):
    if direction not in TAG_DIRECTIONS:
        raise StudioError(f"direction must be one of {TAG_DIRECTIONS}")
    a, b = spr.check_frame(from_), spr.check_frame(to)
    if b < a:
        raise StudioError("tag end is before its start")
    spr.tags = [t for t in spr.tags if t.name != name]
    spr.tags.append(Tag(name, a, b, direction, int(repeat), parse(color) if color else (90, 90, 90, 255)))
    return {"tags": [t.name for t in spr.tags]}


@op("name")
def remove_tag(spr, ctx, name: str):
    spr.tags = [t for t in spr.tags if t.name != name]
    return {}


@op("name, x, y, w, h, frame?=1, pivot? [x,y], center? [x,y,w,h], data?")
def add_slice(
    spr, ctx, name: str, x: int, y: int, w: int, h: int, frame: int = 1, pivot=None, center=None, data: str = ""
):
    key = SliceKey(
        spr.check_frame(frame), x, y, w, h, tuple(center) if center else None, tuple(pivot) if pivot else None
    )
    s = next((s for s in spr.slices if s.name == name), None)
    if s is None:
        spr.slices.append(Slice(name, [key], data))
    else:
        s.keys = [k for k in s.keys if k.frame != key.frame] + [key]
        s.user_text = data or s.user_text
    return {}


@op("name")
def remove_slice(spr, ctx, name: str):
    spr.slices = [s for s in spr.slices if s.name != name]
    return {}


# ---------------------------------------------------------------------------- cels
@op("layer, from, to (frames), to_layer?, linked?=false, dx?=0, dy?=0")
def copy_cel(
    spr, ctx, layer: str, from_: int, to, to_layer: str | None = None, linked: bool = False, dx: int = 0, dy: int = 0
):
    src_l = spr.image_layer(layer)
    dst_l = spr.image_layer(to_layer) if to_layer else src_l
    c = src_l.cels.get(spr.check_frame(from_))
    targets = frames_of(spr, frames=to)
    for f in targets:
        if c is None:
            dst_l.cels.pop(f, None)
            continue
        if linked and not (dx or dy) and dst_l is src_l:
            dst_l.cels[f] = c
        else:
            n = c.clone()
            n.x += dx
            n.y += dy
            dst_l.cels[f] = n
    return {"copied_to": [f + 1 for f in targets]}


@op("layer, dx, dy, frames?")
def move_cel(spr, ctx, layer: str, dx: int, dy: int, frame=None, frames=None):
    la = spr.image_layer(layer)
    seen = set()
    for f in frames_of(spr, frame, frames):
        c = la.cels.get(f)
        if c is not None and id(c) not in seen:
            seen.add(id(c))
            c.x += int(dx)
            c.y += int(dy)
    return {}


@op("layer, frames?, opacity?, z?")
def set_cel(spr, ctx, layer: str, frame=None, frames=None, opacity: int | None = None, z: int | None = None):
    la = spr.image_layer(layer)
    for f in frames_of(spr, frame, frames):
        c = la.cels.get(f)
        if c is None:
            continue
        if opacity is not None:
            spr.unlink(la, f)
            la.cels[f].opacity = max(0, min(255, int(opacity)))
        if z is not None:
            la.cels[f].z = int(z)
    return {}


@op("layer?, frames?, region? [x,y,w,h]")
def clear(spr, ctx, layer=None, frame=None, frames=None, region=None):
    la = spr.image_layer(layer)
    fs = frames_of(spr, frame, frames)
    if region is None:
        for f in fs:
            la.cels.pop(f, None)
        return {"cleared": [f + 1 for f in fs]}
    x, y, w, h = region

    def paint(s: Surface, f: int):
        _wipe(s, x, y, w, h)

    return _draw(spr, layer, fs, False, paint)


def _wipe(s: Surface, x: int, y: int, w: int, h: int) -> None:
    """Make a rectangle transparent, counting the pixels that were there."""
    v = s.view(x, y, w, h)
    s.written += int((v[..., 3] > 0).sum())
    v[...] = 0


@op("layer, source (frame), frames")
def link_cels(spr, ctx, layer: str, source: int, frames):
    la = spr.image_layer(layer)
    c = la.cels.get(spr.check_frame(source))
    if c is None:
        raise StudioError(f"frame {source} of {layer!r} has no cel to link")
    for f in frames_of(spr, frames=frames):
        la.cels[f] = c
    return {}


@op("layer, frames")
def unlink_cels(spr, ctx, layer: str, frames):
    la = spr.image_layer(layer)
    for f in frames_of(spr, frames=frames):
        spr.unlink(la, f)
    return {}


# ---------------------------------------------------------------------------- drawing
@op("points [[x,y],...] or [[x,y,color],...], color?, layer?, frames?, lock_alpha?")
def pixels(
    spr, ctx, points: list, color: str | None = None, layer=None, frame=None, frames=None, lock_alpha: bool = False
):
    def paint(s: Surface, f: int):
        if color is not None:
            xs, ys = _xy([p[:2] for p in points])
            s.plot(xs, ys, parse(color))
        else:
            for p in points:
                if len(p) < 3:
                    raise StudioError("each point needs [x, y, color] when no shared color is given")
                s.plot([p[0]], [p[1]], parse(p[2]))

    return _draw(spr, layer, frames_of(spr, frame, frames), lock_alpha, paint)


@op("rows (one string per row), legend {char: color}, x?=0, y?=0, layer?, frames?, lock_alpha?  ('.' = leave)")
def map_(
    spr,
    ctx,
    rows: list,
    legend: dict,
    x: int = 0,
    y: int = 0,
    layer=None,
    frame=None,
    frames=None,
    lock_alpha: bool = False,
):
    if not rows:
        raise StudioError("map needs at least one row")
    w = len(rows[0])
    bad = [i for i, r in enumerate(rows) if len(r) != w]
    if bad:
        raise StudioError(f"map rows must all be {w} characters; rows {bad[:5]} are not (padding would shift pixels)")
    chars = set("".join(rows)) - {"."}
    unknown = chars - set(legend)
    if unknown:
        raise StudioError(f"map uses characters not in the legend: {sorted(unknown)}")
    cols = {k: parse(v) for k, v in legend.items()}
    grid = np.array([list(r) for r in rows])

    def paint(s: Surface, f: int):
        for ch, col in cols.items():
            ys, xs = np.nonzero(grid == ch)
            if xs.size:
                s.plot(xs + x, ys + y, col)

    return _draw(spr, layer, frames_of(spr, frame, frames), lock_alpha, paint)


@op("x0, y0, x1, y1, color, perfect?=true, layer?, frames?, lock_alpha?")
def line(
    spr,
    ctx,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: str,
    perfect: bool = True,
    layer=None,
    frame=None,
    frames=None,
    lock_alpha: bool = False,
):
    pts = line_points(x0, y0, x1, y1)
    pts = pixel_perfect(pts) if perfect else pts
    return _draw(spr, layer, frames_of(spr, frame, frames), lock_alpha, lambda s, f: s.plot(*_xy(pts), parse(color)))


@op("points [[x,y],...], color, closed?=false, perfect?=true, layer?, frames?")
def polyline(
    spr,
    ctx,
    points: list,
    color: str,
    closed: bool = False,
    perfect: bool = True,
    layer=None,
    frame=None,
    frames=None,
    lock_alpha: bool = False,
):
    pts = polyline_points([tuple(p) for p in points], closed, perfect)
    return _draw(spr, layer, frames_of(spr, frame, frames), lock_alpha, lambda s, f: s.plot(*_xy(pts), parse(color)))


@op("points (3 = quadratic, 4 = cubic Bezier control points), color, layer?, frames?")
def curve(spr, ctx, points: list, color: str, layer=None, frame=None, frames=None, lock_alpha: bool = False):
    pts = bezier_points([tuple(p) for p in points])
    return _draw(spr, layer, frames_of(spr, frame, frames), lock_alpha, lambda s, f: s.plot(*_xy(pts), parse(color)))


@op("x, y, w, h, color, fill?=false, layer?, frames?, lock_alpha?")
def rect(
    spr,
    ctx,
    x: int,
    y: int,
    w: int,
    h: int,
    color: str,
    fill: bool = False,
    layer=None,
    frame=None,
    frames=None,
    lock_alpha: bool = False,
):
    m = np.ones((h, w), bool)
    if not fill and w > 2 and h > 2:
        m[1:-1, 1:-1] = False
    ys, xs = np.nonzero(m)
    return _draw(
        spr, layer, frames_of(spr, frame, frames), lock_alpha, lambda s, f: s.plot(xs + x, ys + y, parse(color))
    )


@op("x, y, w, h (bounding box, even sizes fine), color, fill?=true, layer?, frames?, lock_alpha?")
def ellipse(
    spr,
    ctx,
    x: int,
    y: int,
    w: int,
    h: int,
    color: str,
    fill: bool = True,
    layer=None,
    frame=None,
    frames=None,
    lock_alpha: bool = False,
):
    ys, xs = np.nonzero(ellipse_mask(w, h, fill))
    return _draw(
        spr, layer, frames_of(spr, frame, frames), lock_alpha, lambda s, f: s.plot(xs + x, ys + y, parse(color))
    )


@op("x, y, color, tolerance?=0, diagonal?=false, layer?, frames?")
def fill(
    spr,
    ctx,
    x: int,
    y: int,
    color: str,
    tolerance: int = 0,
    diagonal: bool = False,
    layer=None,
    frame=None,
    frames=None,
):
    def paint(s: Surface, f: int):
        cv = s.canvas()
        if not (0 <= x < spr.width and 0 <= y < spr.height):
            raise StudioError("fill start point is outside the canvas")
        m = flood_region(cv, x, y, tolerance, diagonal)
        ys, xs = np.nonzero(m)
        s.plot(xs, ys, parse(color))

    return _draw(spr, layer, frames_of(spr, frame, frames), False, paint)


@op(
    "text, x, y, color, font?=tiny|bold, bottom_color?, outline?, shadow?, scale?=1, anchor?=topleft|center|"
    "bottom|topright, layer?, frames?"
)
def text(
    spr,
    ctx,
    text: str,
    x: int,
    y: int,
    color: str,
    font: str = "tiny",
    bottom_color: str | None = None,
    outline: str | None = None,
    shadow: str | None = None,
    scale: int = 1,
    anchor: str = "topleft",
    spacing: int = 1,
    layer=None,
    frame=None,
    frames=None,
):
    img = render_text(
        text,
        parse(color),
        font,
        parse(bottom_color) if bottom_color else None,
        parse(outline) if outline else None,
        True,
        parse(shadow) if shadow else None,
        scale,
        spacing,
    )
    h, w = img.shape[:2]
    ox = {"topleft": 0, "center": w // 2, "bottom": w // 2, "topright": w - 1, "top": w // 2}.get(anchor, 0)
    oy = {"center": h // 2, "bottom": h - 1}.get(anchor, 0)
    res = _draw(
        spr, layer, frames_of(spr, frame, frames), False, lambda s, f: s.put_image(img, x - ox, y - oy, blend=False)
    )
    res["text_box"] = [x - ox, y - oy, w, h]
    return res


def _load_image(ctx: Ctx, source: str, source_frame: int = 1, source_layer: str | None = None) -> np.ndarray:
    p = ctx.resolve(source)
    if p.suffix.lower() in (".aseprite", ".ase"):
        from .render import render_frame

        s = ctx.load_sprite(source)
        return render_frame(s, s.check_frame(source_frame), [source_layer] if source_layer else None)
    return np.array(Image.open(p).convert("RGBA"))


@op("source (png/aseprite path), x?=0, y?=0, source_frame?=1, source_layer?, blend?=true, layer?, frames?")
def stamp(
    spr,
    ctx,
    source: str,
    x: int = 0,
    y: int = 0,
    source_frame: int = 1,
    source_layer: str | None = None,
    blend: bool = True,
    layer=None,
    frame=None,
    frames=None,
    lock_alpha: bool = False,
):
    img = _load_image(ctx, source, source_frame, source_layer)
    return _draw(
        spr, layer, frames_of(spr, frame, frames), lock_alpha, lambda s, f: s.put_image(img, x, y, blend=blend)
    )


@op("x, y, w, h, layer?, frames?")
def erase(spr, ctx, x: int, y: int, w: int, h: int, layer=None, frame=None, frames=None):
    def paint(s: Surface, f: int):
        _wipe(s, x, y, w, h)

    return _draw(spr, layer, frames_of(spr, frame, frames), False, paint)


@op("color, where?=outside|inside, diagonal?=false, layer?, frames?")
def outline(spr, ctx, color: str, where: str = "outside", diagonal: bool = False, layer=None, frame=None, frames=None):
    def paint(s: Surface, f: int):
        s.ensure(s.x0 - 1, s.y0 - 1, s.x0 + s.buf.shape[1] + 1, s.y0 + s.buf.shape[0] + 1)
        m = outline_mask(s.buf[..., 3], diagonal, inside=(where == "inside"))
        ys, xs = np.nonzero(m)
        s.plot(xs + s.x0, ys + s.y0, parse(color))

    return _draw(spr, layer, frames_of(spr, frame, frames), False, paint)


@op("from, to, tolerance?=0, layer?, frames?")
def replace_color(spr, ctx, from_: str, to: str, tolerance: int = 0, layer=None, frame=None, frames=None):
    return remap(spr, ctx, {from_: to}, tolerance, layer, frame, frames)


@op("mapping {from_color: to_color}, tolerance?=0, layer?, frames?")
def remap(spr, ctx, mapping: dict, tolerance: int = 0, layer=None, frame=None, frames=None):
    pairs = [(np.array(parse(a), np.int16), np.array(parse(b), np.uint8)) for a, b in mapping.items()]
    changed = {"n": 0}

    def paint(s: Surface, f: int):
        b = s.buf
        orig = b.astype(np.int16)
        hit_any = np.zeros(b.shape[:2], bool)
        for src, dst in pairs:
            hit = (np.abs(orig - src).max(-1) <= tolerance) & ~hit_any & (orig[..., 3] > 0)
            b[hit] = dst
            hit_any |= hit
        changed["n"] += int(hit_any.sum())

    res = _draw(spr, layer, frames_of(spr, frame, frames), False, paint)
    res["pixels_changed"] = changed["n"]
    return res


@op("color?=#ffffff, amount?=1.0 (1 = solid silhouette), layer?, frames?  (hit flash)")
def flash(spr, ctx, color: str = "#ffffff", amount: float = 1.0, layer=None, frame=None, frames=None):
    c = np.array(parse(color)[:3], float)

    def paint(s: Surface, f: int):
        on = s.buf[..., 3] > 0
        rgb = s.buf[..., :3].astype(float)
        new = np.where(on[..., None], np.round(rgb + (c - rgb) * amount), rgb).astype(np.uint8)
        s.written += int((new != s.buf[..., :3]).any(-1).sum())
        s.buf[..., :3] = new

    return _draw(spr, layer, frames_of(spr, frame, frames), False, paint)


@op("axis?=h|v, center? (canvas x for h, y for v; default canvas middle), layer?, frames?")
def flip(spr, ctx, axis: str = "h", center: float | None = None, layer=None, frame=None, frames=None):
    la = spr.image_layer(layer)
    seen = set()
    for f in frames_of(spr, frame, frames):
        c = la.cels.get(f)
        if c is None or id(c) in seen:
            continue
        seen.add(id(c))
        x, y, w, h = c.bounds
        if axis == "h":
            m = (spr.width / 2) if center is None else center + 0.5
            c.image = c.image[:, ::-1].copy()
            c.x = int(round(2 * m - (x + w)))
        else:
            m = (spr.height / 2) if center is None else center + 0.5
            c.image = c.image[::-1].copy()
            c.y = int(round(2 * m - (y + h)))
    return {}


@op("degrees (clockwise), pivot [x,y] (canvas), method?=rotsprite|nearest, layer?, frames?")
def rotate(spr, ctx, degrees: float, pivot: list, method: str = "rotsprite", layer=None, frame=None, frames=None):
    la = spr.image_layer(layer)
    for f in frames_of(spr, frame, frames):
        spr.unlink(la, f)
        c = la.cels.get(f)
        if c is None:
            continue
        img, dx, dy = rotate_img(c.image, degrees, (pivot[0] - c.x, pivot[1] - c.y), method)
        t = trim(img, c.x + dx, c.y + dy)
        if t:
            c.image, c.x, c.y = t
    return {}


@op("amount (0.2 = squash 20%, -0.2 = stretch), anchor?=bottom|top|left|right, layer?, frames?")
def squash_stretch(spr, ctx, amount: float, anchor: str = "bottom", layer=None, frame=None, frames=None):
    la = spr.image_layer(layer)
    for f in frames_of(spr, frame, frames):
        spr.unlink(la, f)
        c = la.cels.get(f)
        if c is None:
            continue
        img, dx, dy = squash(c.image, amount, anchor)
        c.image, c.x, c.y = img, c.x + dx, c.y + dy
    return {}


@op("factor (or sx, sy), anchor?=topleft|center|bottom, layer?, frames?  (nearest neighbour)")
def scale(
    spr,
    ctx,
    factor: float | None = None,
    sx: float | None = None,
    sy: float | None = None,
    anchor: str = "topleft",
    layer=None,
    frame=None,
    frames=None,
):
    fx = sx or factor
    fy = sy or factor
    if not fx or not fy:
        raise StudioError("give factor, or sx and sy")
    la = spr.image_layer(layer)
    for f in frames_of(spr, frame, frames):
        spr.unlink(la, f)
        c = la.cels.get(f)
        if c is None:
            continue
        h, w = c.image.shape[:2]
        img = scale_nearest(c.image, fx, fy)
        nh, nw = img.shape[:2]
        if anchor == "center":
            c.x -= (nw - w) // 2
            c.y -= (nh - h) // 2
        elif anchor == "bottom":
            c.x -= (nw - w) // 2
            c.y -= nh - h
        c.image = img
    return {}


@op("palette? [colors] or source? (.aseprite/.png whose colours to use), layer?, frames?")
def palette_lock(spr, ctx, palette: list | None = None, source: str | None = None, layer=None, frame=None, frames=None):
    """Map every pixel to the perceptually nearest colour of a palette (OKLab)."""
    if source:
        img = _load_image(ctx, source)
        pal = np.unique(img[img[..., 3] > 0][:, :3], axis=0).astype(float)
    elif palette:
        pal = np.array([parse(c)[:3] for c in palette], float)
    else:
        raise StudioError("give palette or source")
    changed = {"n": 0}

    def paint(s: Surface, f: int):
        on = s.buf[..., 3] > 0
        if not on.any():
            return
        cols = s.buf[on][:, :3]
        uniq, inv = np.unique(cols, axis=0, return_inverse=True)
        mapped = pal[nearest(uniq.astype(float), pal)].astype(np.uint8)
        new = mapped[inv.ravel()]
        changed["n"] += int((new != cols).any(1).sum())
        s.buf[on, :3] = new

    res = _draw(spr, layer, frames_of(spr, frame, frames), False, paint)
    res["pixels_changed"] = changed["n"]
    res["palette_size"] = len(pal)
    return res


@op("specks?=2 (max size of detached islands to delete), orphans?=true, protect? [colors], layer?, frames?")
def clean(
    spr, ctx, specks: int = 2, orphans: bool = True, protect: list | None = None, layer=None, frame=None, frames=None
):
    from .pixelize import clean_orphans, remove_specks

    stats = {"specks": 0, "orphans": 0}
    prot = [parse(c)[:3] for c in (protect or [])]

    def paint(s: Surface, f: int):
        if specks:
            stats["specks"] += remove_specks(s.buf, specks)
        if orphans:
            stats["orphans"] += clean_orphans(s.buf, prot)

    res = _draw(spr, layer, frames_of(spr, frame, frames), False, paint)
    res.update({"specks_removed": stats["specks"], "orphans_fixed": stats["orphans"]})
    return res


@op("colors [list], replace?=true")
def set_palette(spr, ctx, colors: list, replace: bool = True):
    pal = [parse(c) for c in colors]
    spr.palette = pal if replace else spr.palette + pal
    return {"palette_size": len(spr.palette)}


# ---------------------------------------------------------------------------- motion & fx
def _ease(t: np.ndarray, ease: str) -> np.ndarray:
    if ease == "ease_in":
        return t * t
    if ease == "ease_out":
        return 1 - (1 - t) ** 2
    if ease == "ease_in_out":
        return np.where(t < 0.5, 2 * t * t, 1 - (-2 * t + 2) ** 2 / 2)
    if ease == "back_out":  # overshoot then settle: the "pop" of a damage number
        c1 = 1.70158
        return 1 + (c1 + 1) * (t - 1) ** 3 + c1 * (t - 1) ** 2
    return t


@op("layer, frames (in order), dx?, dy?, ease?=linear|ease_in|ease_out|ease_in_out|back_out, arc?=0")
def tween_move(spr, ctx, layer: str, frames, dx: int = 0, dy: int = 0, ease: str = "linear", arc: float = 0):
    """Moves each listed frame's cel along the path; the first stays, the last lands on
    (dx, dy). Positions are rounded from the running total so spacing never drifts."""
    la = spr.image_layer(layer)
    fs = frames_of(spr, frames=frames)
    if len(fs) < 2:
        raise StudioError("tween_move needs at least two frames")
    t = np.linspace(0, 1, len(fs))
    e = _ease(t, ease)
    xs = np.round(e * dx).astype(int)
    ys = np.round(e * dy - arc * 4 * t * (1 - t)).astype(int)
    for k, f in enumerate(fs[1:], 1):
        spr.unlink(la, f)
        c = la.cels.get(f)
        if c is not None:
            c.x += int(xs[k])
            c.y += int(ys[k])
    return {"offsets": [[int(a), int(b)] for a, b in zip(xs, ys, strict=False)]}


@op("layer, frames, amplitude?=1, axis?=x|y|both")
def shake(spr, ctx, layer: str, frames, amplitude: int = 1, axis: str = "x"):
    la = spr.image_layer(layer)
    pattern = [1, -1, 1, 0, -1, 1, 0]
    for k, f in enumerate(frames_of(spr, frames=frames)):
        spr.unlink(la, f)
        c = la.cels.get(f)
        if c is None:
            continue
        a = pattern[k % len(pattern)] * amplitude
        if axis in ("x", "both"):
            c.x += a
        if axis in ("y", "both"):
            c.y += -a if axis == "both" else a
    return {}


@op("layer, frames, from?=255, to?=0, ease?=linear")
def fade(spr, ctx, layer: str, frames, from_: int = 255, to: int = 0, ease: str = "linear"):
    la = spr.image_layer(layer)
    fs = frames_of(spr, frames=frames)
    vals = from_ + (to - from_) * _ease(np.linspace(0, 1, len(fs)), ease)
    for f, v in zip(fs, vals, strict=False):
        spr.unlink(la, f)
        c = la.cels.get(f)
        if c is not None:
            c.opacity = int(round(max(0, min(255, v))))
    return {"opacities": [int(round(v)) for v in vals]}


@op(
    "text, x, y (where it pops: its bottom centre), frames, rise?=10, layer?=damage, color?=#fff1a8, "
    "bottom_color?=#ff5a3c, outline?=#3a0c14, font?=bold, pop?=true, fade_frames?=2"
)
def damage_number(
    spr,
    ctx,
    text: str,
    x: int,
    y: int,
    frames,
    rise: int = 10,
    layer: str = "damage",
    color: str = "#fff1a8",
    bottom_color: str | None = "#ff5a3c",
    outline: str | None = "#3a0c14",
    font: str = "bold",
    pop: bool = True,
    fade_frames: int = 2,
):
    """A damage/score number that pops up, rises with ease-out (overshooting first if
    `pop`), and fades out over its last `fade_frames` frames. Creates `layer` if needed."""
    fs = frames_of(spr, frames=frames)
    try:
        la = spr.image_layer(layer)
    except StudioError:
        la = spr.add_layer(layer)
    img = render_text(
        text, parse(color), font, parse(bottom_color) if bottom_color else None, parse(outline) if outline else None
    )
    h, w = img.shape[:2]
    t = np.linspace(0, 1, len(fs))
    path = -rise * _ease(t, "back_out" if pop else "ease_out")
    for k, f in enumerate(fs):
        la.cels[f] = Cel(img.copy(), x - w // 2, int(round(y - h + 1 + path[k])))
        left = len(fs) - k
        if fade_frames and left <= fade_frames:
            la.cels[f].opacity = int(255 * (left / (fade_frames + 1)))
    return {"layer": la.path, "frames": [f + 1 for f in fs], "y_offsets": [int(round(v)) for v in path]}


@op("layer?, frames, color?=#ffffff (white flash) — alternates flash on/off across the frames")
def blink_flash(spr, ctx, frames, layer=None, color: str = "#ffffff", amount: float = 1.0):
    fs = frames_of(spr, frames=frames)
    return flash(spr, ctx, color, amount, layer, frames=[f + 1 for f in fs[::2]])


def describe_ops() -> str:
    return "\n".join(f"  {name}({OP_DOCS[name]})" for name in sorted(OPS))
