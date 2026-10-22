"""The MCP server: tools over the store, ops, renderer, pixelizer, animator and reviewer."""

from __future__ import annotations

import functools
import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image as PILImage

from . import __version__, animate, ase_format, ops, render, review, tween
from .color import parse, ramp, to_hex
from .document import Cel, StudioError, trim
from .pixelize import pixelize as pixelize_image
from .store import Store

INSTRUCTIONS = """\
aseprite-mcp-studio edits .aseprite files directly (no Aseprite process), so every call takes
milliseconds. Frames are 1-based; coordinates are canvas pixels, (0,0) top-left, y down.

Typical work:
* New art: sprite_new -> apply (a list of ops: layers, frames, map/line/rect/ellipse/fill/text,
  cel moves, tags...) -> view to look -> lint / check_animation -> export.
* A character image (also fake pixel art from an image model): pixelize -> it finds the real pixel
  grid and writes a native-resolution sprite plus a side-by-side preview to approve.
* Animating it: rig_preview (cut it into parts with pivots; the preview colours each part) ->
  animate_rig with a preset (idle, walk_side, attack, hurt, death) or your own keyframes ->
  view with onion=true -> check_animation and check_identity against the master.
* A reference strip of poses (drawn or generated): import_strip pixelizes every frame at the
  master's scale, locks it to the master's palette and stands each frame on the master's ground.
* Two key poses (start and end drawings): rig both with the same part names, then tween_poses
  makes the in-betweens; add anticipation / overshoot with `extra`.
* After every animation pass: find_glitches (each frame against the previous and next frame and
  the master, onion-skin view with every problem boxed), then apply op fix_glitches.
Every edit is one transaction (all or nothing) and can be undone with undo.
"""

store = Store()
mcp = MCPServer("aseprite-mcp-studio", instructions=INSTRUCTIONS, version=__version__)


def tool(fn=None, *, description: str | None = None):
    """Register a tool; StudioError becomes a clean ToolError message."""

    def deco(f):
        @functools.wraps(f)
        def wrapper(*a, **kw):
            try:
                return f(*a, **kw)
            except StudioError as e:
                raise ToolError(str(e)) from None

        mcp.tool(description=description)(wrapper)
        return f

    return deco(fn) if fn is not None else deco


def _ctx() -> ops.Ctx:
    return ops.Ctx(resolve=lambda p: store.resolve(p, must_exist=True), load_sprite=lambda p: store.load(p)[1])


def _png(img: np.ndarray, scale: int = 1) -> Image:
    buf = io.BytesIO()
    PILImage.fromarray(render.upscale(img, scale), "RGBA").save(buf, "PNG")
    return Image(data=buf.getvalue(), format="png")


def _load_rgba(path: str, frame: int = 1, layer: str | None = None) -> np.ndarray:
    p = store.resolve(path, must_exist=True)
    if p.suffix.lower() in (".aseprite", ".ase"):
        _, s = store.load(path)
        return render.render_frame(s, s.check_frame(frame), [layer] if layer else None)
    return np.array(PILImage.open(p).convert("RGBA"))


def _frames(spr, frames) -> list[int]:
    if frames in (None, "", "all"):
        return list(range(spr.frame_count))
    return ops.frames_of(spr, frames=frames)


# ---------------------------------------------------------------------------- files
@tool
def sprite_new(
    path: str,
    width: int,
    height: int,
    frames: int = 1,
    layers: list[str] | None = None,
    duration_ms: int = 100,
    overwrite: bool = False,
) -> dict:
    """Create a new transparent RGBA .aseprite sprite. `layers` are created bottom to top
    (default one layer named "Layer 1")."""
    names = layers or ["Layer 1"]
    spr = ase_format.new_sprite(width, height, names[0], frames, duration_ms)
    for n in names[1:]:
        spr.add_layer(n)
    p = store.replace(path, spr, overwrite)
    return {"path": str(p), "width": width, "height": height, "frames": frames, "layers": names}


@tool
def sprite_info(path: str) -> dict:
    """Structure of a sprite: size, frame durations, layer tree (with which frames have cels and
    which are linked), tags, slices, palette size and warnings."""
    p, spr = store.load(path)
    layers = []
    for la in spr.layers:
        d: dict[str, Any] = {"path": la.path, "kind": la.kind, "visible": la.visible, "opacity": la.opacity}
        if la.blend != "normal":
            d["blend"] = la.blend
        if la.kind == "image":
            d["cels"] = sorted(f + 1 for f in la.cels)
            links = {}
            for f, c in la.cels.items():
                links.setdefault(id(c), []).append(f + 1)
            linked = [sorted(v) for v in links.values() if len(v) > 1]
            if linked:
                d["linked"] = linked
        layers.append(d)
    return {
        "path": str(p),
        "width": spr.width,
        "height": spr.height,
        "frames": spr.frame_count,
        "durations_ms": spr.durations,
        "layers": layers,
        "tags": [
            {"name": t.name, "from": t.start + 1, "to": t.end + 1, "direction": t.direction, "repeat": t.repeat}
            for t in spr.tags
        ],
        "slices": [{"name": s.name, "keys": [[k.frame + 1, k.x, k.y, k.w, k.h] for k in s.keys]} for s in spr.slices],
        "palette_size": len(spr.palette),
        "warnings": spr.warnings,
        "history": store.history(path),
    }


class _DryRun(Exception):
    def __init__(self, results):
        self.results = results


APPLY_DOC = """Run edit operations on a sprite as ONE transaction: if any op fails nothing is saved and
the error names the op. ops_list items are {"op": name, ...arguments}. Frames are 1-based; drawing
ops take frame (one) or frames (list, "all", "2-6", "1,3,5-8"); layer defaults to the top image
layer. Pixels drawn past the canvas edge are kept in the cel, never clipped. dry_run=true runs
everything and reports, but saves nothing.

Ops (name(arguments), ? = optional):
""" + ops.describe_ops()


@tool(description=APPLY_DOC)
def apply(path: str, ops_list: list[dict], dry_run: bool = False) -> dict:
    results: list[dict] = []
    try:
        with store.edit(path) as (_, spr):
            ctx = _ctx()
            for i, item in enumerate(ops_list):
                if not isinstance(item, dict) or "op" not in item:
                    raise StudioError(f"op #{i + 1} must be an object with an 'op' key")
                args = {k: v for k, v in item.items() if k != "op"}
                try:
                    r = ops.run(spr, ctx, item["op"], args)
                except StudioError as e:
                    raise StudioError(f"op #{i + 1} ({item['op']}) failed, nothing was saved: {e}") from None
                except (KeyError, IndexError, TypeError, ValueError) as e:
                    raise StudioError(f"op #{i + 1} ({item['op']}) failed, nothing was saved: {e!r}") from None
                results.append({"op": item["op"], **r})
            if dry_run:
                raise _DryRun(results)
    except _DryRun:
        return {"ok": True, "dry_run": True, "saved": False, "results": results}
    return {"ok": True, "frames": spr.frame_count, "layers": [la.path for la in spr.layers], "results": results}


@tool
def undo(path: str, steps: int = 1) -> dict:
    """Undo the last edits made through this server to a file (up to 50 per file)."""
    return {"undone": store.undo(path, steps), **store.history(path)}


@tool
def redo(path: str, steps: int = 1) -> dict:
    """Redo edits that were undone."""
    return {"redone": store.redo(path, steps), **store.history(path)}


# ---------------------------------------------------------------------------- looking
@tool
def view(
    path: str,
    frames: Any = "1",
    scale: int = 0,
    region: list[int] | None = None,
    mode: str = "color",
    onion: bool = False,
    layers: list[str] | None = None,
    grid: bool = True,
    rulers: bool = True,
) -> Image:
    """Look at frames: one image with up to 16 frames side by side, each on a checkerboard with a
    pixel grid (darker every 8 px), rulers in sprite coordinates and its number and duration.
    frames: 1-based list, "all" or "2-6". region [x,y,w,h] zooms into a detail. mode: color,
    value (lightness only: do the forms read without hue?) or silhouette. onion=true shows the
    previous frame in red and the next in blue under each frame."""
    _, spr = store.load(path)
    fs = _frames(spr, frames)[:16]
    if mode not in ("color", "value", "silhouette"):
        raise StudioError("mode is color, value or silhouette")
    data = render.view_image(spr, fs, scale, tuple(region) if region else None, mode, grid, rulers, onion, layers)
    return Image(data=data, format="png")


@tool
def read_pixels(path: str, frame: int = 1, layer: str | None = None, region: list[int] | None = None) -> dict:
    """Pixels as a character map: {"legend": {char: colour}, "rows": [...]} ('.' = transparent).
    Edit the rows and send them back with the map op: a read-edit-write round trip. Without
    `layer` it reads the flattened frame. Region capped at 128x128."""
    _, spr = store.load(path)
    f = spr.check_frame(frame)
    img = render.render_frame(spr, f, [layer] if layer else None)
    x, y, w, h = region or (0, 0, spr.width, spr.height)
    if w * h > 128 * 128:
        raise StudioError("region too large; read at most 128x128 at a time")
    full = np.zeros((h, w, 4), np.uint8)
    sub = img[max(0, y) : y + h, max(0, x) : x + w]
    full[max(0, -y) : max(0, -y) + sub.shape[0], max(0, -x) : max(0, -x) + sub.shape[1]] = sub
    symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789#$%&*+=?@^~"
    legend: dict[str, str] = {}
    lut: dict[bytes, str] = {}
    rows = []
    for r in full:
        s = []
        for px in r:
            if px[3] == 0:
                s.append(".")
                continue
            k = px.tobytes()
            if k not in lut:
                if len(lut) >= len(symbols):
                    raise StudioError("more colours than map symbols; read a smaller region")
                lut[k] = symbols[len(lut)]
                legend[lut[k]] = "#" + "".join(f"{v:02x}" for v in (px if px[3] != 255 else px[:3]))
            s.append(lut[k])
        rows.append("".join(s))
    return {"x": x, "y": y, "legend": legend, "rows": rows}


@tool
def lint(path: str, frame: int = 1, layer: str | None = None) -> dict:
    """Find what is easy to miss, with coordinates: orphan pixels (unlike all 8 neighbours),
    lone pixels, detached parts, colour count, darkest colour share and pixels past the canvas.
    Use a layer for one object; the flattened frame touches everything else."""
    _, spr = store.load(path)
    return review.lint(spr, spr.check_frame(frame), layer)


@tool
def check_animation(
    path: str, kind: str = "loop", layer: str | None = None, frames: Any = None, tag: str | None = None
) -> dict:
    """Measure an animation without playing it: per-frame bounds, ground row, centroid, height,
    repeated frames, colour drift and spacing. kind: "loop" (walk, idle: must close), "oneshot"
    (attack, hurt, death: may hold poses) or "scene" (several actors: report only)."""
    _, spr = store.load(path)
    fs = render.frames_range(spr, None, tag) if tag else (_frames(spr, frames) if frames else None)
    if kind not in ("loop", "oneshot", "scene"):
        raise StudioError("kind is loop, oneshot or scene")
    return review.check_animation(spr, kind, layer, fs)


_GLITCH_COLORS = {
    "cut": (230, 20, 120, 255),
    "block": (255, 120, 0, 255),
    "blip": (150, 40, 230, 255),
    "stray": (0, 170, 90, 255),
    "spur": (0, 120, 220, 255),
    "hole": (200, 160, 0, 255),
    "orphan": (90, 90, 110, 255),
}


@tool
def find_glitches(
    path: str,
    layer: str | None = None,
    master: str | None = None,
    frames: Any = None,
    loop: bool = False,
    master_frame: int = 1,
    master_frames: Any = None,
) -> list:
    """Find what animating broke, frame by frame, with an onion-skin view of every frame that
    has a problem (previous frame red, next blue, each problem boxed and labelled).
    Each frame is compared with its previous and next frame (loop=true wraps round) and with
    the master (default: the first frame), so the design's own details are not reported.
    Kinds: cut (a part sliced open: interior colours on the outer edge), block (a flat fill
    larger than the master has), blip (a small patch unlike both neighbour frames), stray
    (a detached speck), spur (a single pixel sticking out), hole (an enclosed gap), orphan
    (a pixel unlike all its neighbours). Run it after every animate/edit pass; fix until ok.
    master_frames: several frames of this sprite that are real drawings (a transition's start and
    end keys); what any of them contains is design."""
    _, spr = store.load(path)
    fs = _frames(spr, frames) if frames else None
    if master_frames:
        m = [render.render_frame(spr, f, [layer] if layer else None) for f in _frames(spr, master_frames)]
    else:
        m = _load_rgba(master, master_frame) if master else None
    res = review.find_glitches(spr, layer, fs, m, loop)
    marks: dict = {}
    for d in res["issues"]:
        marks.setdefault(d["frame"] - 1, []).append(
            (d["x"], d["y"], d["w"], d["h"], d["kind"].upper(), _GLITCH_COLORS[d["kind"]])
        )
    shown = sorted(marks)[:16]
    out: list = [res]
    if shown:
        # zoom on the layer's content over all frames, with a small margin
        boxes = [
            review._frame_stats(render.render_frame(spr, f, [layer] if layer else None)) for f in range(spr.frame_count)
        ]
        boxes = [b["bbox"] for b in boxes if b]
        x0 = max(0, min(b[0] for b in boxes) - 2)
        y0 = max(0, min(b[1] for b in boxes) - 4)
        x1 = min(spr.width, max(b[0] + b[2] for b in boxes) + 2)
        y1 = min(spr.height, max(b[1] + b[3] for b in boxes) + 2)
        data = render.view_image(
            spr,
            shown,
            0,
            (x0, y0, x1 - x0, y1 - y0),
            "color",
            True,
            True,
            True,
            [layer] if layer else None,
            marks=marks,
        )
        out.append(Image(data=data, format="png"))
    return out


@tool
def check_identity(path: str, master: str, layer: str | None = None, frames: Any = None, master_frame: int = 1) -> dict:
    """Compare every frame with a master image (png or .aseprite): share of pixels that are
    master colours (within a just-noticeable difference), colours the master lacks, and
    height/width drift. The check that a character is still itself after animating."""
    _, spr = store.load(path)
    m = _load_rgba(master, master_frame)
    return review.check_identity(spr, m, layer, _frames(spr, frames) if frames else None)


@tool
def compare(
    path: str,
    frame_a: int,
    frame_b: int,
    other_path: str | None = None,
    layer: str | None = None,
    region: list[int] | None = None,
) -> dict:
    """Pixel difference between two frames (of one sprite, or of two files): changed pixel count
    and the box that changed. 0 changed pixels proves a part did not move."""
    _, a = store.load(path)
    b = store.load(other_path)[1] if other_path else a
    ia = render.render_frame(a, a.check_frame(frame_a), [layer] if layer else None)
    ib = render.render_frame(b, b.check_frame(frame_b), [layer] if layer else None)
    if region:
        x, y, w, h = region
        ia, ib = ia[y : y + h, x : x + w], ib[y : y + h, x : x + w]
    return review.diff(ia, ib)


# ---------------------------------------------------------------------------- conversion
@tool
def pixelize(
    image: str,
    output: str,
    cell: float | None = None,
    target_height: int | None = None,
    max_colors: int = 32,
    merge_delta: float = 3.5,
    background: str = "auto",
    outline: str | None = None,
    palette_from: str | None = None,
    layer: str = "sprite",
    png_output: str | None = None,
    overwrite: bool = False,
) -> list:
    """Convert an image into real pixel art and save it as a native-resolution .aseprite.

    Finds the fake pixel grid of image-model "pixel art" (cell size, drifting grid lines) and
    turns each cell into one pixel by perceptual majority; a real illustration (no grid) or
    an explicit target_height is downscaled by majority vote instead. Palette reduced in OKLab
    (small distinct accents kept), specks and orphan pixels cleaned.
    cell: force the cell size in source px. target_height: the native height wanted.
    background: auto (remove a flat background if there is no transparency) | remove | keep.
    outline: none, "auto" (darkest palette colour) or a colour, adds a 1 px keyline.
    palette_from: lock to another image's colours (e.g. the character's master).
    Returns the report and a preview: source | result enlarged."""
    src = _load_rgba(image)
    pal = None
    if palette_from:
        pm = _load_rgba(palette_from)
        pal = [tuple(int(v) for v in c) for c in np.unique(pm[pm[..., 3] > 0][:, :3], axis=0)]
    native, rep = pixelize_image(
        src,
        cell=cell,
        target_height=target_height,
        max_colors=max_colors,
        merge_delta=merge_delta,
        background=background,
        outline=None if outline in (None, "none") else ("auto" if outline == "auto" else parse(outline)),
        palette=pal,
    )
    h, w = native.shape[:2]
    spr = ase_format.new_sprite(w, h, layer)
    spr.layers[0].cels[0] = Cel(native, 0, 0)
    spr.palette = [(0, 0, 0, 0)] + [parse(c) for c in rep["palette"]]
    p = store.replace(output, spr, overwrite)
    if png_output:
        render.save_png(native, store.resolve(png_output))
    rep["output"] = str(p)
    # preview: source (cropped) | native x scale, on a neutral background
    sc = max(1, rep["suggested_preview_scale"])
    ys, xs = np.nonzero(src[..., 3] > 0)
    s_crop = src[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] if xs.size else src
    big = render.upscale(native, sc)
    fit = max(1, int(np.ceil(max(s_crop.shape[0] / 600, big.shape[0] / 600))))
    s_crop, big = s_crop[::fit, ::fit], big[::fit, ::fit]
    H = max(s_crop.shape[0], big.shape[0])
    sheet = np.zeros((H, s_crop.shape[1] + 16 + big.shape[1], 4), np.uint8)
    sheet[: s_crop.shape[0], : s_crop.shape[1]] = s_crop
    sheet[: big.shape[0], s_crop.shape[1] + 16 :] = big
    return [rep, _png(render.on_background(sheet, (58, 62, 78)))]


@tool
def import_strip(
    path: str,
    image: str,
    layer: str = "reference",
    count: int | None = None,
    master: str | None = None,
    cell: float | None = None,
    align: str = "feet",
    start_frame: int = 1,
    palette_lock: bool = True,
    duration_ms: int = 100,
) -> list:
    """Bring a strip of poses (drawn by hand or generated) into a sprite as frames, at the
    master's scale: every frame is pixelized on one shared cell size (taken from the master's
    height unless `cell` is given), locked to the master's colours, and stood on the master's
    ground line with its feet (or centre) on the master's axis. Frames are added as needed.
    count: frames in the strip (default: split at transparent gaps). Returns a preview."""
    strip = _load_rgba(image)
    m = _load_rgba(master) if master else None
    frames_img, rep = animate.pixelize_strip(strip, count, m, cell, palette_lock)
    with store.edit(path) as (_, spr):
        if m is not None:
            mys, mxs = np.nonzero(m[..., 3] > 0)
            mh = int(np.ptp(mys)) + 1
            # master drawn where it sits in this sprite if it is one of its layers; else bottom-centre
            target_bottom = min(spr.height - 1, spr.height - 2)
            band = mys >= mys.max() - max(1, int(mh * 0.15))
            feet_cx = float(mxs[band].mean() - mxs.min())
            target_cx = (spr.width - (np.ptp(mxs) + 1)) / 2 + feet_cx
        else:
            target_bottom, target_cx = spr.height - 2, spr.width / 2
        placed = animate.register_frames(frames_img, target_bottom, target_cx, align)
        try:
            la = spr.image_layer(layer)
        except StudioError:
            la = spr.add_layer(layer)
        f0 = spr.check_frame(start_frame)
        need = f0 + len(placed) - spr.frame_count
        if need > 0:
            spr.insert_frames(spr.frame_count, need, duration_ms)
        for k, (img, dx, dy) in enumerate(placed):
            t = trim(img, dx, dy)
            if t:
                la.cels[f0 + k] = Cel(t[0], t[1], t[2])
        rep.update(
            {
                "layer": la.path,
                "frames_written": [f0 + k + 1 for k in range(len(placed))],
                "sizes": [[int(i.shape[1]), int(i.shape[0])] for i in frames_img],
            }
        )
        data = render.view_image(
            spr, list(range(f0, f0 + min(len(placed), 8))), 0, None, "color", True, True, False, [la.path]
        )
    return [rep, Image(data=data, format="png")]


# ---------------------------------------------------------------------------- animation
_PART_COLORS = [
    (230, 80, 80),
    (80, 170, 240),
    (110, 210, 110),
    (240, 190, 60),
    (190, 110, 230),
    (60, 210, 200),
    (240, 130, 190),
    (150, 150, 150),
    (250, 150, 70),
    (120, 120, 240),
]


def _rig_source(spr, layer, frame):
    # a named layer is read even when hidden (render_frame): animate_rig hides its source
    return render.render_frame(spr, spr.check_frame(frame), [layer] if layer else None)


@tool
def rig_preview(path: str, rig: dict, layer: str | None = None, frame: int = 1) -> list:
    """Check a rig before animating: which pixels each part got (tinted), each pivot (white dot),
    pixel counts and bounding boxes. Parts claim pixels in list order; leftovers form "rest".
    rig = {"parts": [{"name", "mask": {"rect"|"polygon"|"rows"|"colors"|"union"|"minus"}, "pivot": [x,y],
    "parent"?, "z"?}], "overlap"?: 1}. Name parts head, torso, arm_front, arm_back, leg_front,
    leg_back, weapon, tail, hair to use the motion presets."""
    _, spr = store.load(path)
    src = _rig_source(spr, layer, frame)
    r = animate.Rig(src, rig)
    tint = np.zeros_like(src)
    for k, (name, p) in enumerate(r.parts.items()):
        col = _PART_COLORS[k % len(_PART_COLORS)] if name != "rest" else (90, 90, 90)
        m = p["mask"]
        tint[m, :3] = (src[m, :3] * 0.35 + np.array(col) * 0.65).astype(np.uint8)
        tint[m, 3] = 255
    for p in r.parts.values():
        x, y = (int(v) for v in p["pivot"])
        if 0 <= x < tint.shape[1] and 0 <= y < tint.shape[0]:
            tint[y, x] = (255, 255, 255, 255)
    side = np.concatenate([src, np.zeros((src.shape[0], 4, 4), np.uint8), tint], 1)
    sc = max(2, min(8, 480 // max(side.shape[:2])))
    legend = {
        name: to_hex(_PART_COLORS[k % len(_PART_COLORS)]) if name != "rest" else "#5a5a5a"
        for k, name in enumerate(r.parts)
    }
    return [{"parts": r.summary(), "colors": legend}, _png(render.on_background(side, (40, 42, 54)), sc)]


@tool
def animate_rig(
    path: str,
    rig: dict,
    out_layer: str,
    preset: str | None = None,
    keys: list[dict] | None = None,
    frames: int | None = None,
    source_layer: str | None = None,
    source_frame: int = 1,
    start_frame: int = 1,
    ease: str | None = None,
    feet: list[int] | None = None,
    strength: float = 1.0,
    durations: list[int] | None = None,
    clean: bool = True,
    flash_color: str = "#ffffff",
    hide_source: bool = True,
) -> list:
    """Pose a rigged character over several frames and write them to `out_layer`.

    Motion comes from a preset (idle, walk_side, attack, hurt, death; strength scales it) or
    from keys: [{"frame": 1, "pose": {"arm_front": {"rotate": -30}, "torso": {"dy": 1},
    "root": {"dx": 2, "rotate": 5, "pivot": [x,y]}}}, ...] interpolated over `frames`
    (ease: ease_in_out | linear | hold). Parts turn with RotSprite about their pivots, children
    follow parents, and joint gaps are filled from the source (rig "overlap"). feet [x,y] is
    the pivot for whole-body rotations (default bottom centre). clean removes specks and orphan
    pixels the rotation left. Frames are added if needed; the source layer is hidden.
    Returns a summary and a view with onion skin."""
    with store.edit(path) as (_, spr):
        src = _rig_source(spr, source_layer, source_frame)
        r = animate.Rig(src, rig)
        if preset:
            pr = animate.preset(preset, set(r.parts), tuple(feet) if feet else None, strength)
            n = pr["frames"]
            poses = animate.interpolate(pr["keys"], n, ease or pr["ease"])
            durs = durations or pr["durations"]
            flash_frames = pr.get("flash_frames", [])
        else:
            if not keys:
                raise StudioError("give a preset or keys")
            n = frames or max(k["frame"] for k in keys)
            poses = animate.interpolate(keys, n, ease or "ease_in_out")
            durs = durations
            flash_frames = []
        if feet:
            for p in poses:
                if "root" in p and not p["root"].get("pivot"):
                    p["root"]["pivot"] = list(feet)
        try:
            la = spr.image_layer(out_layer)
        except StudioError:
            la = spr.add_layer(out_layer)
        f0 = spr.check_frame(start_frame)
        need = f0 + n - spr.frame_count
        if need > 0:
            spr.insert_frames(spr.frame_count, need, 100)
        from .pixelize import clean_orphans, remove_specks

        stats = {"specks": 0, "orphans": 0}
        for k, pose in enumerate(poses):
            img, x, y = r.render(pose)
            if clean and any(abs(float(q.get("rotate", 0))) > 1e-6 for q in pose.values()):
                # only rotated frames need cleaning; an unrotated pose is the source's own pixels
                stats["specks"] += remove_specks(img, 1)
                stats["orphans"] += clean_orphans(img)
            if k + 1 in flash_frames:
                on = img[..., 3] > 0
                img[on, :3] = parse(flash_color)[:3]
            t = trim(img, x, y)
            if t:
                la.cels[f0 + k] = Cel(*t)
            else:
                la.cels.pop(f0 + k, None)
        if durs:
            for k, d in enumerate(durs[:n]):
                spr.durations[f0 + k] = int(d)
        if hide_source and source_layer and source_layer != out_layer:
            spr.layer(source_layer).visible = False
        data = render.view_image(spr, list(range(f0, f0 + min(n, 16))), 0, None, "color", True, True, True, [la.path])
    return [
        {
            "layer": la.path,
            "frames": list(range(f0 + 1, f0 + n + 1)),
            "durations": spr.durations[f0 : f0 + n],
            "specks_removed": stats["specks"],
            "orphans_fixed": stats["orphans"],
        },
        Image(data=data, format="png"),
    ]


@tool
def tween_poses(
    path: str,
    rig_start: dict,
    rig_end: dict,
    start_layer: str,
    end_layer: str,
    out_layer: str,
    inbetweens: int = 3,
    start_frame: int = 1,
    end_frame: int = 1,
    first_frame: int = 1,
    ease: str = "ease_in_out",
    extra: list[dict] | None = None,
    durations: list[int] | None = None,
    angle_range: float = 180.0,
    clean: bool = True,
) -> list:
    """In-betweens between two drawn key poses of one character (pose-to-pose).

    rig_start / rig_end cut each drawing into parts with the SAME names (same masks syntax as
    rig_preview); put each pivot on the same joint in both drawings (neck, shoulder, hip, grip).
    For every part the rotation and position that carry it from one drawing to the other are
    fitted; in-betweens in the first half move the start drawing's parts, those in the second half
    move the end drawing's parts back, so every frame is made of real drawn pixels.
    Writes out_layer from first_frame: the start drawing, `inbetweens` frames, the end drawing.
    ease: ease_in_out | ease_in | ease_out | linear. extra: one pose per in-between added on top
    (e.g. anticipation [{"torso": {"dy": 3}}, {}, ...]); parts of both rigs, plus "root".
    Returns the fit per part (angle, match 0..1; under 0.4 = shape changes too much: split it at
    its joint, e.g. a leg into thigh + shin, or redraw it), bone_stretch (joints whose limb is
    over 2 px longer or shorter in one drawing: joints stay closed in every in-between, and the
    length changes where the frames switch drawings), and an onion-skin view.
    Parts that carry others turn by their bones (pivot -> children's pivots)."""
    if ease not in ("ease_in_out", "ease_in", "ease_out", "linear"):
        raise StudioError("ease is ease_in_out, ease_in, ease_out or linear")
    if extra is not None and len(extra) != inbetweens:
        raise StudioError(f"extra needs one pose per in-between ({inbetweens})")
    with store.edit(path) as (_, spr):
        a = render.render_frame(spr, spr.check_frame(start_frame), [start_layer], include_hidden=True)
        b = render.render_frame(spr, spr.check_frame(end_frame), [end_layer], include_hidden=True)
        tw = tween.Tween(a, b, rig_start, rig_end, angle_range)
        try:
            la = spr.image_layer(out_layer)
        except StudioError:
            la = spr.add_layer(out_layer)
        n = inbetweens + 2
        f0 = spr.check_frame(first_frame)
        if f0 + n > spr.frame_count:
            spr.insert_frames(spr.frame_count, f0 + n - spr.frame_count, 100)
        from .pixelize import clean_orphans, remove_specks

        drawn_by = []
        for k in range(n):
            if k == 0:
                img, x, y, who = a, 0, 0, "start drawing"
            elif k == n - 1:
                img, x, y, who = b, 0, 0, "end drawing"
            else:
                t = tween.progress(inbetweens, ease)[k - 1]
                img, x, y, who = tw.frame(t, extra[k - 1] if extra else None)
                if clean:
                    remove_specks(img, 1)
                    clean_orphans(img)
                who = f"{who} rig at {t:.2f}"
            drawn_by.append(who)
            t_ = trim(img, x, y)
            if t_:
                la.cels[f0 + k] = Cel(*t_)
            else:
                la.cels.pop(f0 + k, None)
        if durations:
            for k, d in enumerate(durations[:n]):
                spr.durations[f0 + k] = int(d)
        for name in (start_layer, end_layer):
            if name != out_layer:
                spr.layer(name).visible = False
        data = render.view_image(spr, list(range(f0, f0 + min(n, 16))), 0, None, "color", True, True, True, [la.path])
    rep = tw.report()
    rep.update({"layer": la.path, "frames": list(range(f0 + 1, f0 + n + 1)), "drawn_by": drawn_by})
    return [rep, Image(data=data, format="png")]


# ---------------------------------------------------------------------------- export & misc
@tool
def export(
    path: str,
    output: str,
    kind: str | None = None,
    scale: int = 1,
    frames: Any = None,
    tag: str | None = None,
    background: str | None = None,
    layers: list[str] | None = None,
    layout: str = "horizontal",
    columns: int | None = None,
    padding: int = 0,
    data_output: str | None = None,
    loop: bool = True,
    overwrite: bool = False,
) -> dict:
    """Export without Aseprite. kind (default from the extension): "gif" (real transparency or a
    background, global palette, per-frame durations, loop or play once), "png" (one frame),
    "frames" (output must contain {frame}: one png per frame), "sheet" (layout horizontal |
    vertical | grid with columns, padding, optional JSON frame data). Reference layers and
    hidden layers are left out; `layers` restricts to some layers."""
    _, spr = store.load(path)
    fs = render.frames_range(spr, None, tag) if tag else _frames(spr, frames)
    out = store.resolve(output)
    kind = kind or ("gif" if out.suffix.lower() == ".gif" else "frames" if "{frame}" in output else "png")
    if kind != "frames" and out.exists() and not overwrite:
        raise StudioError(f"{out} exists; pass overwrite=true")
    bg = parse(background) if background else None
    if kind == "gif":
        imgs = [render.render_frame(spr, f, layers) for f in fs]
        render.save_gif(imgs, [spr.durations[f] for f in fs], out, scale, bg, loop)
        return {"output": str(out), "frames": len(fs)}
    if kind == "png":
        img = render.render_frame(spr, fs[0], layers)
        render.save_png(render.on_background(img, bg) if bg else img, out, scale)
        return {"output": str(out)}
    if kind == "frames":
        paths = []
        for f in fs:
            q = store.resolve(output.replace("{frame}", str(f + 1)))
            if q.exists() and not overwrite:
                raise StudioError(f"{q} exists; pass overwrite=true")
            img = render.render_frame(spr, f, layers)
            render.save_png(render.on_background(img, bg) if bg else img, q, scale)
            paths.append(str(q))
        return {"outputs": paths}
    if kind == "sheet":
        info = render.save_sheet(
            spr, fs, out, layout, columns, padding, scale, store.resolve(data_output) if data_output else None, layers
        )
        return {"output": str(out), **info}
    raise StudioError("kind is gif, png, frames or sheet")


@tool
def make_ramp(base: str, steps: int = 5, hue_shift: float = 20.0, contrast: float = 0.55) -> dict:
    """A shading ramp around one colour, darkest first, built in OKLCh: even lightness steps,
    hue leaning warm in the lights and cool in the shadows, chroma tapering at both ends."""
    if not 2 <= steps <= 12:
        raise StudioError("steps must be 2..12")
    return {"ramp": ramp(base, steps, hue_shift, contrast)}


@tool
def verify_with_aseprite(path: str) -> dict:
    """Open the file with the real Aseprite (if installed: ASEPRITE_PATH or the usual install
    folder), render every frame there and compare with this server's renderer. Proves the
    file is valid and that previews here match what Aseprite shows."""
    exe = (
        os.environ.get("ASEPRITE_PATH")
        or shutil.which("aseprite")
        or next(
            (
                c
                for c in [
                    r"C:\Program Files\Aseprite\Aseprite.exe",
                    r"C:\Program Files (x86)\Steam\steamapps\common\Aseprite\Aseprite.exe",
                    "/Applications/Aseprite.app/Contents/MacOS/aseprite",
                ]
                if Path(c).exists()
            ),
            None,
        )
    )
    if not exe:
        raise StudioError("Aseprite not found; set ASEPRITE_PATH")
    p, spr = store.load(path)
    with tempfile.TemporaryDirectory() as td:
        pattern = str(Path(td) / "f_{frame}.png")
        r = subprocess.run([exe, "-b", str(p), "--save-as", pattern], capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return {"ok": False, "aseprite_error": (r.stderr or r.stdout)[-800:]}
        outs = sorted(Path(td).glob("f_*.png"), key=lambda q: int("".join(c for c in q.stem if c.isdigit()) or 0))
        if len(outs) != spr.frame_count:
            return {"ok": False, "error": f"Aseprite wrote {len(outs)} frames, expected {spr.frame_count}"}
        bad = []
        for f, q in enumerate(outs):
            ref = np.array(PILImage.open(q).convert("RGBA")).astype(int)
            ref[ref[..., 3] == 0] = 0
            mine = render.render_frame(spr, f).astype(int)
            d = int(np.abs(ref - mine).max())
            if d > 1:
                bad.append(
                    {"frame": f + 1, "max_channel_diff": d, "pixels": int((np.abs(ref - mine).max(-1) > 1).sum())}
                )
    return {"ok": not bad, "frames_checked": spr.frame_count, "mismatches": bad}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
