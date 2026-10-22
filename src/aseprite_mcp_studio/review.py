"""Measurements that catch what eyes miss, each with exact coordinates or frame numbers.

* lint: stray pixels, detached bits, palette size, darkest-colour share, content past the
  canvas edge.
* check_animation: per-frame bounds, ground contact row, centroid, height; repeated frames;
  colour drift from frame 1. What counts as a problem depends on `kind`: a "loop" must not
  repeat its first frame at the end; a "oneshot" (attack, hurt, death) and a "scene" may hold
  poses, so holds there are information, not errors.
* check_identity: every frame against a master: palette conformance (share of pixels within
  a just-noticeable difference of a master colour), colours the master does not have, and
  height / width drift. This is the "did the character stay itself" check.
"""

from __future__ import annotations

import numpy as np

from .color import rgb_to_oklab, to_hex
from .document import Sprite
from .raster import components, pack
from .render import render_frame


def _opaque(img: np.ndarray) -> np.ndarray:
    return img[..., 3] > 0


def lint_image(img: np.ndarray, offset=(0, 0), max_list: int = 40) -> dict:
    ox, oy = offset
    on = _opaque(img)
    if not on.any():
        return {"empty": True}
    p = pack(img)
    h, w = p.shape
    pad = np.pad(p, 1)
    apad = np.pad(on, 1)
    same = np.zeros((h, w), bool)
    any_nb = np.zeros((h, w), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            same |= pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] == p
            any_nb |= apad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    orphans = on & ~same & any_nb
    lone = on & ~any_nb
    labels, n = components(on, diagonal=True)
    sizes = np.bincount(labels.ravel())[1:]
    main = int(sizes.argmax()) + 1 if n else 0
    detached = []
    for lab in range(1, n + 1):
        if lab == main:
            continue
        ys, xs = np.nonzero(labels == lab)
        detached.append(
            {
                "pixels": int(xs.size),
                "bbox": [int(xs.min()) + ox, int(ys.min()) + oy, int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1],
            }
        )
    detached.sort(key=lambda d: d["pixels"])
    cols, counts = np.unique(img[on][:, :3], axis=0, return_counts=True)
    lum = rgb_to_oklab(cols.astype(float))[:, 0]
    darkest = cols[lum.argmin()]
    ys, xs = np.nonzero(orphans)
    return {
        "colors": int(len(cols)),
        "orphan_pixels": int(orphans.sum()),
        "orphans": [
            {"x": int(x) + ox, "y": int(y) + oy, "color": to_hex(img[y, x])}
            for y, x in list(zip(ys, xs, strict=False))[:max_list]
        ],
        "lone_pixels": [[int(x) + ox, int(y) + oy] for y, x in list(zip(*np.nonzero(lone), strict=False))[:max_list]],
        "detached_parts": detached[:max_list],
        "pieces": int(n),
        "darkest": to_hex(darkest),
        "darkest_share": round(float(counts[lum.argmin()] / counts.sum()), 3),
    }


def lint(spr: Sprite, frame: int, layer: str | None = None) -> dict:
    if layer:
        la = spr.image_layer(layer)
        c = la.cels.get(frame)
        if c is None:
            return {"empty": True}
        res = lint_image(c.image, (c.x, c.y))
        x, y, w, h = c.bounds
        off = 0
        if x < 0 or y < 0 or x + w > spr.width or y + h > spr.height:
            inside = np.zeros(c.image.shape[:2], bool)
            inside[max(0, -y) : min(h, spr.height - y), max(0, -x) : min(w, spr.width - x)] = True
            off = int((_opaque(c.image) & ~inside).sum())
        res["pixels_off_canvas"] = off
        return res
    return lint_image(render_frame(spr, frame))


def _frame_stats(img: np.ndarray) -> dict | None:
    on = _opaque(img)
    if not on.any():
        return None
    ys, xs = np.nonzero(on)
    return {
        "bbox": [int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1],
        "bottom": int(ys.max()),
        "centroid": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
        "height": int(np.ptp(ys)) + 1,
    }


def check_animation(spr: Sprite, kind: str = "loop", layer: str | None = None, frames: list[int] | None = None) -> dict:
    fs = frames or list(range(spr.frame_count))
    imgs = [render_frame(spr, f, [layer] if layer else None) for f in fs]
    stats = [_frame_stats(i) for i in imgs]
    keys = [pack(i).tobytes() for i in imgs]
    errors, warnings, notes = [], [], []
    holds = [[fs[i] + 1, fs[i + 1] + 1] for i in range(len(fs) - 1) if keys[i] == keys[i + 1]]
    if holds:
        msg = f"identical consecutive frames {holds}: a longer duration on one frame does the same with fewer frames"
        (warnings if kind == "loop" else notes).append(msg)
    if kind == "loop" and len(fs) > 1 and keys[0] == keys[-1]:
        errors.append(f"frame {fs[-1] + 1} equals frame {fs[0] + 1}: the loop shows that pose twice at the wrap")
    empties = [fs[i] + 1 for i, s in enumerate(stats) if s is None]
    if empties:
        (errors if kind != "scene" else notes).append(f"empty frames {empties}")
    good = [s for s in stats if s]
    if good:
        bottoms = [s["bottom"] for s in good]
        if max(bottoms) - min(bottoms) > 0 and kind in ("loop",):
            warnings.append(
                f"the ground contact row moves between {min(bottoms)} and {max(bottoms)}; feet should "
                "stay on one line unless the character leaves the ground"
            )
        heights = [s["height"] for s in good]
        h0 = heights[0]
        drift = [fs[i] + 1 for i, s in enumerate(stats) if s and abs(s["height"] - h0) > max(2, h0 * 0.08)]
        if drift and kind == "loop":
            warnings.append(f"frames {drift} are more than 8% taller/shorter than frame {fs[0] + 1}")
    base_cols = {tuple(c) for c in np.unique(imgs[0][_opaque(imgs[0])][:, :3], axis=0)} if good else set()
    new_cols = {}
    for i, im in enumerate(imgs[1:], 1):
        cols = {tuple(c) for c in np.unique(im[_opaque(im)][:, :3], axis=0)} - base_cols
        if cols:
            new_cols[fs[i] + 1] = len(cols)
    if new_cols and kind == "loop":
        warnings.append(f"colours not present in frame {fs[0] + 1} appear in frames {new_cols} (count per frame)")
    spacing = []
    for a, b in zip(good, good[1:], strict=False):
        spacing.append(
            round(float(np.hypot(b["centroid"][0] - a["centroid"][0], b["centroid"][1] - a["centroid"][1])), 2)
        )
    return {
        "kind": kind,
        "frames": [f + 1 for f in fs],
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
        "per_frame": stats,
        "durations": [spr.durations[f] for f in fs],
        "centroid_spacing": spacing,
        "new_colors_vs_first": new_cols,
    }


def check_identity(
    spr: Sprite, master: np.ndarray, layer: str | None = None, frames: list[int] | None = None, jnd: float = 2.5
) -> dict:
    mon = _opaque(master)
    mcols = np.unique(master[mon][:, :3], axis=0).astype(float)
    mlab = rgb_to_oklab(mcols)
    mys, mxs = np.nonzero(mon)
    mh, mw = int(np.ptp(mys)) + 1, int(np.ptp(mxs)) + 1
    fs = frames or list(range(spr.frame_count))
    out = []
    for f in fs:
        img = render_frame(spr, f, [layer] if layer else None)
        on = _opaque(img)
        if not on.any():
            out.append({"frame": f + 1, "empty": True})
            continue
        cols, counts = np.unique(img[on][:, :3], axis=0, return_counts=True)
        d = np.linalg.norm(rgb_to_oklab(cols.astype(float))[:, None] - mlab[None], axis=-1).min(1) * 100
        off = d > jnd
        ys, xs = np.nonzero(on)
        h, w = int(np.ptp(ys)) + 1, int(np.ptp(xs)) + 1
        foreign = sorted(zip(counts[off], cols[off].tolist(), strict=False), reverse=True)[:6]
        out.append(
            {
                "frame": f + 1,
                "palette_conformance": round(float(counts[~off].sum() / counts.sum()), 3),
                "foreign_colors": [{"color": to_hex(c), "pixels": int(n)} for n, c in foreign],
                "height": h,
                "height_vs_master": round(h / mh, 3),
                "width": w,
            }
        )
    worst = min((o for o in out if "palette_conformance" in o), key=lambda o: o["palette_conformance"], default=None)
    return {
        "master_size": [mw, mh],
        "master_colors": int(len(mcols)),
        "frames": out,
        "worst_frame": worst["frame"] if worst else None,
        "verdict": (
            "palette holds"
            if worst is None or worst["palette_conformance"] >= 0.98
            else f"frame {worst['frame']} drifts: {worst['palette_conformance']:.0%} of its pixels are master colours"
        ),
    }


def diff(a: np.ndarray, b: np.ndarray) -> dict:
    H, W = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    pa = np.zeros((H, W, 4), np.uint8)
    pb = np.zeros((H, W, 4), np.uint8)
    pa[: a.shape[0], : a.shape[1]] = a
    pb[: b.shape[0], : b.shape[1]] = b
    pa[pa[..., 3] == 0] = 0
    pb[pb[..., 3] == 0] = 0
    ch = (pa != pb).any(-1)
    if not ch.any():
        return {"changed_pixels": 0}
    ys, xs = np.nonzero(ch)
    return {
        "changed_pixels": int(ch.sum()),
        "percent": round(100 * float(ch.mean()), 2),
        "bbox": [int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1],
    }
