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


# ---------------------------------------------------------------------------- glitches
# Kinds, most serious first. Each is judged against the master, so the design's own spikes,
# gaps and lone highlights are not reported: only what animating introduced.
GLITCH_KINDS = {
    "cut": "a part was sliced: interior colours lie on the outer edge where the master has outline",
    "block": "a flat patch of one colour bigger than any in the master (an unnatural fill)",
    "blip": "a small patch unlike both the previous and the next frame (onion skin): a flicker",
    "stray": "a detached speck of a few pixels",
    "spur": "a single pixel sticking out of the silhouette, a shape the master never has",
    "hole": "an enclosed transparent pixel the master never has",
    "orphan": "a pixel unlike all 8 neighbours, in a colour pairing the master never has",
}


def _n4(on: np.ndarray) -> np.ndarray:
    p = np.pad(on, 1)
    return p[:-2, 1:-1].astype(np.int8) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]


def _n8(on: np.ndarray) -> np.ndarray:
    p = np.pad(on, 1).astype(np.int8)
    h, w = on.shape
    return sum(p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)


def outer_edge(on: np.ndarray) -> np.ndarray:
    """Opaque pixels touching the outside: transparency connected to the image border (an
    enclosed hole is not outside, so the pixels round it are not edge)."""
    labels, _ = components(~on, diagonal=False)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    outside = np.isin(labels, border[border > 0])
    p = np.pad(outside, 1)
    return on & (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])


def _shape_sig(on: np.ndarray) -> np.ndarray:
    """3x3 opacity pattern around every pixel as a 9-bit number."""
    p = np.pad(on, 1)
    h, w = on.shape
    sig = np.zeros((h, w), np.int32)
    bit = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sig |= p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w].astype(np.int32) << bit
            bit += 1
    return sig


def _shape_set(img: np.ndarray, where: np.ndarray, sig: np.ndarray) -> set:
    """(shape, colour) of the selected pixels, rotated/flipped shapes included: a spike in the
    master pointing up is still a design spike after the part turns."""
    out = set()
    p = pack(img)
    ys, xs = np.nonzero(where)
    for y, x in zip(ys, xs, strict=False):
        bits = [(int(sig[y, x]) >> b) & 1 for b in range(9)]
        g = np.array(bits).reshape(3, 3)
        for k in range(4):
            r = np.rot90(g, k)
            for m in (r, r[:, ::-1]):
                out.add((int((m.ravel() * (1 << np.arange(9))).sum()), int(p[y, x])))
    return out


def _orphans(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orphan mask and, per pixel, the most common neighbour colour (packed)."""
    on = _opaque(img)
    p = pack(img)
    h, w = p.shape
    pad = np.pad(p, 1)
    nbs = np.stack(
        [pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx], -1
    )
    orphan = on & ~(nbs == p[..., None]).any(-1) & (_n8(on) == 8)
    major = np.zeros((h, w), np.uint32)
    for y, x in zip(*np.nonzero(orphan), strict=False):
        v, c = np.unique(nbs[y, x], return_counts=True)
        major[y, x] = v[c.argmax()]
    return orphan, major


def _max_squares(img: np.ndarray) -> tuple[dict, dict]:
    """Per colour, the side of the largest solid square and where one is (x, y)."""
    p = pack(img)
    on = _opaque(img)
    h, w = p.shape
    dp = np.zeros((h, w), np.int32)
    best: dict = {}
    where: dict = {}
    for y in range(h):
        for x in range(w):
            if not on[y, x]:
                continue
            c = p[y, x]
            if y and x and p[y - 1, x] == c and p[y, x - 1] == c and p[y - 1, x - 1] == c:
                dp[y, x] = 1 + min(dp[y - 1, x], dp[y, x - 1], dp[y - 1, x - 1])
            else:
                dp[y, x] = 1
            k = int(c)
            if dp[y, x] > best.get(k, 0):
                best[k] = int(dp[y, x])
                where[k] = (int(x - dp[y, x] + 1), int(y - dp[y, x] + 1))
    return best, where


def _patches(p: np.ndarray) -> np.ndarray:
    """The 3x3 neighbourhood of every pixel, packed colours, shape (h, w, 9)."""
    h, w = p.shape
    pad = np.pad(p, 1)
    return np.stack([pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], -1)


def _explained(p: np.ndarray, q: np.ndarray, radius: int = 2, agree: int = 8) -> np.ndarray:
    """True where the 3x3 patch of p matches (at least `agree` of 9 pixels, centre included)
    the patch of q at some offset within `radius`. Transparent areas count as explained."""
    pp, qq = _patches(p), _patches(q)
    h, w = p.shape
    out = p == 0
    qpad = np.pad(qq, ((radius, radius), (radius, radius), (0, 0)))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sh = qpad[radius + dy : radius + dy + h, radius + dx : radius + dx + w]
            eq = pp == sh
            out |= eq[..., 4] & (eq.sum(-1) >= agree)
    return out


def _boxes(mask: np.ndarray, kind: str, frame: int, note=None, max_items: int = 40) -> list[dict]:
    labels, n = components(mask, diagonal=True)
    out = []
    for lab in range(1, n + 1):
        ys, xs = np.nonzero(labels == lab)
        d = {
            "frame": frame,
            "kind": kind,
            "x": int(xs.min()),
            "y": int(ys.min()),
            "w": int(np.ptp(xs)) + 1,
            "h": int(np.ptp(ys)) + 1,
            "pixels": int(xs.size),
        }
        if note:
            d["note"] = note(ys, xs)
        out.append(d)
    return out[:max_items]


class _MasterModel:
    """What the design itself contains, so it is not reported as a glitch."""

    def __init__(self, m: np.ndarray):
        on = _opaque(m)
        p = pack(m)
        sig = _shape_sig(on)
        self.spurs = _shape_set(m, on & (_n4(on) <= 1), sig)
        holes = ~on & (_n4(on) >= 3)
        self.holes = {s for s, _ in _shape_set(m, holes, sig)}
        orphan, major = _orphans(m)
        self.orphan_pairs = {(int(p[y, x]), int(major[y, x])) for y, x in zip(*np.nonzero(orphan), strict=False)}
        edge = outer_edge(on)
        cols, total = np.unique(p[on], return_counts=True)
        on_edge = dict(zip(*np.unique(p[edge], return_counts=True), strict=False))
        # colours that (nearly) never touch the outside in the master: interior colours
        self.interior = {int(c) for c, t in zip(cols, total, strict=False) if t >= 6 and on_edge.get(c, 0) / t < 0.04}
        self.squares, _ = _max_squares(m)
        labels, n = components(on, diagonal=True)
        sizes = np.bincount(labels.ravel())[1:] if n else np.array([])
        self.speck_sizes = {int(s) for s in sizes if s <= 4}


def find_glitches(
    spr: Sprite,
    layer: str | None = None,
    frames: list[int] | None = None,
    master: np.ndarray | None = None,
    loop: bool = False,
) -> dict:
    fs = frames or list(range(spr.frame_count))
    imgs = {f: render_frame(spr, f, [layer] if layer else None) for f in range(spr.frame_count)}
    model = _MasterModel(master if master is not None else imgs[fs[0]])
    issues: list[dict] = []
    for f in fs:
        img = imgs[f]
        on = _opaque(img)
        if not on.any():
            continue
        p = pack(img)
        sig = _shape_sig(on)
        hex_at = lambda ys, xs, img=img: to_hex(img[ys[0], xs[0]])  # noqa: E731
        # cut: interior colours exposed on the silhouette edge
        edge = outer_edge(on)
        cut = edge & np.isin(p, list(model.interior)) if model.interior else np.zeros_like(on)
        issues += [b for b in _boxes(cut, "cut", f + 1, hex_at) if b["pixels"] >= 2]
        # block: solid squares larger than the master ever has in that colour
        best, where = _max_squares(img)
        for c, side in best.items():
            if side >= 3 and side >= model.squares.get(c, 0) + 2:
                x, y = where[c]
                col = to_hex(np.array([c], np.uint32).view(np.uint8))
                issues.append(
                    {
                        "frame": f + 1,
                        "kind": "block",
                        "x": x,
                        "y": y,
                        "w": side,
                        "h": side,
                        "pixels": side * side,
                        "note": f"{col} {side}x{side}, master max {model.squares.get(c, 0)}",
                    }
                )
        # blip: onion skin. Every frame is compared with its previous and next frame (only the one
        # that exists at the ends, wrapping round for a loop). Pixels that differ from all of them
        # in a small isolated patch are a flicker; a moving part changes as one large patch.
        nbr = [g for g in (f - 1, f + 1) if 0 <= g < spr.frame_count]
        if loop and spr.frame_count > 2:
            nbr = [(f - 1) % spr.frame_count, (f + 1) % spr.frame_count]
        new_here = np.zeros_like(on)
        if nbr:
            # a pixel is explained when its 3x3 surroundings appear in a neighbour frame within
            # 2 px (centre included): motion compensation, so a part that moved or breathed is not
            # a flicker. Patches are sized with a strict match (8 of 9: a moving part stays one
            # large patch) and kept only if even a loose match (6 of 9, a part that turned a
            # little) fails somewhere in them.
            new_here = np.ones_like(on)
            loose = np.ones_like(on)
            for g in nbr:
                q = pack(imgs[g])
                new_here &= ~_explained(p, q, radius=2, agree=8)
                loose &= ~_explained(p, q, radius=2, agree=6)
            labels, n = components(new_here, diagonal=True)
            for lab in range(1, n + 1):
                m = labels == lab
                if m.sum() <= 6 and (m & loose).any():
                    b = _boxes(m, "blip", f + 1)[0]
                    b["note"] = "unlike frames " + " and ".join(str(g + 1) for g in nbr)
                    issues.append(b)
        # stray: small detached pieces the master does not have
        labels, n = components(on, diagonal=True)
        if n > 1:
            sizes = np.bincount(labels.ravel())
            for lab in range(1, n + 1):
                if sizes[lab] <= 4 and int(sizes[lab]) not in model.speck_sizes:
                    issues += _boxes(labels == lab, "stray", f + 1)
        # spur / hole / orphan: shapes and pairings the master never has
        spur = on & (_n4(on) <= 1)
        bad = np.zeros_like(on)
        for y, x in zip(*np.nonzero(spur), strict=False):
            if (int(sig[y, x]), int(p[y, x])) not in model.spurs:
                bad[y, x] = True
        issues += _boxes(bad, "spur", f + 1)
        hole = ~on & (_n4(on) >= 3)
        bad = np.zeros_like(on)
        for y, x in zip(*np.nonzero(hole), strict=False):
            if int(sig[y, x]) not in model.holes:
                bad[y, x] = True
        issues += _boxes(bad, "hole", f + 1)
        # orphans that persist across frames are texture; only new ones (unlike the neighbour
        # frames too) are noise that animating added
        orphan, major = _orphans(img)
        if nbr:
            orphan &= new_here
        bad = np.zeros_like(on)
        for y, x in zip(*np.nonzero(orphan), strict=False):
            if (int(p[y, x]), int(major[y, x])) not in model.orphan_pairs:
                bad[y, x] = True
        issues += _boxes(bad, "orphan", f + 1)
    order = list(GLITCH_KINDS)
    issues.sort(key=lambda d: (d["frame"], order.index(d["kind"])))
    per_frame: dict = {}
    for d in issues:
        per_frame.setdefault(d["frame"], {}).setdefault(d["kind"], 0)
        per_frame[d["frame"]][d["kind"]] += 1
    return {
        "ok": not issues,
        "total": len(issues),
        "per_frame": per_frame,
        "issues": issues,
        "kinds": GLITCH_KINDS,
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
