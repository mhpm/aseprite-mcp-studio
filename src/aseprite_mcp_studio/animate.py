"""Character animation: cut-out rigs posed with RotSprite, keyframe interpolation, motion
presets, and importing a reference strip registered onto a master.

Rig
---
A rig cuts one source image into parts. Each part has a mask, a pivot (the joint it turns
about, in canvas coordinates of the source), an optional parent and a z order:

    {"parts": [
        {"name": "torso", "mask": {"rect": [10, 20, 12, 10]}, "pivot": [16, 29]},
        {"name": "head",  "mask": {"rect": [8, 4, 18, 16]},  "pivot": [16, 20], "parent": "torso", "z": 2},
        {"name": "arm",   "mask": {"polygon": [[20,22],[26,22],[26,30],[20,30]]}, "pivot": [21, 23],
         "parent": "torso", "z": 3}
     ],
     "overlap": 1}

Masks: {"rect": [x,y,w,h]}, {"polygon": [[x,y],...]}, {"rows": [...], "x": 0, "y": 0} (any
character but '.' selects), {"colors": [...], "region": [x,y,w,h]?}, and combinations
{"union": [m, ...]}, {"minus": [m_keep, m_cut]}. Pixels are claimed by the first part, in
list order, whose mask covers them; whatever no part claims becomes the static part "rest".

Pose
----
{"torso": {"dy": 1}, "arm": {"rotate": -40}, "root": {"dx": 2, "rotate": 5, "pivot": [16, 47]}}
Rotation is clockwise in degrees about the part's pivot and inherited by children; dx/dy move
the part (and its children). "hide": true hides a part.

Joints: rotating a limb opens a gap at the joint. With `overlap` = n, each part also carries
the source pixels within n px around it (taken from its neighbours), drawn *under* every part,
so the gap is filled with the right colours instead of showing a hole. With
"overlap_from": "parent" that padding comes only from the part's parent, for limbs that swing
far (a pose-to-pose tween) and would otherwise carry off pixels of whatever they touched.
"""

from __future__ import annotations

import math

import numpy as np

from .color import nearest, parse
from .document import StudioError
from .pixelize import pixelize
from .transform import rotate


# ---------------------------------------------------------------------------- masks
def _polygon_mask(h: int, w: int, pts: list) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    px, py = xx + 0.5, yy + 0.5
    inside = np.zeros((h, w), bool)
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        x1, y1, x2, y2 = x1 + 0.5, y1 + 0.5, x2 + 0.5, y2 + 0.5
        cond = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2 - x1) * (py - y1) / (y2 - y1) + x1
        inside ^= cond & (px < xint)
    # include the boundary pixels themselves, so a polygon drawn on pixel corners is inclusive
    from .raster import polyline_points

    for x, y in polyline_points([tuple(p) for p in pts], closed=True, perfect=False):
        if 0 <= y < h and 0 <= x < w:
            inside[y, x] = True
    return inside


def build_mask(spec: dict, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    m = np.zeros((h, w), bool)
    if "rect" in spec:
        x, y, rw, rh = spec["rect"]
        m[max(0, y) : max(0, y + rh), max(0, x) : max(0, x + rw)] = True
    elif "polygon" in spec:
        m = _polygon_mask(h, w, spec["polygon"])
    elif "rows" in spec:
        ox, oy = spec.get("x", 0), spec.get("y", 0)
        for j, row in enumerate(spec["rows"]):
            for i, ch in enumerate(row):
                if ch != "." and 0 <= oy + j < h and 0 <= ox + i < w:
                    m[oy + j, ox + i] = True
    elif "colors" in spec:
        cols = [parse(c) for c in spec["colors"]]
        for c in cols:
            m |= (img == np.array(c, np.uint8)).all(-1)
        if "region" in spec:
            x, y, rw, rh = spec["region"]
            r = np.zeros_like(m)
            r[max(0, y) : y + rh, max(0, x) : x + rw] = True
            m &= r
    elif "union" in spec:
        for s in spec["union"]:
            m |= build_mask(s, img)
    elif "minus" in spec:
        keep, cut = spec["minus"]
        m = build_mask(keep, img) & ~build_mask(cut, img)
    else:
        raise StudioError(f"unknown mask {list(spec)}; use rect, polygon, rows, colors, union or minus")
    return m


def _dilate(m: np.ndarray, n: int) -> np.ndarray:
    for _ in range(n):
        p = np.pad(m, 1)
        m = m | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return m


# ---------------------------------------------------------------------------- rig
class Rig:
    def __init__(self, source: np.ndarray, spec: dict):
        self.src = source
        self.overlap = int(spec.get("overlap", 1))
        # "any": the joint padding comes from every neighbour (fine for small turns);
        # "parent": only from the part's parent, so a limb swinging far does not carry off a
        # ring of hair or cloth from whatever it happened to touch
        self.overlap_from = spec.get("overlap_from", "any")
        if self.overlap_from not in ("any", "parent"):
            raise StudioError("overlap_from is any or parent")
        opaque = source[..., 3] > 0
        claimed = np.zeros(opaque.shape, bool)
        self.parts: dict[str, dict] = {}
        names = [p["name"] for p in spec.get("parts", [])]
        if len(set(names)) != len(names):
            raise StudioError("part names must be unique")
        for k, p in enumerate(spec.get("parts", [])):
            m = build_mask(p["mask"], source) & opaque & ~claimed
            claimed |= m
            if p.get("parent") and p["parent"] not in names:
                raise StudioError(f"part {p['name']!r} has unknown parent {p['parent']!r}")
            self.parts[p["name"]] = {
                "mask": m,
                "pivot": tuple(p.get("pivot") or self._center(m)),
                "parent": p.get("parent"),
                "z": p.get("z", k),
            }
        rest = opaque & ~claimed
        if rest.any():
            self.parts["rest"] = {"mask": rest, "pivot": self._center(rest), "parent": None, "z": -1}
        for name, p in self.parts.items():
            if not p["mask"].any():
                raise StudioError(f"part {name!r} selects no opaque pixel; check its mask")
            ring = _dilate(p["mask"], self.overlap) & opaque & ~p["mask"] if self.overlap else None
            if ring is not None and self.overlap_from == "parent":
                ring &= self.parts[p["parent"]]["mask"] if p["parent"] else np.zeros_like(ring)
            p["image"] = self._cut(p["mask"])
            p["pad"] = self._cut(ring) if ring is not None and ring.any() else None
        # cycle check
        for name in self.parts:
            seen, q = set(), name
            while q:
                if q in seen:
                    raise StudioError(f"the parent chain of {name!r} loops")
                seen.add(q)
                q = self.parts[q]["parent"]

    @staticmethod
    def _center(m: np.ndarray) -> tuple[int, int]:
        ys, xs = np.nonzero(m)
        return (int(xs.mean()), int(ys.mean())) if xs.size else (0, 0)

    def _cut(self, m: np.ndarray) -> np.ndarray:
        out = np.zeros_like(self.src)
        out[m] = self.src[m]
        return out

    def summary(self) -> dict:
        out = {}
        for name, p in self.parts.items():
            ys, xs = np.nonzero(p["mask"])
            out[name] = {
                "pixels": int(xs.size),
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                "pivot": list(p["pivot"]),
                "parent": p["parent"],
                "z": p["z"],
            }
        return out

    # ------------------------------------------------------------------ posing
    def _world(self, pose: dict) -> dict[str, tuple[float, float, float]]:
        """Per part: (total angle, world pivot x, world pivot y)."""
        root = pose.get("root", {})
        r_ang = float(root.get("rotate", 0))
        r_piv = root.get("pivot") or [self.src.shape[1] / 2, self.src.shape[0] - 1]
        r_dx, r_dy = float(root.get("dx", 0)), float(root.get("dy", 0))

        def root_tf(x: float, y: float) -> tuple[float, float]:
            return _rot(x, y, r_piv[0], r_piv[1], r_ang)[0] + r_dx, _rot(x, y, r_piv[0], r_piv[1], r_ang)[1] + r_dy

        out: dict[str, tuple[float, float, float]] = {}

        def solve(name: str) -> tuple[float, float, float]:
            if name in out:
                return out[name]
            p = self.parts[name]
            q = pose.get(name, {})
            px, py = p["pivot"]
            if p["parent"]:
                pa, ppx, ppy = solve(p["parent"])
                rest_parent = self.parts[p["parent"]]["pivot"]
                # the parent maps x to world(pp) + Rot(pa)(x - pp), root rotation included in pa
                wx, wy = _rot(px, py, rest_parent[0], rest_parent[1], pa)
                wx += ppx - rest_parent[0]
                wy += ppy - rest_parent[1]
                ang = pa + float(q.get("rotate", 0))
            else:
                wx, wy = root_tf(px, py)
                ang = r_ang + float(q.get("rotate", 0))
            wx += float(q.get("dx", 0))
            wy += float(q.get("dy", 0))
            out[name] = (ang, wx, wy)
            return out[name]

        for n in self.parts:
            solve(n)
        return out

    def render(self, pose: dict, method: str = "rotsprite") -> tuple[np.ndarray, int, int]:
        """Returns (image, x, y): the posed character and its top-left on the canvas."""
        world = self._world(pose)
        layers = []
        order = sorted(self.parts, key=lambda n: self.parts[n]["z"])
        for kind in ("pad", "image"):
            for name in order:
                if pose.get(name, {}).get("hide"):
                    continue
                p = self.parts[name]
                img = p[kind]
                if img is None:
                    continue
                ang, wx, wy = world[name]
                px, py = p["pivot"]
                ys, xs = np.nonzero(img[..., 3] > 0)
                if xs.size == 0:
                    continue
                x0, y0 = xs.min(), ys.min()
                crop = img[y0 : ys.max() + 1, x0 : xs.max() + 1]
                if pose.get(name, {}).get("flip"):
                    crop = crop[:, ::-1]
                if abs(ang) > 1e-6:
                    rimg, dx, dy = rotate(crop, ang, (px - x0, py - y0), method)
                else:
                    rimg, dx, dy = crop, 0, 0
                ox = int(round(wx - px)) + x0 + dx
                oy = int(round(wy - py)) + y0 + dy
                layers.append((rimg, ox, oy))
        if not layers:
            return np.zeros((1, 1, 4), np.uint8), 0, 0
        X0 = min(o[1] for o in layers)
        Y0 = min(o[2] for o in layers)
        X1 = max(o[1] + o[0].shape[1] for o in layers)
        Y1 = max(o[2] + o[0].shape[0] for o in layers)
        canvas = np.zeros((Y1 - Y0, X1 - X0, 4), np.uint8)
        for img, ox, oy in layers:
            h, w = img.shape[:2]
            region = canvas[oy - Y0 : oy - Y0 + h, ox - X0 : ox - X0 + w]
            m = img[..., 3] > 0
            region[m] = img[m]
        return canvas, X0, Y0


def _rot(x: float, y: float, cx: float, cy: float, deg: float) -> tuple[float, float]:
    t = math.radians(deg)
    dx, dy = x - cx, y - cy
    return cx + dx * math.cos(t) - dy * math.sin(t), cy + dx * math.sin(t) + dy * math.cos(t)


# ---------------------------------------------------------------------------- keys
def interpolate(keys: list[dict], frames: int, ease: str = "ease_in_out") -> list[dict]:
    """Keyframes [{"frame": 1, "pose": {...}}, ...] -> one pose per frame. Numeric values are
    interpolated (eased) between keys; booleans hold. Positions round at render time."""
    if not keys:
        return [{} for _ in range(frames)]
    keys = sorted(keys, key=lambda k: k["frame"])
    out = []
    for f in range(1, frames + 1):
        prev = max((k for k in keys if k["frame"] <= f), key=lambda k: k["frame"], default=keys[0])
        nxt = min((k for k in keys if k["frame"] >= f), key=lambda k: k["frame"], default=keys[-1])
        if prev is nxt or prev["frame"] == nxt["frame"]:
            out.append(_copy_pose(prev["pose"]))
            continue
        t = (f - prev["frame"]) / (nxt["frame"] - prev["frame"])
        if ease == "ease_in_out":
            t = t * t * (3 - 2 * t)
        elif ease == "hold":
            t = 0.0
        pose: dict = {}
        for part in set(prev["pose"]) | set(nxt["pose"]):
            a, b = prev["pose"].get(part, {}), nxt["pose"].get(part, {})
            q = {}
            for k in set(a) | set(b):
                va, vb = a.get(k, 0 if k != "pivot" else b.get(k)), b.get(k, 0 if k != "pivot" else a.get(k))
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
                    q[k] = va + (vb - va) * t
                else:
                    q[k] = va if t < 0.5 else vb
            pose[part] = q
        out.append(pose)
    return out


def _copy_pose(p: dict) -> dict:
    return {k: dict(v) for k, v in p.items()}


# ---------------------------------------------------------------------------- presets
def preset(name: str, parts: set[str], feet: tuple[int, int] | None = None, strength: float = 1.0) -> dict:
    """Motion presets as keyframes, written for parts named head, torso, arm_front, arm_back,
    leg_front, leg_back, weapon, tail, hair (any subset; missing parts are skipped).
    Returns {"frames": n, "keys": [...], "durations": [...], "loop": bool}."""
    s = strength

    def P(**kw):  # keep only parts the rig has
        return {k: v for k, v in kw.items() if k in parts or k == "root"}

    root_piv = list(feet) if feet else None
    if name == "idle":
        keys = [
            {
                "frame": 1,
                "pose": P(torso={"dy": 0}, arm_front={"rotate": 0}, arm_back={"rotate": 0}, hair={"rotate": 0}),
            },
            {
                "frame": 2,
                "pose": P(
                    torso={"dy": 0}, arm_front={"rotate": 2 * s}, arm_back={"rotate": -2 * s}, hair={"rotate": 2 * s}
                ),
            },
            {
                "frame": 3,
                "pose": P(
                    torso={"dy": 1},
                    arm_front={"rotate": 4 * s},
                    arm_back={"rotate": -4 * s},
                    hair={"rotate": 3 * s},
                    tail={"rotate": 6 * s},
                ),
            },
            {
                "frame": 4,
                "pose": P(
                    torso={"dy": 1},
                    arm_front={"rotate": 2 * s},
                    arm_back={"rotate": -2 * s},
                    hair={"rotate": 1 * s},
                    tail={"rotate": 3 * s},
                ),
            },
        ]
        return {"frames": 4, "keys": keys, "durations": [220, 160, 220, 160], "loop": True, "ease": "hold"}
    if name == "walk_side":
        a = 24 * s
        seq = [
            (a, -a, 0),
            (a / 2, -a / 2, 1),
            (0, 0, 0),
            (-a / 2, a / 2, 1),
            (-a, a, 0),
            (-a / 2, a / 2, 1),
            (0, 0, 0),
            (a / 2, -a / 2, 1),
        ]
        keys = []
        for i, (lf, lb, bob) in enumerate(seq, 1):
            keys.append(
                {
                    "frame": i,
                    "pose": P(
                        leg_front={"rotate": lf},
                        leg_back={"rotate": lb},
                        torso={"dy": -bob},
                        arm_front={"rotate": -lf * 0.8},
                        arm_back={"rotate": -lb * 0.8},
                        head={"dy": 0},
                        tail={"rotate": lf * 0.4},
                        hair={"rotate": -bob * 4},
                    ),
                }
            )
        return {"frames": 8, "keys": keys, "durations": [100] * 8, "loop": True, "ease": "hold"}
    if name == "attack":
        keys = [
            {"frame": 1, "pose": P(torso={"rotate": 0}, arm_front={"rotate": 0})},
            {
                "frame": 2,
                "pose": P(
                    root={"dx": -1, "pivot": root_piv},
                    torso={"rotate": -6 * s},
                    arm_front={"rotate": -110 * s},
                    head={"rotate": -3 * s},
                ),
            },
            {
                "frame": 3,
                "pose": P(
                    root={"dx": 3, "pivot": root_piv},
                    torso={"rotate": 8 * s},
                    arm_front={"rotate": 75 * s},
                    head={"rotate": 3 * s},
                ),
            },
            {
                "frame": 4,
                "pose": P(root={"dx": 3, "pivot": root_piv}, torso={"rotate": 6 * s}, arm_front={"rotate": 95 * s}),
            },
            {
                "frame": 5,
                "pose": P(root={"dx": 1, "pivot": root_piv}, torso={"rotate": 2 * s}, arm_front={"rotate": 30 * s}),
            },
        ]
        return {"frames": 5, "keys": keys, "durations": [120, 200, 60, 120, 140], "loop": False, "ease": "hold"}
    if name == "hurt":
        keys = [
            {"frame": 1, "pose": P(root={"dx": -2, "rotate": -8 * s, "pivot": root_piv}, head={"rotate": -6 * s})},
            {"frame": 2, "pose": P(root={"dx": -1, "rotate": -4 * s, "pivot": root_piv})},
            {"frame": 3, "pose": P(root={"dx": 0, "rotate": 0, "pivot": root_piv})},
        ]
        return {
            "frames": 3,
            "keys": keys,
            "durations": [80, 120, 160],
            "loop": False,
            "ease": "hold",
            "flash_frames": [1],
        }
    if name == "death":
        keys = [
            {"frame": i + 1, "pose": P(root={"rotate": -ang, "dy": 0, "pivot": root_piv})}
            for i, ang in enumerate([0, 10, 30, 60, 85, 90])
        ]
        return {"frames": 6, "keys": keys, "durations": [100, 90, 80, 70, 120, 400], "loop": False, "ease": "hold"}
    raise StudioError("presets are idle, walk_side, attack, hurt, death")


# ---------------------------------------------------------------------------- reference strips
def split_strip(img: np.ndarray, count: int | None = None, gap: int = 1) -> list[tuple[np.ndarray, int]]:
    """Cut a strip into frames: equal widths when `count` is given, otherwise at runs of
    fully transparent columns at least `gap` wide. Returns (frame image, x offset)."""
    w = img.shape[1]
    if count:
        edges = np.linspace(0, w, count + 1).round().astype(int)
        return [(img[:, a:b], int(a)) for a, b in zip(edges, edges[1:], strict=False)]
    used = (img[..., 3] > 0).any(0)
    out, x = [], 0
    while x < w:
        while x < w and not used[x]:
            x += 1
        start = x
        while x < w and (used[x] or (x + gap <= w and used[x : x + gap].any())):
            x += 1
        if x > start:
            out.append((img[:, start:x], start))
    return out


def register_frames(frames: list[np.ndarray], target_bottom: int, target_cx: float, align: str = "feet") -> list:
    """Place each frame so its lowest row sits on `target_bottom` and its feet (the centre
    of its bottom 15% of rows) or its whole-body centre is on `target_cx`."""
    placed = []
    for f in frames:
        ys, xs = np.nonzero(f[..., 3] > 0)
        if xs.size == 0:
            placed.append((f, 0, 0))
            continue
        bottom = ys.max()
        if align == "feet":
            band = ys >= bottom - max(1, int((ys.max() - ys.min()) * 0.15))
            cx = xs[band].mean()
        elif align == "center":
            cx = (xs.min() + xs.max()) / 2
        else:
            placed.append((f, 0, 0))
            continue
        placed.append((f, int(round(target_cx - cx)), int(target_bottom - bottom)))
    return placed


def pixelize_strip(
    strip: np.ndarray,
    count: int | None,
    master: np.ndarray | None,
    cell: float | None,
    palette_lock: bool = True,
    max_colors: int = 32,
) -> tuple[list[np.ndarray], dict]:
    """Pixelize every frame of a strip on one shared cell size: the master's height decides
    it when no cell is given, so the reference comes out at the master's scale."""
    from .pixelize import remove_background

    strip, _ = remove_background(np.ascontiguousarray(strip, np.uint8))
    parts = split_strip(strip, count)
    if not parts:
        raise StudioError("the strip has no frames (nothing opaque found)")
    if cell is None:
        if master is None:
            raise StudioError("give cell, or a master to take the size from")
        mh = int(np.ptp(np.nonzero(master[..., 3] > 0)[0]) + 1)
        heights = [int(np.ptp(np.nonzero(p[..., 3] > 0)[0]) + 1) for p, _ in parts if (p[..., 3] > 0).any()]
        cell = float(np.median(heights)) / mh
    pal = None
    if master is not None and palette_lock:
        pal = [tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] > 0][:, :3], axis=0)]
    out = []
    for p, _ in parts:
        img, _rep = pixelize(
            p,
            cell=cell if cell >= 2 else None,
            target_height=None if cell >= 2 else max(1, round(p.shape[0] / cell)),
            max_colors=max_colors,
            background="keep",
            palette=pal,
            crop=True,
        )
        out.append(img)
    return out, {"frames": len(out), "cell_px": round(cell, 3), "palette_locked": pal is not None}


def lock_to_palette(img: np.ndarray, palette: np.ndarray) -> np.ndarray:
    on = img[..., 3] > 0
    out = img.copy()
    if on.any():
        out[on, :3] = palette[nearest(img[on][:, :3].astype(float), palette.astype(float))].astype(np.uint8)
    return out
