"""In-betweens between two drawn key poses (pose-to-pose animation).

Given a start drawing and an end drawing of the same character, each cut into the same named
parts (a rig per drawing, same part names, pivots on the same joints), the frames between them
are made like this:

1. Fit: for every part, the rotation and position that carry the start drawing's part onto
   the end drawing's part are searched (pixel and colour agreement), and the other way round.
   The result is a pose of the start rig that reaches the end, and a pose of the end rig that
   goes back to the start.
2. Two-way render: in-betweens in the first half move the START drawing's parts toward the
   end; those in the second half move the END drawing's parts back from the start. Every
   frame is made of the pixels of the nearest real drawing, so nothing is invented and each
   half stays faithful to its key.
3. Timing: eased progress (slow out of the start, slow into the end) plus optional extra
   poses per in-between (anticipation, overshoot, follow-through), added on top.

A part whose shape changes a lot between the keys (foreshortening, a hand that opens) has a low
match score; it is reported so it can be drawn by hand.
"""

from __future__ import annotations

import numpy as np

from .animate import Rig
from .document import StudioError
from .transform import rotate


def _crop(img: np.ndarray) -> tuple[np.ndarray, int, int] | None:
    ys, xs = np.nonzero(img[..., 3] > 0)
    if xs.size == 0:
        return None
    return img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], int(xs.min()), int(ys.min())


def _score(placed: np.ndarray, ox: int, oy: int, target: np.ndarray) -> int:
    """Agreement of an image placed at (ox, oy) with a canvas-sized target: matching colours
    count 2, overlapping opaque pixels of other colours 1, opaque on one side only -1."""
    h, w = placed.shape[:2]
    H, W = target.shape[:2]
    x0, y0, x1, y1 = max(0, ox), max(0, oy), min(W, ox + w), min(H, oy + h)
    pa = placed[..., 3] > 0
    total_p = int(pa.sum())
    total_t = int((target[..., 3] > 0).sum())
    if x0 >= x1 or y0 >= y1:
        return -(total_p + total_t)
    sub = placed[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    tg = target[y0:y1, x0:x1]
    a, b = sub[..., 3] > 0, tg[..., 3] > 0
    both = a & b
    same = both & (sub[..., :3] == tg[..., :3]).all(-1)
    return int(2 * same.sum() + (both & ~same).sum() - (total_p - both.sum()) - (total_t - both.sum()))


def fit_part(
    src: np.ndarray,
    pivot: tuple[float, float],
    target: np.ndarray,
    target_pivot: tuple[float, float],
    angle_range: float = 180.0,
    shift: int = 3,
) -> dict:
    """Rotation (clockwise degrees about `pivot`) and pivot position on the target canvas that
    best lay the part image `src` (canvas sized) over `target` (canvas sized)."""
    c = _crop(src)
    if c is None or not (target[..., 3] > 0).any():
        raise StudioError("a part is empty in one of the two drawings")
    crop, x0, y0 = c
    px, py = pivot[0] - x0, pivot[1] - y0
    tx0, ty0 = target_pivot

    def best_at(angle: float, method: str) -> tuple[int, float, float]:
        if abs(angle) > 1e-6:
            r, dx, dy = rotate(crop, angle, (px, py), method)
        else:
            r, dx, dy = crop, 0, 0
        best = (-(10**9), 0.0, 0.0)
        for sy in range(-shift, shift + 1):
            for sx in range(-shift, shift + 1):
                ox = int(round(tx0 + sx - pivot[0])) + x0 + dx
                oy = int(round(ty0 + sy - pivot[1])) + y0 + dy
                s = _score(r, ox, oy, target)
                if s > best[0]:
                    best = (s, tx0 + sx, ty0 + sy)
        return best

    coarse = [a for a in np.arange(-angle_range, angle_range + 1e-9, 6.0)]
    scored = [(best_at(a, "nearest"), a) for a in coarse]
    (_, _, _), a0 = max(scored, key=lambda t: t[0][0])
    fine = [(best_at(a, "nearest"), a) for a in np.arange(a0 - 5, a0 + 5.01, 1.0)]
    (s, wx, wy), ang = max(fine, key=lambda t: t[0][0])
    s, wx, wy = best_at(ang, "rotsprite")
    n_target = int((target[..., 3] > 0).sum())
    return {
        "angle": round(float((ang + 180) % 360 - 180), 1),
        "pivot": [round(wx, 1), round(wy, 1)],
        "match": round(max(0.0, s) / (2 * n_target), 2) if n_target else 0.0,
    }


def _part_image(rig: Rig, name: str) -> np.ndarray:
    return rig.parts[name]["image"]


def pose_reaching(rig: Rig, world: dict[str, dict]) -> dict:
    """The pose of `rig` that puts each part at a world angle and pivot position."""
    pose: dict = {name: {} for name in rig.parts}
    order: list[str] = []

    def visit(n: str) -> None:
        if n in order:
            return
        par = rig.parts[n]["parent"]
        if par:
            visit(par)
        order.append(n)

    for n in rig.parts:
        visit(n)
    for n in order:
        if n not in world:
            continue
        par = rig.parts[n]["parent"]
        parent_angle = world[par]["angle"] if par and par in world else 0.0
        pose[n]["rotate"] = world[n]["angle"] - parent_angle
        ang, wx, wy = rig._world(pose)[n]
        pose[n]["dx"] = world[n]["pivot"][0] - wx
        pose[n]["dy"] = world[n]["pivot"][1] - wy
    return {n: q for n, q in pose.items() if q}


def _scaled(pose: dict, t: float) -> dict:
    return {n: {k: v * t for k, v in q.items() if isinstance(v, (int, float))} for n, q in pose.items()}


def _added(a: dict, b: dict | None) -> dict:
    out = {n: dict(q) for n, q in a.items()}
    for n, q in (b or {}).items():
        d = out.setdefault(n, {})
        for k, v in q.items():
            d[k] = d.get(k, 0) + v if isinstance(v, (int, float)) and not isinstance(v, bool) else v
    return out


def ease(t: float, kind: str) -> float:
    if kind == "linear":
        return t
    if kind == "ease_in":
        return t * t
    if kind == "ease_out":
        return 1 - (1 - t) * (1 - t)
    return t * t * (3 - 2 * t)  # ease_in_out


class Tween:
    def __init__(self, start: np.ndarray, end: np.ndarray, rig_start: dict, rig_end: dict, angle_range: float = 180.0):
        names_a = [p["name"] for p in rig_start.get("parts", [])]
        names_b = [p["name"] for p in rig_end.get("parts", [])]
        if sorted(names_a) != sorted(names_b):
            raise StudioError(f"both rigs need the same part names: {sorted(names_a)} vs {sorted(names_b)}")
        self.a = Rig(start, rig_start)
        self.b = Rig(end, rig_end)
        self.fit: dict[str, dict] = {}
        world_a_to_b: dict[str, dict] = {}
        world_b_to_a: dict[str, dict] = {}
        for n in names_a:
            pa, pb = self.a.parts[n], self.b.parts[n]
            f = fit_part(_part_image(self.a, n), pa["pivot"], _part_image(self.b, n), pb["pivot"], angle_range)
            g = fit_part(_part_image(self.b, n), pb["pivot"], _part_image(self.a, n), pa["pivot"], angle_range)
            bone = self._bone_angle(n)
            if bone is not None:
                # a part that carries others turns so that its children's joints land where the
                # other drawing has them (shoulder -> grip): anatomy beats pixel agreement when
                # the part is foreshortened, and keeps the hand on the sword's grip
                f["angle"], g["angle"] = bone, -bone
                f["pivot"], g["pivot"] = list(pb["pivot"]), list(pa["pivot"])
            self.fit[n] = {
                "angle": f["angle"],
                "match": f["match"],
                "back_angle": g["angle"],
                "back_match": g["match"],
                "by": "bones" if bone is not None else "pixels",
            }
            world_a_to_b[n] = {"angle": f["angle"], "pivot": f["pivot"]}
            world_b_to_a[n] = {"angle": g["angle"], "pivot": g["pivot"]}
        self.pose_a = pose_reaching(self.a, world_a_to_b)  # start rig, posed as the end
        self.pose_b = pose_reaching(self.b, world_b_to_a)  # end rig, posed as the start
        # How far each child sits from where its parent's turn carries it (a limb that is longer
        # or shorter in the other drawing). Moving the child there would open the joint in the
        # in-betweens, so joints stay closed: each half keeps the bone lengths of its own drawing.
        self.stretch = {}
        for rig, pose in ((self.a, self.pose_a), (self.b, self.pose_b)):
            for n, q in pose.items():
                if rig.parts[n]["parent"]:
                    gap = float(np.hypot(q.pop("dx", 0.0), q.pop("dy", 0.0)))
                    self.stretch[n] = round(max(self.stretch.get(n, 0.0), gap), 1)

    def _bone_angle(self, name: str) -> float | None:
        """Turn of a part from its bones to its children's pivots (start drawing -> end
        drawing), a length-weighted circular mean; None for a part without children."""
        kids = [k for k, p in self.a.parts.items() if p["parent"] == name and k in self.b.parts]
        if not kids:
            return None
        ax, ay = self.a.parts[name]["pivot"]
        bx, by = self.b.parts[name]["pivot"]
        sx = sy = 0.0
        for k in kids:
            ka, kb = self.a.parts[k]["pivot"], self.b.parts[k]["pivot"]
            va = (ka[0] - ax, ka[1] - ay)
            vb = (kb[0] - bx, kb[1] - by)
            la, lb = np.hypot(*va), np.hypot(*vb)
            if la < 1 or lb < 1:
                continue
            turn = np.arctan2(vb[1], vb[0]) - np.arctan2(va[1], va[0])
            w = min(la, lb)
            sx += w * np.cos(turn)
            sy += w * np.sin(turn)
        if sx == 0 and sy == 0:
            return None
        return round(float(np.degrees(np.arctan2(sy, sx))), 1)

    def frame(self, t: float, extra: dict | None = None) -> tuple[np.ndarray, int, int, str]:
        """In-between at progress t (0 = start, 1 = end): (image, x, y, which key drew it)."""
        if t < 0.5:
            img, x, y = self.a.render(_added(_scaled(self.pose_a, t), extra))
            return img, x, y, "start"
        img, x, y = self.b.render(_added(_scaled(self.pose_b, 1 - t), extra))
        return img, x, y, "end"

    def report(self) -> dict:
        weak = sorted(n for n, f in self.fit.items() if min(f["match"], f["back_match"]) < 0.4)
        # joints whose bone length differs by more than 2 px between the drawings: closed in every
        # in-between, but the limb changes length where the frames switch from one drawing to the other
        stretched = {n: g for n, g in self.stretch.items() if g > 2}
        return {"parts": self.fit, "weak_parts": weak, "bone_stretch": stretched}


def progress(n: int, kind: str = "ease_in_out") -> list[float]:
    """Eased progress of n in-betweens strictly between the two keys."""
    return [ease(k / (n + 1), kind) for k in range(1, n + 1)]
