"""In-memory model of a sprite: layers, frames, cels, tags, slices and palette.

Images are numpy arrays of shape (h, w, 4), dtype uint8, straight (not premultiplied) RGBA.
A cel may extend past the canvas: Aseprite allows it, and keeping those pixels means a part
drawn partly off-canvas and moved back later is still whole.

Cels live on their layer, keyed by 0-based frame index. Linked cels are the *same* Cel
object stored under several frames, which is exactly Aseprite's semantics: editing one edits
all of them. `Sprite.unlink` gives a frame its own copy.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

RGBA = tuple[int, int, int, int]

BLEND_MODES = [
    "normal",
    "multiply",
    "screen",
    "overlay",
    "darken",
    "lighten",
    "color_dodge",
    "color_burn",
    "hard_light",
    "soft_light",
    "difference",
    "exclusion",
    "hue",
    "saturation",
    "color",
    "luminosity",
    "addition",
    "subtract",
    "divide",
]
TAG_DIRECTIONS = ["forward", "reverse", "pingpong", "pingpong_reverse"]


class StudioError(ValueError):
    """An error worth reporting to the caller as-is (bad argument, missing layer, ...)."""


@dataclass
class Cel:
    image: np.ndarray
    x: int = 0
    y: int = 0
    opacity: int = 255
    z: int = 0
    user_text: str = ""

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        h, w = self.image.shape[:2]
        return self.x, self.y, w, h

    def clone(self) -> Cel:
        return Cel(self.image.copy(), self.x, self.y, self.opacity, self.z, self.user_text)


@dataclass
class Layer:
    name: str
    kind: str = "image"  # "image" | "group"
    parent: Layer | None = None
    visible: bool = True
    opacity: int = 255
    blend: str = "normal"
    editable: bool = True
    background: bool = False
    reference: bool = False
    collapsed: bool = False
    user_text: str = ""
    cels: dict[int, Cel] = field(default_factory=dict)

    @property
    def path(self) -> str:
        names = [self.name]
        p = self.parent
        while p is not None:
            names.append(p.name)
            p = p.parent
        return "/".join(reversed(names))

    @property
    def depth(self) -> int:
        d, p = 0, self.parent
        while p is not None:
            d, p = d + 1, p.parent
        return d

    def effectively_visible(self) -> bool:
        p: Layer | None = self
        while p is not None:
            if not p.visible:
                return False
            p = p.parent
        return True


@dataclass
class Tag:
    name: str
    start: int  # 0-based, inclusive
    end: int
    direction: str = "forward"
    repeat: int = 0  # 0 = forever
    color: RGBA = (0, 0, 0, 255)
    user_text: str = ""


@dataclass
class SliceKey:
    frame: int
    x: int
    y: int
    w: int
    h: int
    center: tuple[int, int, int, int] | None = None
    pivot: tuple[int, int] | None = None


@dataclass
class Slice:
    name: str
    keys: list[SliceKey]
    user_text: str = ""


@dataclass
class Sprite:
    width: int
    height: int
    durations: list[int] = field(default_factory=lambda: [100])
    layers: list[Layer] = field(default_factory=list)  # file order: bottom to top, groups before children
    tags: list[Tag] = field(default_factory=list)
    slices: list[Slice] = field(default_factory=list)
    palette: list[RGBA] = field(default_factory=list)
    grid: tuple[int, int, int, int] = (0, 0, 16, 16)
    pixel_ratio: tuple[int, int] = (1, 1)
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ basics
    @property
    def frame_count(self) -> int:
        return len(self.durations)

    def clone(self) -> Sprite:
        return copy.deepcopy(self)

    def check_frame(self, frame: int) -> int:
        """1-based frame from the API -> 0-based index, with a helpful error."""
        if not isinstance(frame, int) or not 1 <= frame <= self.frame_count:
            raise StudioError(f"frame {frame} does not exist; this sprite has frames 1..{self.frame_count}")
        return frame - 1

    def image_layers(self) -> list[Layer]:
        return [la for la in self.layers if la.kind == "image"]

    def layer(self, name: str | int | None) -> Layer:
        """Find a layer by name, group path ('arms/left') or 1-based index among all layers.
        None means the top image layer."""
        if name is None:
            imgs = self.image_layers()
            if not imgs:
                raise StudioError("the sprite has no image layer; add one with add_layer")
            return imgs[-1]
        if isinstance(name, int):
            if not 1 <= name <= len(self.layers):
                raise StudioError(f"layer {name} does not exist; there are {len(self.layers)} layers")
            return self.layers[name - 1]
        hits = [la for la in self.layers if la.path == name] or [la for la in self.layers if la.name == name]
        if not hits:
            raise StudioError(f"no layer named {name!r}; layers are {[la.path for la in self.layers]}")
        if len(hits) > 1:
            raise StudioError(
                f"layer name {name!r} is ambiguous; use the group path, one of {[la.path for la in hits]}"
            )
        return hits[0]

    def image_layer(self, name: str | int | None) -> Layer:
        la = self.layer(name)
        if la.kind != "image":
            raise StudioError(f"layer {la.path!r} is a group; draw on one of its image layers")
        return la

    def children(self, group: Layer | None) -> list[Layer]:
        return [la for la in self.layers if la.parent is group]

    # ------------------------------------------------------------------ layers
    def add_layer(self, name: str, group: str | None = None, kind: str = "image", **props) -> Layer:
        if any(la.name == name and la.parent is (self.layer(group) if group else None) for la in self.layers):
            raise StudioError(f"a layer named {name!r} already exists there")
        parent = self.layer(group) if group else None
        if parent is not None and parent.kind != "group":
            raise StudioError(f"{group!r} is not a group layer")
        la = Layer(name=name, kind=kind, parent=parent, **props)
        if parent is None:
            self.layers.append(la)
        else:  # insert after the last descendant of the group, so it lands on top inside it
            idx = self.layers.index(parent)
            last = idx
            for i in range(idx + 1, len(self.layers)):
                if self._is_descendant(self.layers[i], parent):
                    last = i
            self.layers.insert(last + 1, la)
        return la

    @staticmethod
    def _is_descendant(la: Layer, group: Layer) -> bool:
        p = la.parent
        while p is not None:
            if p is group:
                return True
            p = p.parent
        return False

    def remove_layer(self, name: str) -> None:
        la = self.layer(name)
        doomed = [la] + [x for x in self.layers if self._is_descendant(x, la)]
        self.layers = [x for x in self.layers if x not in doomed]

    def move_layer(self, name: str, to_index: int) -> None:
        """Move a top-level layer (with its children) to a 1-based stack position among
        top-level layers (1 = bottom)."""
        la = self.layer(name)
        block = [la] + [x for x in self.layers if self._is_descendant(x, la)]
        rest = [x for x in self.layers if x not in block]
        siblings = [x for x in rest if x.parent is la.parent]
        to_index = max(1, min(to_index, len(siblings) + 1))
        if to_index > len(siblings):
            anchor = siblings[-1] if siblings else None
            pos = (
                (max(i for i, x in enumerate(rest) if x is anchor or (anchor and self._is_descendant(x, anchor))) + 1)
                if anchor
                else len(rest)
            )
        else:
            pos = rest.index(siblings[to_index - 1])
        self.layers = rest[:pos] + block + rest[pos:]

    # ------------------------------------------------------------------ frames
    def insert_frames(self, at: int, count: int = 1, duration: int = 100) -> None:
        """Insert `count` empty frames before 0-based index `at` (at == frame_count appends)."""
        for la in self.layers:
            la.cels = {(f + count if f >= at else f): c for f, c in la.cels.items()}
        self.durations[at:at] = [duration] * count
        for t in self.tags:
            if t.start >= at:
                t.start += count
            if t.end >= at:
                t.end += count
        for s in self.slices:
            for k in s.keys:
                if k.frame >= at:
                    k.frame += count

    def remove_frame(self, index: int) -> None:
        if self.frame_count == 1:
            raise StudioError("a sprite needs at least one frame")
        for la in self.layers:
            la.cels = {(f - 1 if f > index else f): c for f, c in la.cels.items() if f != index}
        del self.durations[index]
        for t in list(self.tags):
            if t.start > index:
                t.start -= 1
            if t.end >= index:
                t.end -= 1
            if t.end < t.start:
                self.tags.remove(t)
        for s in self.slices:
            s.keys = [k for k in s.keys if k.frame != index or len(s.keys) == 1]
            for k in s.keys:
                if k.frame > index:
                    k.frame -= 1

    def duplicate_frame(self, index: int, linked: bool = False) -> int:
        """Copy frame `index` to index+1. Returns the new 0-based index."""
        self.insert_frames(index + 1, 1, self.durations[index])
        for la in self.layers:
            c = la.cels.get(index)
            if c is not None:
                la.cels[index + 1] = c if linked else c.clone()
        return index + 1

    def move_frame(self, src: int, dst: int) -> None:
        order = list(range(self.frame_count))
        f = order.pop(src)
        order.insert(dst, f)
        self.reorder_frames(order)

    def reorder_frames(self, order: list[int]) -> None:
        """New frame i shows old frame order[i]."""
        inv = {old: new for new, old in enumerate(order)}
        for la in self.layers:
            la.cels = {inv[f]: c for f, c in la.cels.items() if f in inv}
        self.durations = [self.durations[o] for o in order]

    # ------------------------------------------------------------------ cels
    def unlink(self, layer: Layer, frame: int) -> None:
        c = layer.cels.get(frame)
        if c is not None and any(other is c for f, other in layer.cels.items() if f != frame):
            layer.cels[frame] = c.clone()

    def linked_frames(self, layer: Layer, frame: int) -> list[int]:
        c = layer.cels.get(frame)
        return [] if c is None else sorted(f for f, o in layer.cels.items() if o is c and f != frame)

    def canvas_of(self, layer: Layer, frame: int) -> np.ndarray:
        """The cel as a canvas-sized RGBA image (pixels past the canvas are not included)."""
        out = np.zeros((self.height, self.width, 4), np.uint8)
        c = layer.cels.get(frame)
        if c is not None:
            paste(out, c.image, c.x, c.y)
        return out


def paste(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """Copy src into dst at (x, y), clipped, replacing pixels (no blending)."""
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 < x1 and y0 < y1:
        dst[y0:y1, x0:x1] = src[y0 - y : y1 - y, x0 - x : x1 - x]


def trim(image: np.ndarray, x: int, y: int) -> tuple[np.ndarray, int, int] | None:
    """Crop an RGBA image to its opaque pixels. None when it is empty."""
    a = image[..., 3]
    rows = np.flatnonzero(a.any(1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(a.any(0))
    return (image[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy(), x + int(cols[0]), y + int(rows[0]))
