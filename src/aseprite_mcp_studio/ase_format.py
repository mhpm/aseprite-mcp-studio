"""Read and write .aseprite / .ase files directly, following Aseprite's published file
format specification (docs/ase-file-specs.md in the Aseprite repository).

No Aseprite process is involved, which is what makes every operation in this server take
milliseconds instead of the seconds a headless Aseprite launch costs.

Supported: RGBA, grayscale and indexed sprites (all read into RGBA), image and group layers,
raw / compressed / linked cels, tags, slices (with 9-patch and pivot), palettes, cel opacity
and z-index, text and colour user data. Tilemap layers are read as empty image layers with a
warning, since this server does not edit tilesets. Files are always written as RGBA.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from .document import BLEND_MODES, TAG_DIRECTIONS, Cel, Layer, Slice, SliceKey, Sprite, StudioError, Tag

HEADER_MAGIC = 0xA5E0
FRAME_MAGIC = 0xF1FA

CHUNK_OLD_PALETTE_4 = 0x0004
CHUNK_OLD_PALETTE_11 = 0x0011
CHUNK_LAYER = 0x2004
CHUNK_CEL = 0x2005
CHUNK_CEL_EXTRA = 0x2006
CHUNK_COLOR_PROFILE = 0x2007
CHUNK_TAGS = 0x2018
CHUNK_PALETTE = 0x2019
CHUNK_USER_DATA = 0x2020
CHUNK_SLICE = 0x2022
CHUNK_TILESET = 0x2023


class _Reader:
    def __init__(self, data: bytes, pos: int = 0):
        self.d, self.p = data, pos

    def take(self, fmt: str):
        fmt = "<" + fmt  # little-endian, no alignment padding
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, self.d, self.p)
        self.p += size
        return vals if len(vals) > 1 else vals[0]

    def bytes(self, n: int) -> bytes:
        b = self.d[self.p : self.p + n]
        self.p += n
        return b

    def string(self) -> str:
        n = self.take("H")
        return self.bytes(n).decode("utf-8", "replace")


def _pixels_to_rgba(
    raw: bytes, w: int, h: int, depth: int, palette: np.ndarray, transparent_index: int, opaque_index: bool
) -> np.ndarray:
    if depth == 32:
        return np.frombuffer(raw, np.uint8, w * h * 4).reshape(h, w, 4).copy()
    if depth == 16:
        ga = np.frombuffer(raw, np.uint8, w * h * 2).reshape(h, w, 2)
        out = np.empty((h, w, 4), np.uint8)
        out[..., 0] = out[..., 1] = out[..., 2] = ga[..., 0]
        out[..., 3] = ga[..., 1]
        return out
    if depth == 8:
        idx = np.frombuffer(raw, np.uint8, w * h).reshape(h, w)
        out = palette[idx].copy()
        if not opaque_index:
            out[idx == transparent_index] = 0
        return out
    raise StudioError(f"unsupported colour depth {depth}")


def read(path: str | Path) -> Sprite:
    data = Path(path).read_bytes()
    return read_bytes(data)


def read_bytes(data: bytes) -> Sprite:
    if len(data) < 128:
        raise StudioError("not an Aseprite file (too short)")
    r = _Reader(data)
    (
        _size,
        magic,
        nframes,
        width,
        height,
        depth,
        flags,
        _speed,
        _z1,
        _z2,
        transparent_index,
        _pad,
        ncolors,
        pw,
        ph,
        gx,
        gy,
        gw,
        gh,
    ) = r.take("IHHHHHIHIIB3sHBBhhHH")
    if magic != HEADER_MAGIC:
        raise StudioError("not an Aseprite file (bad magic number)")
    r.p = 128
    spr = Sprite(width=width, height=height, durations=[])
    spr.pixel_ratio = (pw or 1, ph or 1)
    spr.grid = (gx, gy, gw or 16, gh or 16)
    has_uuid = bool(flags & 4)
    palette = np.zeros((256, 4), np.uint8)
    palette[:, 3] = 255
    pal_list: list[tuple[int, int, int, int]] = []
    layer_parents: list[int] = []  # child level per layer, for reconstruction
    tilemap_layers: set[int] = set()
    last_obj: object | None = None
    tag_queue: list[Tag] = []
    cel_links: list[tuple[Layer, int, int]] = []  # (layer, frame, linked-to frame)

    for fi in range(nframes):
        fstart = r.p
        fbytes, fmagic, old_chunks, duration, _res, new_chunks = r.take("IHHH2sI")
        if fmagic != FRAME_MAGIC:
            raise StudioError(f"corrupt frame {fi + 1} (bad magic)")
        spr.durations.append(duration or 100)
        nchunks = new_chunks if (old_chunks == 0xFFFF and new_chunks) else old_chunks
        for _ in range(nchunks):
            cstart = r.p
            csize, ctype = r.take("IH")
            cend = cstart + csize
            if ctype == CHUNK_LAYER:
                lflags, ltype, child, _dw, _dh, blend, opacity, _r3 = r.take("HHHHHHB3s")
                name = r.string()
                if ltype == 2:
                    r.take("I")
                if has_uuid:
                    r.bytes(16)
                la = Layer(
                    name=name,
                    kind="group" if ltype == 1 else "image",
                    visible=bool(lflags & 1),
                    editable=bool(lflags & 2),
                    background=bool(lflags & 8),
                    collapsed=bool(lflags & 32),
                    reference=bool(lflags & 64),
                    opacity=opacity if flags & 1 else 255,
                    blend=BLEND_MODES[blend] if blend < len(BLEND_MODES) else "normal",
                )
                # parent = last layer with child level one less
                parent = None
                for prev, lvl in zip(reversed(spr.layers), reversed(layer_parents), strict=False):
                    if lvl == child - 1 and prev.kind == "group":
                        parent = prev
                        break
                la.parent = parent if child > 0 else None
                if ltype == 2:
                    tilemap_layers.add(len(spr.layers))
                    spr.warnings.append(f"layer {name!r} is a tilemap; it was read as an empty image layer")
                spr.layers.append(la)
                layer_parents.append(child)
                last_obj = la
            elif ctype == CHUNK_CEL:
                li, x, y, opacity, ctyp, z = r.take("HhhBHh")
                r.bytes(5)
                la = spr.layers[li]
                if ctyp == 1:
                    link = r.take("H")
                    cel_links.append((la, fi, link))
                    last_obj = None
                elif ctyp in (0, 2) and li not in tilemap_layers:
                    w, h = r.take("HH")
                    raw = r.bytes(cend - r.p)
                    if ctyp == 2:
                        raw = zlib.decompress(raw)
                    img = _pixels_to_rgba(raw, w, h, depth, palette, transparent_index, la.background)
                    cel = Cel(img, x, y, opacity, z)
                    la.cels[fi] = cel
                    last_obj = cel
                else:
                    last_obj = None
            elif ctype == CHUNK_PALETTE:
                total, first, last = r.take("III")
                r.bytes(8)
                if total > len(pal_list):
                    pal_list.extend([(0, 0, 0, 255)] * (total - len(pal_list)))
                for i in range(first, last + 1):
                    pf, cr, cg, cb, ca = r.take("HBBBB")
                    if pf & 1:
                        r.string()
                    pal_list[i] = (cr, cg, cb, ca)
                    if i < 256:
                        palette[i] = (cr, cg, cb, ca)
                last_obj = spr
            elif ctype in (CHUNK_OLD_PALETTE_4, CHUNK_OLD_PALETTE_11) and not pal_list:
                packets = r.take("H")
                idx = 0
                for _ in range(packets):
                    skip, n = r.take("BB")
                    idx += skip
                    n = n or 256
                    for _ in range(n):
                        cr, cg, cb = r.take("BBB")
                        if ctype == CHUNK_OLD_PALETTE_11:
                            cr, cg, cb = (cr << 2) | (cr >> 4), (cg << 2) | (cg >> 4), (cb << 2) | (cb >> 4)
                        if idx < 256:
                            palette[idx] = (cr, cg, cb, 255)
                        idx += 1
            elif ctype == CHUNK_TAGS:
                ntags = r.take("H")
                r.bytes(8)
                tag_queue = []
                for _ in range(ntags):
                    start, end, direction, repeat = r.take("HHBH")
                    r.bytes(6)
                    cr, cg, cb = r.take("BBB")
                    r.bytes(1)
                    name = r.string()
                    t = Tag(
                        name,
                        start,
                        end,
                        TAG_DIRECTIONS[direction] if direction < 4 else "forward",
                        repeat,
                        (cr, cg, cb, 255),
                    )
                    spr.tags.append(t)
                    tag_queue.append(t)
                last_obj = "tags"
            elif ctype == CHUNK_SLICE:
                nkeys, sflags, _res = r.take("III")
                name = r.string()
                keys = []
                for _ in range(nkeys):
                    kf, kx, ky, kw, kh = r.take("IiiII")
                    center = r.take("iiII") if sflags & 1 else None
                    pivot = r.take("ii") if sflags & 2 else None
                    keys.append(SliceKey(kf, kx, ky, kw, kh, center, pivot))
                s = Slice(name, keys)
                spr.slices.append(s)
                last_obj = s
            elif ctype == CHUNK_USER_DATA:
                uflags = r.take("I")
                text = r.string() if uflags & 1 else ""
                color = r.take("BBBB") if uflags & 2 else None
                target = last_obj
                if target == "tags":
                    target = tag_queue.pop(0) if tag_queue else None
                    if not tag_queue:
                        last_obj = None
                if target is not None and target is not spr and hasattr(target, "user_text"):
                    target.user_text = text
                    if isinstance(target, Tag) and color:
                        target.color = tuple(color)
            r.p = cend
        r.p = fstart + fbytes

    # finish palette and links
    spr.palette = [tuple(int(v) for v in c) for c in pal_list] or []
    for la, fi, link in cel_links:
        src = la.cels.get(link)
        if src is not None:
            la.cels[fi] = src
    if depth != 32:
        spr.warnings.append("indexed/grayscale sprite read as RGBA; it will be saved as RGBA")
    return spr


# ---------------------------------------------------------------------------- writing
def _string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<H", len(b)) + b


def _chunk(ctype: int, payload: bytes) -> bytes:
    return struct.pack("<IH", len(payload) + 6, ctype) + payload


def _user_data(text: str, color: tuple | None = None) -> bytes:
    flags = (1 if text else 0) | (2 if color else 0)
    out = struct.pack("<I", flags)
    if text:
        out += _string(text)
    if color:
        out += struct.pack("<BBBB", *color)
    return _chunk(CHUNK_USER_DATA, out)


def _palette_for(spr: Sprite) -> list[tuple[int, int, int, int]]:
    if spr.palette:
        return list(spr.palette[:256])
    seen: dict[tuple, None] = {}
    for la in spr.layers:
        for c in la.cels.values():
            a = c.image.reshape(-1, 4)
            a = a[a[:, 3] > 0]
            for col in np.unique(a, axis=0)[:256]:
                seen.setdefault(tuple(int(v) for v in col), None)
                if len(seen) >= 255:
                    break
    return [(0, 0, 0, 0)] + list(seen)[:255]


def write(spr: Sprite, path: str | Path) -> int:
    data = write_bytes(spr)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(p)  # atomic on the same volume: a crash never leaves half a file
    return len(data)


def write_bytes(spr: Sprite) -> bytes:
    layer_index = {id(la): i for i, la in enumerate(spr.layers)}
    pal = _palette_for(spr)
    frames = []
    for fi in range(spr.frame_count):
        chunks: list[bytes] = []
        if fi == 0:
            chunks.append(_chunk(CHUNK_COLOR_PROFILE, struct.pack("<HHI8s", 1, 0, 0, b"\0" * 8)))
            body = struct.pack("<III8s", len(pal), 0, len(pal) - 1, b"\0" * 8)
            for c in pal:
                body += struct.pack("<HBBBB", 0, *c)
            chunks.append(_chunk(CHUNK_PALETTE, body))
            for la in spr.layers:
                lflags = (
                    (1 if la.visible else 0)
                    | (2 if la.editable else 0)
                    | (8 if la.background else 0)
                    | (32 if la.collapsed else 0)
                    | (64 if la.reference else 0)
                )
                body = struct.pack(
                    "<HHHHHHB3s",
                    lflags,
                    1 if la.kind == "group" else 0,
                    la.depth,
                    0,
                    0,
                    BLEND_MODES.index(la.blend) if la.blend in BLEND_MODES else 0,
                    max(0, min(255, la.opacity)),
                    b"\0\0\0",
                ) + _string(la.name)
                chunks.append(_chunk(CHUNK_LAYER, body))
                if la.user_text:
                    chunks.append(_user_data(la.user_text))
            if spr.tags:
                body = struct.pack("<H8s", len(spr.tags), b"\0" * 8)
                for t in spr.tags:
                    body += struct.pack(
                        "<HHBH6sBBBB",
                        t.start,
                        t.end,
                        TAG_DIRECTIONS.index(t.direction),
                        t.repeat,
                        b"\0" * 6,
                        *t.color[:3],
                        0,
                    ) + _string(t.name)
                chunks.append(_chunk(CHUNK_TAGS, body))
                for t in spr.tags:
                    chunks.append(_user_data(t.user_text, t.color))
            for s in spr.slices:
                sflags = (1 if any(k.center for k in s.keys) else 0) | (2 if any(k.pivot for k in s.keys) else 0)
                body = struct.pack("<III", len(s.keys), sflags, 0) + _string(s.name)
                for k in s.keys:
                    body += struct.pack("<IiiII", k.frame, k.x, k.y, k.w, k.h)
                    if sflags & 1:
                        body += struct.pack("<iiII", *(k.center or (0, 0, k.w, k.h)))
                    if sflags & 2:
                        body += struct.pack("<ii", *(k.pivot or (0, 0)))
                chunks.append(_chunk(CHUNK_SLICE, body))
                if s.user_text:
                    chunks.append(_user_data(s.user_text))
        for la in spr.layers:
            if la.kind != "image":
                continue
            c = la.cels.get(fi)
            if c is None:
                continue
            li = layer_index[id(la)]
            first = min(f for f, o in la.cels.items() if o is c)
            head = struct.pack(
                "<HhhBHh5s", li, c.x, c.y, max(0, min(255, c.opacity)), 1 if first != fi else 2, c.z, b"\0" * 5
            )
            if first != fi:
                chunks.append(_chunk(CHUNK_CEL, head + struct.pack("<H", first)))
                continue
            h, w = c.image.shape[:2]
            img = np.ascontiguousarray(c.image, dtype=np.uint8)
            chunks.append(_chunk(CHUNK_CEL, head + struct.pack("<HH", w, h) + zlib.compress(img.tobytes(), 6)))
            if c.user_text:
                chunks.append(_user_data(c.user_text))
        body = b"".join(chunks)
        n = len(chunks)
        frames.append(
            struct.pack(
                "<IHHH2sI",
                16 + len(body),
                FRAME_MAGIC,
                min(n, 0xFFFF),
                max(1, min(65535, spr.durations[fi])),
                b"\0\0",
                n,
            )
            + body
        )
    payload = b"".join(frames)
    gx, gy, gw, gh = spr.grid
    header = struct.pack(
        "<IHHHHHIHIIB3sHBBhhHH",
        128 + len(payload),
        HEADER_MAGIC,
        spr.frame_count,
        spr.width,
        spr.height,
        32,
        1 | 2,
        100,
        0,
        0,
        0,
        b"\0\0\0",
        len(pal) if len(pal) < 256 else 0,
        spr.pixel_ratio[0],
        spr.pixel_ratio[1],
        gx,
        gy,
        gw,
        gh,
    )
    header += b"\0" * (128 - len(header))
    return header + payload


def new_sprite(width: int, height: int, layer: str = "Layer 1", frames: int = 1, duration: int = 100) -> Sprite:
    if not (1 <= width <= 8192 and 1 <= height <= 8192):
        raise StudioError("canvas size must be between 1 and 8192 px per side")
    spr = Sprite(width=width, height=height, durations=[duration] * frames)
    spr.layers.append(Layer(name=layer))
    return spr
