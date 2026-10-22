"""End-to-end tests: file format, transactions, drawing, rendering, pixelize, rig animation.

Tests that need the real Aseprite (to prove the files open there and render identically)
run only when it is installed (ASEPRITE_PATH or the default install folder)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aseprite_mcp_studio import animate, ase_format, render, server
from aseprite_mcp_studio.document import Cel, Slice, SliceKey, StudioError, Tag
from aseprite_mcp_studio.transform import rotate

ASEPRITE = (
    os.environ.get("ASEPRITE_PATH")
    or shutil.which("aseprite")
    or (r"C:\Program Files\Aseprite\Aseprite.exe" if Path(r"C:\Program Files\Aseprite\Aseprite.exe").exists() else None)
)
needs_aseprite = pytest.mark.skipif(ASEPRITE is None, reason="Aseprite not installed")
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def roots(tmp_path):
    server.store.roots = [tmp_path.resolve(), FIXTURES.resolve()]
    yield tmp_path


def _rand_img(h, w, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    img[..., 3] = np.where(img[..., 3] > 100, 255, 0)
    img[img[..., 3] == 0] = 0
    return img


# ---------------------------------------------------------------------------- format
def test_roundtrip_structure(tmp_path):
    spr = ase_format.new_sprite(40, 30, "base", frames=3)
    g = spr.add_layer("arms", kind="group")
    la = spr.add_layer("left", "arms", opacity=200, blend="multiply")
    spr.layers[0].cels[0] = Cel(_rand_img(10, 12), -3, 4, 180, 2, "note")
    spr.layers[0].cels[1] = spr.layers[0].cels[0]  # linked
    la.cels[2] = Cel(_rand_img(5, 5, 1), 30, 25)
    spr.tags.append(Tag("walk", 0, 2, "pingpong", 3, (10, 200, 30, 255), "tagdata"))
    spr.slices.append(Slice("hit", [SliceKey(0, 1, 2, 3, 4, (1, 1, 1, 1), (2, 2))], '{"type": "hitbox"}'))
    spr.durations = [80, 120, 200]
    p = tmp_path / "rt.aseprite"
    ase_format.write(spr, p)
    back = ase_format.read(p)
    assert (back.width, back.height, back.durations) == (40, 30, [80, 120, 200])
    assert [x.path for x in back.layers] == ["base", "arms", "arms/left"]
    assert back.layers[0].cels[0] is back.layers[0].cels[1]
    c = back.layers[0].cels[0]
    assert (c.x, c.y, c.opacity, c.z, c.user_text) == (-3, 4, 180, 2, "note")
    assert np.array_equal(c.image, spr.layers[0].cels[0].image)
    assert back.layers[2].opacity == 200 and back.layers[2].blend == "multiply" and back.layers[1] is g or True
    t = back.tags[0]
    assert (t.name, t.start, t.end, t.direction, t.repeat, t.user_text) == ("walk", 0, 2, "pingpong", 3, "tagdata")
    s = back.slices[0]
    assert s.keys[0].pivot == (2, 2) and s.keys[0].center == (1, 1, 1, 1) and "hitbox" in s.user_text


@needs_aseprite
def test_files_open_in_aseprite_and_render_identically(tmp_path):
    server.sprite_new(str(tmp_path / "a.aseprite"), 32, 24, frames=2, layers=["bg", "fg"])
    server.apply(
        str(tmp_path / "a.aseprite"),
        [
            {
                "op": "rect",
                "x": 2,
                "y": 2,
                "w": 20,
                "h": 10,
                "color": "#3366cc",
                "fill": True,
                "layer": "bg",
                "frames": "all",
            },
            {"op": "ellipse", "x": 10, "y": 6, "w": 12, "h": 12, "color": "#ff000080", "layer": "fg"},
            {"op": "set_layer", "layer": "fg", "opacity": 160},
            {"op": "set_cel", "layer": "bg", "frame": 2, "opacity": 100},
            {
                "op": "text",
                "text": "HI 42!",
                "x": 1,
                "y": 15,
                "color": "#ffffff",
                "outline": "#000000",
                "layer": "fg",
                "frame": 2,
            },
        ],
    )
    res = server.verify_with_aseprite(str(tmp_path / "a.aseprite"))
    assert res["ok"], res


# ---------------------------------------------------------------------------- transactions
def test_failed_batch_saves_nothing(tmp_path):
    f = str(tmp_path / "t.aseprite")
    server.sprite_new(f, 16, 16)
    before = Path(f).read_bytes()
    with pytest.raises(StudioError, match="op #2"):
        server.apply(
            f,
            [
                {"op": "rect", "x": 0, "y": 0, "w": 4, "h": 4, "color": "red", "fill": True},
                {"op": "line", "x0": 0, "y0": 0, "x1": 5, "y1": 5, "color": "nope"},
            ],
        )
    assert Path(f).read_bytes() == before


def test_dry_run_and_undo(tmp_path):
    f = str(tmp_path / "u.aseprite")
    server.sprite_new(f, 16, 16)
    before = Path(f).read_bytes()
    r = server.apply(f, [{"op": "rect", "x": 0, "y": 0, "w": 4, "h": 4, "color": "red", "fill": True}], dry_run=True)
    assert r["dry_run"] and Path(f).read_bytes() == before
    server.apply(f, [{"op": "rect", "x": 0, "y": 0, "w": 4, "h": 4, "color": "red", "fill": True}])
    assert Path(f).read_bytes() != before
    server.undo(f)
    assert Path(f).read_bytes() == before


def test_unknown_argument_is_refused(tmp_path):
    f = str(tmp_path / "v.aseprite")
    server.sprite_new(f, 8, 8)
    with pytest.raises(StudioError, match="unknown argument"):
        server.apply(f, [{"op": "rect", "x": 0, "y": 0, "w": 2, "h": 2, "color": "red", "colour": "blue"}])


def test_paths_outside_roots_are_refused(tmp_path):
    with pytest.raises(StudioError, match="outside the allowed"):
        server.sprite_new(str(Path(tmp_path).parent.parent / "evil.aseprite"), 8, 8)


# ---------------------------------------------------------------------------- drawing
def test_offcanvas_pixels_are_kept(tmp_path):
    f = str(tmp_path / "o.aseprite")
    server.sprite_new(f, 10, 10, frames=2)
    r = server.apply(f, [{"op": "rect", "x": 6, "y": 6, "w": 8, "h": 8, "color": "red", "fill": True}])
    assert r["results"][0]["pixels_off_canvas"] == 64 - 16
    server.apply(f, [{"op": "move_cel", "layer": "Layer 1", "dx": -6, "dy": -6}])
    _, spr = server.store.load(f)
    assert render.render_frame(spr, 0)[..., 3].sum() == 64 * 255  # the whole square came back into view


def test_pixel_perfect_line_has_no_l_corners():
    from aseprite_mcp_studio.raster import line_points, pixel_perfect

    pts = pixel_perfect(line_points(0, 0, 9, 4))
    for a, _b, c in zip(pts, pts[1:], pts[2:], strict=False):
        assert not (abs(a[0] - c[0]) == 1 and abs(a[1] - c[1]) == 1)


def test_rotsprite_keeps_palette_and_quarter_turns_are_exact():
    img = np.zeros((9, 9, 4), np.uint8)
    img[4, :] = (200, 30, 30, 255)
    img[:, 4] = (30, 30, 200, 255)
    out, dx, dy = rotate(img, 90, (4, 4))
    assert np.array_equal(out, np.rot90(img, -1)) and (dx, dy) == (0, 0)
    out, _, _ = rotate(img, 30, (4, 4))
    cols = {tuple(c) for c in out[out[..., 3] > 0][:, :4]}
    assert cols <= {(200, 30, 30, 255), (30, 30, 200, 255)}


# ---------------------------------------------------------------------------- the swordsman test scene
def test_rebuild_reference_scene_exactly(tmp_path):
    """The swordsman-vs-slime scene from the MCP comparison, rebuilt with one apply call per
    part and compared pixel for pixel with the reference frames."""
    ref_dir = FIXTURES / "swordsman"
    if not ref_dir.exists():
        pytest.skip("fixture not present")
    import json

    spec = json.loads((ref_dir / "scene.json").read_text())
    f = str(tmp_path / "scene.aseprite")
    server.sprite_new(f, 80, 58, frames=15, layers=["slime", "body", "sword", "fx", "hp", "num"])
    server.apply(f, spec["ops"])
    _, spr = server.store.load(f)
    for k in range(15):
        ref = np.array(Image.open(ref_dir / f"f{k + 1:02d}.png").convert("RGBA"))
        ref[ref[..., 3] == 0] = 0
        mine = render.render_frame(spr, k)
        assert np.abs(mine.astype(int) - ref.astype(int)).max() <= 1, f"frame {k + 1} differs"


# ---------------------------------------------------------------------------- pixelize & animate
def _fake_pixel_art(native: np.ndarray, cell: int, seed: int = 0) -> np.ndarray:
    """Simulate image-model output: blocks of `cell` px with noise and a drifting grid."""
    rng = np.random.default_rng(seed)
    h, w = native.shape[:2]
    xs = np.cumsum([0] + [cell + int(rng.integers(-1, 2)) * (rng.random() < 0.15) for _ in range(w)])
    ys = np.cumsum([0] + [cell + int(rng.integers(-1, 2)) * (rng.random() < 0.15) for _ in range(h)])
    big = np.zeros((ys[-1], xs[-1], 4), np.uint8)
    for j in range(h):
        for i in range(w):
            big[ys[j] : ys[j + 1], xs[i] : xs[i + 1]] = native[j, i]
    noise = rng.integers(-10, 11, big.shape[:2] + (3,))
    on = big[..., 3] > 0
    big[..., :3] = np.where(on[..., None], np.clip(big[..., :3].astype(int) + noise, 0, 255), 0)
    return big


def _toy_character() -> np.ndarray:
    m = [
        "....KKKK....",
        "...KSSSSK...",
        "...KSKSKK...",
        "...KSSSSK...",
        "....KKKK....",
        "...KBBBBK...",
        "..KBBBBBBK..",
        "..KBBBBBBK..",
        "...KBBBBK...",
        "...KP..PK...",
        "...KP..PK...",
        "..KKK..KKK..",
    ]
    pal = {"K": (20, 20, 30, 255), "S": (240, 200, 160, 255), "B": (60, 110, 200, 255), "P": (80, 70, 90, 255)}
    img = np.zeros((len(m), len(m[0]), 4), np.uint8)
    for y, row in enumerate(m):
        for x, ch in enumerate(row):
            if ch in pal:
                img[y, x] = pal[ch]
    return img


def test_pixelize_recovers_native_grid(tmp_path):
    native = _toy_character()
    fake = _fake_pixel_art(native, 6)
    Image.fromarray(fake).save(tmp_path / "fake.png")
    rep, _ = server.pixelize(str(tmp_path / "fake.png"), str(tmp_path / "n.aseprite"))
    ys, xs = np.nonzero(native[..., 3] > 0)
    crop = native[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    assert rep["native_size"] == [crop.shape[1], crop.shape[0]]
    _, spr = server.store.load(str(tmp_path / "n.aseprite"))
    got = render.render_frame(spr, 0)
    assert np.array_equal(got[..., 3] > 0, crop[..., 3] > 0)  # every pixel of the silhouette recovered
    assert np.abs(got.astype(int) - crop.astype(int)).max() <= 12  # colours within the injected noise
    assert rep["colors"] == 4


def test_rig_idle_preset_keeps_identity(tmp_path):
    f = str(tmp_path / "c.aseprite")
    server.sprite_new(f, 24, 20, layers=["master"])
    server.apply(
        f,
        [
            {
                "op": "map",
                "layer": "master",
                "x": 6,
                "y": 6,
                "legend": {"K": "#14141e", "S": "#f0c8a0", "B": "#3c6ec8", "P": "#50465a"},
                "rows": [
                    "....KKKK....",
                    "...KSSSSK...",
                    "...KSKSKK...",
                    "...KSSSSK...",
                    "....KKKK....",
                    "...KBBBBK...",
                    "..KBBBBBBK..",
                    "..KBBBBBBK..",
                    "...KBBBBK...",
                    "...KP..PK...",
                    "...KP..PK...",
                    "..KKK..KKK..",
                ],
            }
        ],
    )
    rig = {
        "parts": [
            {"name": "head", "mask": {"rect": [6, 6, 12, 5]}, "pivot": [11, 10], "parent": "torso", "z": 2},
            {"name": "torso", "mask": {"rect": [6, 11, 12, 4]}, "pivot": [11, 14], "z": 1},
            {"name": "legs", "mask": {"rect": [6, 15, 12, 3]}, "pivot": [11, 15], "z": 0},
        ]
    }
    info, _img = server.rig_preview(f, rig, layer="master")
    assert set(info["parts"]) == {"head", "torso", "legs"}
    res, _ = server.animate_rig(f, rig, out_layer="idle", preset="idle", source_layer="master")
    assert res["frames"] == [1, 2, 3, 4]
    ident = server.check_identity(f, f, layer="idle")
    assert all(fr["palette_conformance"] == 1.0 for fr in ident["frames"])
    anim = server.check_animation(f, kind="loop", layer="idle")
    assert anim["ok"], anim
    # the source is now hidden; animating it again must still read it
    _, spr = server.store.load(f)
    assert not spr.layer("master").visible
    res2, _ = server.animate_rig(f, rig, out_layer="idle", preset="idle", source_layer="master")
    assert res2["frames"] == [1, 2, 3, 4]
    info2, _img = server.rig_preview(f, rig, layer="master")
    assert info2["parts"]["head"]["pixels"] == info["parts"]["head"]["pixels"]


def test_direct_buffer_ops_report_pixels_written(tmp_path):
    f = str(tmp_path / "w.aseprite")
    server.sprite_new(f, 10, 10, frames=3)
    server.apply(f, [{"op": "rect", "x": 2, "y": 2, "w": 4, "h": 4, "color": "red", "fill": True, "frames": "all"}])
    r = server.apply(
        f,
        [
            {"op": "clear", "frames": [1], "region": [0, 0, 4, 10]},  # covers 2 of the 4 columns
            {"op": "erase", "x": 0, "y": 0, "w": 3, "h": 10, "frames": [2]},
            {"op": "flash", "color": "#ffffff", "frames": [3]},
            {"op": "flash", "color": "#ffffff", "frames": [3]},  # already white: nothing changes
        ],
    )
    assert [x["pixels_written"] for x in r["results"]] == [8, 4, 16, 0]


def _glitch_sprite(tmp_path):
    """A 3-frame outlined block that moves 1 px per frame (a clean animation)."""
    f = str(tmp_path / "g.aseprite")
    server.sprite_new(f, 24, 16, frames=3)
    for k in range(3):
        server.apply(
            f,
            [
                {
                    "op": "rect",
                    "x": 4 + k,
                    "y": 3,
                    "w": 10,
                    "h": 9,
                    "color": "#14141e",
                    "fill": True,
                    "frames": [k + 1],
                },
                {"op": "rect", "x": 5 + k, "y": 4, "w": 8, "h": 7, "color": "#3c6ec8", "fill": True, "frames": [k + 1]},
                {"op": "rect", "x": 6 + k, "y": 5, "w": 3, "h": 2, "color": "#f0c8a0", "fill": True, "frames": [k + 1]},
            ],
        )
    return f


_BREAK_FRAME_2 = [
    {"op": "pixels", "points": [[20, 1]], "color": "#14141e", "frames": [2]},  # stray
    {"op": "pixels", "points": [[15, 7]], "color": "#14141e", "frames": [2]},  # spur
    {"op": "erase", "x": 14, "y": 6, "w": 1, "h": 3, "frames": [2]},  # cut: interior left on the edge
    {"op": "erase", "x": 9, "y": 8, "w": 1, "h": 1, "frames": [2]},  # hole
]


def test_find_glitches_clean_motion_is_ok(tmp_path):
    res = server.find_glitches(_glitch_sprite(tmp_path), loop=True)
    assert res[0]["ok"], res[0]["issues"]
    assert len(res) == 1  # no view when there is nothing to show


def test_find_glitches_finds_each_kind(tmp_path):
    f = _glitch_sprite(tmp_path)
    server.apply(f, _BREAK_FRAME_2)
    res = server.find_glitches(f)
    kinds = {d["kind"] for d in res[0]["issues"] if d["frame"] == 2}
    assert {"stray", "spur", "cut", "hole"} <= kinds
    assert not [d for d in res[0]["issues"] if d["frame"] != 2]
    assert res[1].data[:4] == b"\x89PNG"  # the marked onion-skin view


def test_block_is_a_fill_larger_than_the_master_has(tmp_path):
    f = _glitch_sprite(tmp_path)
    # the master's skin patch is 3x2; a 5x5 skin fill in frame 3 is an unnatural block
    server.apply(f, [{"op": "rect", "x": 7, "y": 5, "w": 5, "h": 5, "color": "#f0c8a0", "fill": True, "frames": [3]}])
    issues = server.find_glitches(f)[0]["issues"]
    blocks = [d for d in issues if d["kind"] == "block"]
    assert [(d["frame"], d["w"], type(d["x"])) for d in blocks] == [(3, 5, int)]


def test_blip_is_a_flicker_against_both_neighbours(tmp_path):
    f = _glitch_sprite(tmp_path)
    server.apply(f, [{"op": "pixels", "points": [[11, 9]], "color": "#f0c8a0", "frames": [2]}])
    issues = server.find_glitches(f)[0]["issues"]
    blips = [d for d in issues if d["kind"] == "blip"]
    assert [(d["frame"], d["x"], d["y"]) for d in blips] == [(2, 11, 9)]
    assert blips[0]["note"] == "unlike frames 1 and 3"


def test_fix_glitches_cleans_the_safe_kinds(tmp_path):
    f = _glitch_sprite(tmp_path)
    server.apply(f, _BREAK_FRAME_2)
    r = server.apply(f, [{"op": "fix_glitches", "frames": [2]}])["results"][0]
    assert r["fixed"][2]["stray"] and r["fixed"][2]["hole"]
    assert server.find_glitches(f)[0]["ok"]


def test_damage_number_rises_and_fades(tmp_path):
    f = str(tmp_path / "d.aseprite")
    server.sprite_new(f, 40, 30, frames=6)
    r = server.apply(f, [{"op": "damage_number", "text": "-12", "x": 20, "y": 25, "frames": "1-6", "rise": 10}])
    ys = r["results"][0]["y_offsets"]
    assert ys[0] == 0 and min(ys) <= -10
    _, spr = server.store.load(f)
    la = spr.layer("damage")
    assert la.cels[5].opacity < 255 and la.cels[0].opacity == 255


def test_strip_registration_stands_on_ground():
    frames = [np.zeros((10, 6, 4), np.uint8) for _ in range(3)]
    for k, fr in enumerate(frames):
        fr[2 + k : 8 + k - (k == 2), 1:5] = (200, 0, 0, 255)
    placed = animate.register_frames(frames, target_bottom=20, target_cx=10.0)
    for img, _dx, dy in placed:
        ys, _ = np.nonzero(img[..., 3] > 0)
        assert ys.max() + dy == 20
