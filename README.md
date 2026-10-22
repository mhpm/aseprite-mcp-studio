# aseprite-mcp-studio

A fast, pure-Python [MCP](https://modelcontextprotocol.io) server for pixel art and sprite
animation in `.aseprite` files. It reads and writes the Aseprite format itself, so an edit takes
milliseconds instead of the seconds a headless Aseprite launch costs, and Aseprite does not
even need to be installed (it is only used, if present, to double-check files).

What it is for:

- **Turning any character image into real pixel art.** It finds the fake pixel grid that image
  models produce (blocks of ~4–14 px, drifting grid lines, thousands of colours) and rebuilds
  the sprite at its true resolution with a small perceptual palette.
- **Animating that character correctly.** It cuts the character into a rig of parts with
  pivots, then poses them with RotSprite rotation, keyframes and motion presets (idle,
  walk_side, attack, hurt, death). Joint gaps are filled automatically. It can also import a
  reference strip of poses at the master's scale and palette, standing on its ground line.
- **Drawing and editing** with a transactional batch of operations, plus review tools that
  report problems with exact coordinates.

## Highlights

| | |
|---|---|
| Speed | Native `.aseprite` reader/writer, an in-memory cache and one write per transaction. No process launch per call. |
| Transactions | `apply` runs any number of ops atomically: all succeed or nothing is saved. `dry_run` validates without saving. Undo/redo up to 50 steps per file. |
| Exact | The compositor matches Aseprite's own render pixel for pixel (verified in the tests against the real Aseprite). Pixels drawn past the canvas stay in the cel instead of being clipped. |
| Pixelize | FFT autocorrelation for the cell size, dynamic-programming grid fitting that follows drift, perceptual-mode cell sampling (no muddy averages), agglomerative OKLab palette reduction that keeps small deliberate accents, background removal, speck and orphan cleanup that never eats eyes or highlights. |
| Animation | Cut-out rigs (rect / polygon / character-map / colour masks), parent chains, RotSprite rotation (colour-safe, line-safe), joint overlap fill, keyframe interpolation, presets, hit flash. |
| Text | Built-in bitmap fonts (`tiny` 3x5 with A–Z and punctuation, `bold` 5x7 numbers) with outline, shadow and two-tone fill, crisp at 1x. One op makes a popping, rising, fading damage number. |
| Review | `view` (many frames, grid, rulers, onion skin, value and silhouette modes), `lint` (orphan, lone and detached pixels with coordinates), `check_animation` (aware of loop / one-shot / scene), `check_identity` (palette and proportion drift against a master), `find_glitches` (each frame against its previous and next frame and the master, with a marked onion-skin view), `compare`. |
| Safe | Paths are confined to allowed roots. Writes are atomic (temp file + rename), so a crash never leaves half a file. |

## Install

```bash
uv tool install git+https://github.com/mhpm/aseprite-mcp-studio
```

or from a clone:

```bash
uv sync
```

Add it to your MCP client (Claude Code `.mcp.json`, Claude Desktop config, ...):

```json
{
  "mcpServers": {
    "aseprite-studio": {
      "command": "uv",
      "args": ["--directory", "/path/to/aseprite-mcp-studio", "run", "aseprite-mcp-studio"],
      "env": { "ASE_STUDIO_ROOTS": "/path/to/your/game/assets" }
    }
  }
}
```

Environment:

- `ASE_STUDIO_ROOTS`: folders the server may read and write, separated by `;` on Windows and `:` elsewhere. Default: the working directory.
- `ASEPRITE_PATH`: optional, only for `verify_with_aseprite`.

## Tools

| Tool | Does |
|---|---|
| `sprite_new` | New RGBA sprite with layers and frames. |
| `sprite_info` | Layers (and linked cels), frames and durations, tags, slices. |
| `apply` | Batch of edit ops as one transaction (see below). |
| `undo` / `redo` | Per-file history. |
| `view` | Look at frames: grid, rulers, onion skin, colour / value / silhouette. |
| `read_pixels` | Pixels as a character map, ready to edit and send back with `map`. |
| `lint` | Orphan, lone and detached pixels with coordinates; colour stats. |
| `check_animation` | Bounds, ground row, holds, drift, spacing; `loop` / `oneshot` / `scene`. |
| `check_identity` | Each frame against a master: palette conformance and size drift. |
| `find_glitches` | What animating broke, per frame: cuts, flat blocks, one-frame flicker (against the previous and next frame, motion compensated), strays, spurs, holes, orphans; the design's own details (judged against the master) are not reported. Returns an onion-skin view with each problem boxed. |
| `compare` | Pixel diff between frames or files. |
| `pixelize` | Image to native pixel art, with a side-by-side preview. |
| `import_strip` | A strip of poses brought in at the master's scale, palette and ground. |
| `rig_preview` | Shows which pixels each rig part takes, and where each pivot is. |
| `animate_rig` | Poses a rig over frames from a preset or from keyframes. |
| `tween_poses` | In-betweens between two drawn key poses: fits each part's turn (bones for parts that carry others, pixels for the rest) and draws each in-between from the nearest real drawing; anticipation and overshoot via `extra`. |
| `export` | GIF, PNG, one PNG per frame, or a sprite sheet with JSON metadata. |
| `make_ramp` | Hue-shifted shading ramp in OKLCh. |
| `verify_with_aseprite` | Renders the file in the real Aseprite and compares. |
| `live_open` | Opens Studio Live, the built-in viewer (see below), and returns its URL. |
| `review_notes` | What the user marked in Studio Live: frames, painted pixels, tags, text, with a picture of the previous / marked / next frame. |
| `resolve_note` | Answers a note (the user sees the reply) and marks it resolved or reopens it. |

### `apply` ops

Layers: `add_layer`, `add_group`, `remove_layer`, `set_layer`, `move_layer`, `duplicate_layer`.
Frames: `add_frames`, `remove_frames`, `duplicate_frame`, `move_frame`, `set_duration`, `set_durations`, `reverse_frames`.
Tags and slices: `add_tag`, `remove_tag`, `add_slice`, `remove_slice`.
Cels: `copy_cel`, `move_cel`, `set_cel`, `clear`, `link_cels`, `unlink_cels`.
Drawing: `pixels`, `map`, `line`, `polyline`, `curve`, `rect`, `ellipse`, `fill`, `text`, `stamp`, `erase`, `outline`.
Colour: `replace_color`, `remap`, `flash`, `palette_lock`, `set_palette`, `clean`.
Repair: `fix_glitches` (erases strays and spurs, fills holes, outlines cuts, until `find_glitches` finds none), `seal_cuts`.
Transform: `flip`, `rotate` (RotSprite), `scale`, `squash_stretch`.
Motion and FX: `tween_move`, `shake`, `fade`, `damage_number`, `blink_flash`.

Drawing ops take `frame` or `frames` (`[1,3]`, `"all"`, `"2-6"`, `"1,3,5-8"`), so one op can draw
on many frames at once. Add `lock_alpha: true` to paint only over existing pixels.

```json
[
  {"op": "add_layer", "name": "fx"},
  {"op": "map", "layer": "fx", "frame": 4, "x": 23, "y": 20,
   "legend": {"w": "#ffffff", "q": "#bfe0ff"}, "rows": ["..www.", ".wwqq.", "wwq..."]},
  {"op": "flash", "layer": "slime", "frames": "5-6", "color": "#e04040", "amount": 0.8},
  {"op": "damage_number", "text": "-12", "x": 58, "y": 32, "frames": "6-14", "rise": 10}
]
```

## From an image to an animated character

1. `pixelize(image, "hero.aseprite")`: check the preview. If the grid guess is off, pass `cell`.
   For a painted image with no grid, pass `target_height`.
2. `rig_preview("hero.aseprite", rig)`: adjust the masks and pivots until each part is clean.
3. `animate_rig("hero.aseprite", rig, out_layer="idle", preset="idle", source_layer="sprite")`.
4. `find_glitches("hero.aseprite", layer="idle", loop=true)`, then `apply` op `fix_glitches` and check again.
   `check_animation(..., kind="loop")`, `check_identity(..., master="hero.aseprite")`, then `view(..., onion=true)`.
5. `export("hero.aseprite", "hero_idle.gif", scale=4)`.

## Between two key poses

1. Put both drawings on one canvas, on two layers (`start`, `end`), where they should be in the
   scene (the end of a jump higher up).
2. Write a rig for each with the same part names and pivots on the same joints (neck, shoulder,
   hip, grip). Children hang from their parent (`"parent"`); use `"overlap_from": "parent"` so a
   limb that swings far does not drag pixels it only touched.
3. `tween_poses(path, rig_start, rig_end, "start", "end", "jump", inbetweens=4,
   extra=[{"torso": {"dy": 3}}, {"torso": {"dy": -1}}, {}, {}])`: here a crouch before the take-off.
   The report gives each part's turn and how well it matched; under 0.4 the part changes shape a
   lot (foreshortening), so look at it closely.
4. `find_glitches(path, layer="jump", master_frames=[1, 6])` (both drawings are references), then
   `fix_glitches`, and fix by hand what is left.

## Studio Live (built-in viewer)

The server starts a local web viewer with it (127.0.0.1, port 7790 or `ASE_STUDIO_LIVE_PORT`;
`ASE_STUDIO_LIVE=0` turns it off; two sessions share one viewer). `live_open` returns the URL;
`python -m aseprite_mcp_studio.live [--port N] [folder ...]` runs it on its own. It draws with the
same renderer as the tools, so it shows exactly what gets exported.

- Plays frames at their real durations, by tag (forward, reverse, ping-pong), at 0.25x to 2x.
- Onion skin (previous red, next blue), pixel grid, checker / green / dark / light backgrounds,
  integer zoom with the wheel (anchored on the cursor), drag the view holding Space, with the
  middle button or the Move tool.
- Compare two frames side by side, as a difference (changed pixels in magenta) or blinking.
- Versions: every save is kept (last 30 per file in memory); pick an older one and see what a
  fix changed against the current file.
- `find_glitches` boxes over the frames and serious-problem counts on the timeline; the "keys"
  field takes the key drawings of a transition (e.g. `1,6`).
- Review: select wrong frames (Ctrl/Shift-click or the check box), paint the exact pixels with
  the brush or a box, add quick tags (hole, stray pixel, cut, limb wrong, flicker, stiff, pose,
  colour, shape, timing) and a note. Notes are saved next to the sprite as
  `<name>.aseprite.review.json`; the agent reads them with `review_notes`, fixes, and answers
  with `resolve_note`, and the page shows the answer live.
- Updates live (server-sent events) when a tool edits a file or it is saved in Aseprite; each
  version of a sprite renders once, as one strip the page slices.

Keys: V move, B brush, R box, E eraser, Space play (hold it to drag the view), wheel zoom, arrows step, S select frame, O onion,
G grid, X glitches, M notes, D compare, 1-9 zoom, N save note, Esc clear, Ctrl+Z undo and
Ctrl+Y / Ctrl+Shift+Z redo (strokes, boxes, erasing, frame selection, clearing).

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```

The tests rebuild a 15-frame reference scene exactly and, when Aseprite is installed, open the
written files in Aseprite and compare its renders with this server's.

## Acknowledgements

The ideas behind several tools were inspired by two other open Aseprite MCP servers,
[MalloyTheDev/aseprite-mcp](https://github.com/MalloyTheDev/aseprite-mcp) and
[diivi/aseprite-mcp](https://github.com/diivi/aseprite-mcp). This is an independent
implementation; no code was taken from either. RotSprite is Xenowhirl's algorithm. OKLab is
Björn Ottosson's colour space. The `.aseprite` format follows Aseprite's published specification.

## License

MIT
