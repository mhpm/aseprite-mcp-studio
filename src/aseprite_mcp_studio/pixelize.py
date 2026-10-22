"""Turn an image into real pixel art: one stored pixel per logical pixel, a small palette.

Two kinds of input are handled:

* "Fake" pixel art (what image models produce): it imitates pixel art, but every logical
  pixel is a soft, noisy block of roughly s x s screen pixels, the grid drifts by a pixel
  here and there, and there are thousands of colours. Here the grid is recovered:

    1. edge-energy profile per column and per row (colour and alpha jumps);
    2. cell size s from the autocorrelation of those profiles (computed with an FFT), taking
       the first clear peak so sub-harmonics s/2, s/3 are not chosen, refined with the
       later peaks 2s, 3s;
    3. grid lines per axis by dynamic programming: steps of round(s)-1 .. round(s)+1, each
       cut landing on edge energy, penalised for straying from s. This follows the drift
       instead of assuming a perfect lattice, which is why columns do not "slip";
    4. each cell becomes one pixel. Coverage decides alpha; the colour is the perceptual
       *mode* of the cell interior (densest cluster in OKLab), not the median or the mean,
       so a cell straddling an outline does not average into a muddy in-between shade.

* A real illustration (no grid found) or an explicit `target_height`: the same cell
  sampling runs on a uniform grid, which is a majority-vote downscale rather than a blur.

Then the palette is reduced in OKLab by agglomerative merging, weighted by how many pixels
each colour covers, but a small cluster that is perceptually far from everything (an eye
highlight, a blush) is kept rather than swallowed. Detached specks are removed and
optionally an outline is added. Every number that drove a decision comes back in the report.
"""

from __future__ import annotations

import numpy as np

from .color import nearest, rgb_to_oklab, to_hex
from .raster import components, outline_mask


# ---------------------------------------------------------------------------- background
def remove_background(a: np.ndarray, tolerance: float = 6.0) -> tuple[np.ndarray, bool]:
    """If the image has no transparency, flood the border colour away (OKLab distance
    `tolerance` x 0.01). Returns (image, removed?)."""
    if (a[..., 3] < 250).mean() > 0.01:
        return a, False
    h, w = a.shape[:2]
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])[:, :3]
    vals, counts = np.unique(border, axis=0, return_counts=True)
    bg = vals[counts.argmax()].astype(float)
    if counts.max() < 0.4 * len(border):
        return a, False
    lab = rgb_to_oklab(a[..., :3].astype(float))
    d = np.linalg.norm(lab - rgb_to_oklab(bg), axis=-1) * 100
    near = d <= tolerance
    # only background connected to the border, so same-coloured details inside survive
    labels, _ = components(near, diagonal=False)
    edge_labels = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))) - {0}
    bgmask = np.isin(labels, list(edge_labels))
    out = a.copy()
    out[bgmask] = 0
    return out, True


# ---------------------------------------------------------------------------- grid
def edge_profiles(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = a.astype(np.int32)
    on = a[..., 3:4] >= 128
    rgb = a[..., :3] * on
    alpha = on[..., 0].astype(np.int32) * 255
    dx = np.abs(np.diff(rgb, axis=1)).sum(-1) + np.abs(np.diff(alpha, axis=1))
    dy = np.abs(np.diff(rgb, axis=0)).sum(-1) + np.abs(np.diff(alpha, axis=0))
    thr = 60  # only clear edges vote; the noise inside a block must not
    dx, dy = np.where(dx > thr, dx, 0), np.where(dy > thr, dy, 0)
    return (np.concatenate([[0], dx.sum(0)]).astype(float), np.concatenate([[0], dy.sum(1)]).astype(float))


def _acf(p: np.ndarray, maxlag: int) -> np.ndarray:
    p = p - p.mean()
    n = len(p)
    f = np.fft.rfft(p, 2 * n)
    r = np.fft.irfft(f * np.conj(f))[:maxlag]
    return r / max(r[0], 1e-9)


def _parabolic(r: np.ndarray, t: int) -> float:
    a, b, c = r[t - 1], r[t], r[t + 1]
    den = a - 2 * b + c
    return t + (0.5 * (a - c) / den if den else 0.0)


def detect_cell(px: np.ndarray, py: np.ndarray, lo: int = 2, hi: int = 24) -> tuple[float, float]:
    """(cell size, confidence). Cell 1.0 means no grid was found."""
    maxlag = min(hi * 4 + 2, len(px) - 1, len(py) - 1)
    if maxlag <= lo + 2:
        return 1.0, 0.0
    r = (_acf(px, maxlag) + _acf(py, maxlag)) / 2
    first = None
    for t in range(max(lo, 2), min(hi, maxlag - 2) + 1):
        if r[t] >= r[t - 1] and r[t] >= r[t + 1] and r[t] > 0.12 and r[t] - r[1:t].min() > 0.12:
            first = t
            break
    if first is None:
        return 1.0, float(r[lo : min(hi, maxlag)].max())
    est = [_parabolic(r, first)]
    for k in (2, 3, 4):
        c = round(est[0] * k)
        if c + 2 >= maxlag:
            break
        t = max(range(c - 1, c + 2), key=lambda i: r[i])
        if r[t] > 0.08:
            est.append(_parabolic(r, t) / k)
    return float(np.median(est)), float(r[first])


def fit_cuts(p: np.ndarray, s: float) -> list[int]:
    """Grid lines along one axis by dynamic programming (see module docstring)."""
    n = len(p)
    prof = np.concatenate([p, [0.0]])
    pen = 2.0 * (prof[prof > 0].mean() if (prof > 0).any() else 1.0)
    steps = list(range(max(2, round(s) - 1), round(s) + 2))
    dmax = max(steps)
    score = np.full(n + 1, -np.inf)
    back = np.full(n + 1, -1)
    score[:dmax] = prof[:dmax]
    costs = {d: pen * (d - s) ** 2 for d in steps}
    for x in range(dmax, n + 1):
        cands = [(score[x - d] - costs[d], x - d) for d in steps]
        best, arg = max(cands)
        score[x] = prof[x] + best
        back[x] = arg
    end = n + 1 - dmax + int(score[n + 1 - dmax :].argmax())
    cuts = [end]
    while back[cuts[-1]] >= 0:
        cuts.append(int(back[cuts[-1]]))
    cuts = cuts[::-1]
    # close the borders: a sliver under half a cell joins its neighbour, a larger one is a cell
    if cuts[0] > 0:
        if cuts[0] >= s * 0.5:
            cuts.insert(0, 0)
        else:
            cuts[0] = 0
    if cuts[-1] < n:
        if n - cuts[-1] >= s * 0.5:
            cuts.append(n)
        else:
            cuts[-1] = n
    return cuts


# ---------------------------------------------------------------------------- sampling
def _cell_mode(lab: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """The colour of the densest cluster among a cell's pixels (OKLab)."""
    if len(lab) <= 2:
        return np.median(rgb, 0)
    d = np.linalg.norm(lab[:, None] - lab[None], axis=-1)
    support = (d < 0.04).sum(1)  # neighbours within ~4 JND
    core = d[support.argmax()] < 0.04
    return np.median(rgb[core], 0)


def sample_cells(a: np.ndarray, cx: list[int], cy: list[int], coverage: float = 0.5) -> np.ndarray:
    out = np.zeros((len(cy) - 1, len(cx) - 1, 4), np.uint8)
    lab_all = rgb_to_oklab(a[..., :3].astype(float))
    for j, (y0, y1) in enumerate(zip(cy, cy[1:], strict=False)):
        for i, (x0, x1) in enumerate(zip(cx, cx[1:], strict=False)):
            cell = a[y0:y1, x0:x1]
            opaque = cell[..., 3] >= 128
            if opaque.mean() < coverage:
                continue
            # the interior: generated blocks are blurred at their borders
            iy0, iy1, ix0, ix1 = y0, y1, x0, x1
            if y1 - y0 >= 4:
                iy0, iy1 = y0 + 1, y1 - 1
            if x1 - x0 >= 4:
                ix0, ix1 = x0 + 1, x1 - 1
            inner = a[iy0:iy1, ix0:ix1]
            io = inner[..., 3] >= 128
            if io.sum() >= 1:
                rgb = inner[io][:, :3].astype(float)
                lab = lab_all[iy0:iy1, ix0:ix1][io]
            else:
                rgb = cell[opaque][:, :3].astype(float)
                lab = lab_all[y0:y1, x0:x1][opaque]
            out[j, i, :3] = np.round(_cell_mode(lab, rgb)).clip(0, 255)
            out[j, i, 3] = 255
    return out


# ---------------------------------------------------------------------------- palette
def reduce_palette(
    img: np.ndarray, max_colors: int = 24, merge_delta: float = 3.5, keep_delta: float = 12.0
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Agglomerative palette reduction in OKLab.

    Colours closer than `merge_delta` (x0.01 OKLab, ~JND units) always merge, the smaller
    into the larger. Then, while there are more than `max_colors`, the cheapest pair merges,
    cost = distance x size of the smaller cluster, so tiny clusters merge first; but a
    cluster farther than `keep_delta` from every other one is protected, because that is a
    deliberate accent (eye highlight, gem, blush) and not noise."""
    on = img[..., 3] > 0
    if not on.any():
        return img, []
    cols, inv, counts = np.unique(img[on][:, :3], axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    labs = rgb_to_oklab(cols.astype(float))
    w = counts.astype(float)
    # initial coarse grouping (exact neighbours) when there are very many colours
    if len(cols) > 256:
        from PIL import Image as _I

        q = _I.fromarray(cols.reshape(1, -1, 3).astype(np.uint8), "RGB").quantize(
            256, method=_I.Quantize.MEDIANCUT, dither=_I.Dither.NONE
        )
        grp = np.array(q).ravel().astype(np.int64)
    else:
        grp = np.arange(len(cols))
    ng = grp.max() + 1
    gw = np.bincount(grp, weights=w, minlength=ng)
    glab = (
        np.stack([np.bincount(grp, weights=w * labs[:, k], minlength=ng) for k in range(3)], 1)
        / np.maximum(gw, 1e-9)[:, None]
    )
    alive = gw > 0
    parent = np.arange(ng)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    while True:
        idx = np.flatnonzero(alive)
        if len(idx) <= 1:
            break
        L = glab[idx]
        d = np.linalg.norm(L[:, None] - L[None], axis=-1) * 100
        np.fill_diagonal(d, np.inf)
        nearest_d = d.min(1)
        i, j = np.unravel_index(d.argmin(), d.shape)
        if d[i, j] >= merge_delta:
            if len(idx) <= max_colors:
                break
            small = np.minimum(gw[idx][:, None], gw[idx][None])
            cost = d * small
            protected = nearest_d > keep_delta
            cost[protected, :] = np.inf
            cost[:, protected] = np.inf
            if not np.isfinite(cost).any():
                cost = d * small
            i, j = np.unravel_index(cost.argmin(), cost.shape)
        a_, b_ = idx[i], idx[j]
        keep, drop = (a_, b_) if gw[a_] >= gw[b_] else (b_, a_)
        tot = gw[keep] + gw[drop]
        glab[keep] = (glab[keep] * gw[keep] + glab[drop] * gw[drop]) / tot
        gw[keep] = tot
        alive[drop] = False
        parent[drop] = keep
    roots = np.array([root(g) for g in range(ng)])
    # each final colour is the most common original colour in its cluster (a real sample,
    # not an average that may not occur anywhere in the art)
    final_group = roots[grp]
    pal_rgb = {}
    for g in np.unique(final_group):
        members = np.flatnonzero(final_group == g)
        pal_rgb[g] = cols[members[w[members].argmax()]]
    mapped = np.array([pal_rgb[g] for g in final_group])
    out = img.copy()
    out[on, :3] = mapped[inv]
    palette = sorted(
        {tuple(int(v) for v in c) for c in pal_rgb.values()}, key=lambda c: rgb_to_oklab(np.array(c, float))[0]
    )
    return out, palette


def remove_specks(img: np.ndarray, max_size: int = 2) -> int:
    """Delete detached opaque islands of <= max_size px (8-connected)."""
    labels, n = components(img[..., 3] > 0, diagonal=True)
    if n == 0:
        return 0
    sizes = np.bincount(labels.ravel())
    small = np.flatnonzero(sizes <= max_size)
    small = small[small > 0]
    mask = np.isin(labels, small)
    img[mask] = 0
    return int(mask.sum())


def clean_orphans(img: np.ndarray, protect: list[tuple[int, int, int]] | None = None, max_delta: float = 15.0) -> int:
    """A pixel unlike all 8 neighbours, inside the shape, whose neighbours mostly agree on
    one colour, takes that colour, but only when it is perceptually close to it (OKLab
    distance under `max_delta`, ~JND units): that is sampling noise. A high-contrast single
    pixel (an eye, a highlight, a rivet) is a deliberate detail and is kept, as are colours
    listed in `protect`."""
    from .raster import pack

    p = pack(img)
    h, w = p.shape
    pad = np.pad(p, 1)
    nb = np.stack(
        [pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)],
        -1,
    )
    alpha_nb = np.stack(
        [
            np.pad(img[..., 3], 1)[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dx, dy) != (0, 0)
        ],
        -1,
    )
    orphan = (img[..., 3] > 0) & ~(nb == p[..., None]).any(-1) & (alpha_nb > 0).all(-1)
    fixed = 0
    prot = {tuple(c) for c in (protect or [])}
    for y, x in zip(*np.nonzero(orphan), strict=False):
        if tuple(int(v) for v in img[y, x, :3]) in prot:
            continue
        vals, cnt = np.unique(nb[y, x], return_counts=True)
        if cnt.max() < 5:
            continue
        major = np.array([vals[cnt.argmax()]], np.uint32).view(np.uint8)
        d = np.linalg.norm(rgb_to_oklab(img[y, x, :3].astype(float)) - rgb_to_oklab(major[:3].astype(float))) * 100
        if d < max_delta:
            img[y, x] = major
            fixed += 1
    return fixed


def add_outline(img: np.ndarray, color) -> tuple[int, np.ndarray]:
    """A 1 px orthogonal keyline outside the shape (the image grows by 1 px per side)."""
    pad = np.pad(img, ((1, 1), (1, 1), (0, 0)))
    m = outline_mask(pad[..., 3], diagonal=False)
    pad[m] = color
    return int(m.sum()), pad


# ---------------------------------------------------------------------------- pipeline
def pixelize(
    src: np.ndarray,
    *,
    cell: float | None = None,
    target_height: int | None = None,
    max_colors: int = 32,
    merge_delta: float = 3.5,
    background: str = "auto",
    outline: str | None = None,
    speck_size: int = 2,
    clean: bool = True,
    crop: bool = True,
    palette: list[tuple[int, int, int]] | None = None,
) -> tuple[np.ndarray, dict]:
    """Full conversion. Returns (native RGBA image, report)."""
    a = np.ascontiguousarray(src, np.uint8)
    report: dict = {"source_size": [a.shape[1], a.shape[0]]}
    if background in ("auto", "remove"):
        a, removed = remove_background(a)
        report["background_removed"] = removed
    px, py = edge_profiles(a)
    if target_height:
        mode = "downscale"
        bbox_rows = np.flatnonzero((a[..., 3] > 0).any(1))
        content_h = (bbox_rows[-1] - bbox_rows[0] + 1) if bbox_rows.size else a.shape[0]
        s = content_h / target_height
        conf = None
    elif cell:
        mode, s, conf = "grid", float(cell), None
    else:
        s, conf = detect_cell(px, py)
        mode = "grid" if s >= 2 else "native"
    report.update({"mode": mode, "cell_px": round(s, 3), "grid_confidence": None if conf is None else round(conf, 3)})
    h, w = a.shape[:2]
    if mode == "native":
        small = a.copy()
        small[small[..., 3] < 128] = 0
        small[small[..., 3] >= 128, 3] = 255
    elif mode == "grid":
        cx, cy = fit_cuts(px, s), fit_cuts(py, s)
        widths = np.diff(cx)
        report["cell_width_range"] = [int(widths.min()), int(widths.max())]
        small = sample_cells(a, cx, cy)
    else:
        cx = [int(round(v)) for v in np.arange(0, w + 1e-9, s)] + ([w] if (w / s) % 1 else [])
        cy = [int(round(v)) for v in np.arange(0, h + 1e-9, s)] + ([h] if (h / s) % 1 else [])
        cx, cy = sorted(set(cx)), sorted(set(cy))
        small = sample_cells(a, cx, cy, coverage=0.45)
    if palette:
        on = small[..., 3] > 0
        pal = np.array(palette, float)
        idx = nearest(small[on][:, :3].astype(float), pal)
        small[on, :3] = pal[idx].astype(np.uint8)
        pal_out = [tuple(int(v) for v in c) for c in palette]
        report["palette_locked"] = True
    else:
        small, pal_out = reduce_palette(small, max_colors, merge_delta)
    report["specks_removed"] = remove_specks(small, speck_size) if speck_size else 0
    report["orphans_cleaned"] = clean_orphans(small) if clean else 0
    if outline:
        col = (
            min(pal_out, key=lambda c: rgb_to_oklab(np.array(c, float))[0]) + (255,)
            if outline == "auto"
            else tuple(outline)
        )
        n_out, small = add_outline(small, col)
        report["outline_pixels_added"] = n_out
    if crop:
        rows = np.flatnonzero((small[..., 3] > 0).any(1))
        cols = np.flatnonzero((small[..., 3] > 0).any(0))
        if rows.size:
            report["crop_offset"] = [int(cols[0]), int(rows[0])]
            small = small[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy()
    used = np.unique(small[small[..., 3] > 0][:, :3], axis=0)
    report.update(
        {
            "native_size": [small.shape[1], small.shape[0]],
            "colors": int(len(used)),
            "palette": [to_hex(c) for c in used],
            "suggested_preview_scale": max(1, round(s)),
        }
    )
    return small, report
