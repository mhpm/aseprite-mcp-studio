"""Studio Live: a local web viewer built into the server, and a review channel back to the agent.

The server renders with the same code as every tool, so what the page shows is exactly what the
agent sees and exports. It runs in a background thread of the MCP process (or on its own with
`python -m aseprite_mcp_studio.live`), on 127.0.0.1 only, with the standard library.

What it serves
* /                      the page (live_ui.html)
* /api/files             .aseprite files under the allowed roots, newest first
* /api/sprite?path=      size, frames, durations, tags, layers, versions
* /api/strip?path=&v=    every frame of one version side by side, 1x PNG (the page slices it;
                         one request per version, cached until the file changes)
* /api/glitches?path=    find_glitches for the file (boxes drawn over the frames)
* /api/notes             review notes: GET list, POST add, PATCH update, DELETE remove
* /api/events            server-sent events: a file changed, a note changed

Versions: every change of a file (an edit by a tool, or a save in Aseprite) is kept in memory
(last 30 per file), so the page can step back and show what a fix changed.

Notes live next to the sprite in `<name>.aseprite.review.json`, so they survive restarts and the
agent reads them with the review_notes tool:
  {"notes": [{"id": "n3", "frames": [3, 4], "marks": {"4": [[x, y], ...]}, "tags": ["hole"],
              "text": "...", "status": "open|resolved", "reply": "...", "created": ..., "updated": ...}]}
Frames are 1-based and marks are sprite pixel coordinates.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from . import ase_format, render, review
from .document import Sprite, StudioError

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
HISTORY_DEPTH = 30
NAME = "aseprite-mcp-studio live"


def notes_path(p: Path) -> Path:
    return p.with_name(p.name + ".review.json")


class Notes:
    """Review notes per sprite, in a JSON file next to it (one lock for all writes)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def read(self, p: Path) -> list[dict]:
        f = notes_path(p)
        if not f.exists():
            return []
        try:
            return json.loads(f.read_text(encoding="utf-8")).get("notes", [])
        except (OSError, ValueError):
            return []

    def _write(self, p: Path, notes: list[dict]) -> None:
        f = notes_path(p)
        tmp = f.with_name(f.name + ".tmp")
        tmp.write_text(json.dumps({"notes": notes}, indent=1), encoding="utf-8")
        tmp.replace(f)

    def add(self, p: Path, note: dict) -> dict:
        with self._lock:
            notes = self.read(p)
            n = 1 + max((int(x["id"][1:]) for x in notes if str(x.get("id", "")).startswith("n")), default=0)
            now = time.time()
            clean = {
                "id": f"n{n}",
                "frames": sorted({int(f) for f in note.get("frames", [])}),
                "marks": {str(int(k)): [[int(a), int(b)] for a, b in v] for k, v in (note.get("marks") or {}).items()},
                "tags": [str(t) for t in note.get("tags", [])][:12],
                "text": str(note.get("text", ""))[:2000],
                "status": "open",
                "reply": "",
                "version": note.get("version"),
                "created": now,
                "updated": now,
            }
            clean["frames"] = sorted(set(clean["frames"]) | {int(k) for k in clean["marks"]})
            if not clean["frames"]:
                raise StudioError("a note needs at least one frame (select frames or paint on one)")
            notes.append(clean)
            self._write(p, notes)
            return clean

    def update(self, p: Path, nid: str, **fields) -> dict:
        with self._lock:
            notes = self.read(p)
            for x in notes:
                if x["id"] == nid:
                    for k in ("status", "reply", "text", "tags"):
                        if k in fields and fields[k] is not None:
                            x[k] = fields[k]
                    x["updated"] = time.time()
                    self._write(p, notes)
                    return x
            raise StudioError(f"no note {nid} for {p.name}")

    def delete(self, p: Path, nid: str) -> None:
        with self._lock:
            notes = [x for x in self.read(p) if x["id"] != nid]
            self._write(p, notes)


class Live:
    def __init__(self, store, port: int | None = None, host: str = "127.0.0.1") -> None:
        self.store = store
        self.host = host
        self.port = int(os.environ.get("ASE_STUDIO_LIVE_PORT", "7790")) if port is None else port
        self.notes = Notes()
        self._server: ThreadingHTTPServer | None = None
        self._clients: list[queue.Queue] = []
        self._clients_lock = threading.Lock()
        self._mtimes: dict[Path, float] = {}
        self._versions: dict[Path, list[dict]] = {}  # newest last: {"v": n, "time": t, "data": bytes}
        self._counter = 0
        self._strips: OrderedDict = OrderedDict()  # (path, v, layers) -> png bytes
        self._glitches: dict = {}
        self._lock = threading.Lock()
        self.url = ""

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> str:
        if self._server:
            return self.url
        try:
            self._server = ThreadingHTTPServer((self.host, self.port), _handler(self))
        except OSError:
            # the port is taken: if it is another studio live (a second session), share it
            url = f"http://{self.host}:{self.port}/"
            if _ping(url):
                self.url = url
                return url
            self._server = ThreadingHTTPServer((self.host, 0), _handler(self))
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self.url = f"http://{self.host}:{self.port}/"
        threading.Thread(target=self._server.serve_forever, name="studio-live", daemon=True).start()
        threading.Thread(target=self._watch, name="studio-live-watch", daemon=True).start()
        self.store.listeners.append(self.changed)
        return self.url

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self.changed in self.store.listeners:
            self.store.listeners.remove(self.changed)

    # ------------------------------------------------------------------ files and versions
    def files(self) -> list[dict]:
        out = []
        for root in self.store.roots:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
                for fn in filenames:
                    if fn.lower().endswith((".aseprite", ".ase")):
                        p = Path(dirpath) / fn
                        try:
                            st = p.stat()
                        except OSError:
                            continue
                        rel = p.relative_to(root).as_posix()
                        notes = self.notes.read(p) if notes_path(p).exists() else []
                        out.append(
                            {
                                "path": str(p),
                                "rel": rel,
                                "mtime": st.st_mtime,
                                "open_notes": sum(1 for n in notes if n.get("status") == "open"),
                            }
                        )
        out.sort(key=lambda f: -f["mtime"])
        return out

    def _snapshot(self, p: Path) -> dict | None:
        try:
            data = p.read_bytes()
        except OSError:
            return None
        with self._lock:
            hist = self._versions.setdefault(p, [])
            if hist and hist[-1]["data"] == data:
                return hist[-1]
            self._counter += 1
            v = {"v": self._counter, "time": time.time(), "data": data}
            hist.append(v)
            del hist[:-HISTORY_DEPTH]
            return v

    def version(self, p: Path) -> dict | None:
        hist = self._versions.get(p)
        if not hist or self._mtimes.get(p) != _mtime(p):
            self._mtimes[p] = _mtime(p)
            return self._snapshot(p)
        return hist[-1]

    def sprite_at(self, p: Path, v: int | None) -> tuple[Sprite, int]:
        cur = self.version(p)
        if cur is None:
            raise StudioError(f"{p} does not exist")
        if v is None or v == cur["v"]:
            return ase_format.read_bytes(cur["data"]), cur["v"]
        for h in self._versions.get(p, []):
            if h["v"] == v:
                return ase_format.read_bytes(h["data"]), v
        raise StudioError(f"version {v} of {p.name} is no longer kept")

    def changed(self, p: Path) -> None:
        """Called by the store after every write, and by the watcher for outside saves."""
        p = Path(p)
        self._mtimes[p] = _mtime(p)
        v = self._snapshot(p) if p.exists() else None
        with self._lock:
            for k in [k for k in self._strips if k[0] == p]:
                self._strips.pop(k, None)
            self._glitches = {k: g for k, g in self._glitches.items() if k[0] != p}
        self.broadcast({"type": "file", "path": str(p), "v": v["v"] if v else None})

    def _watch(self) -> None:
        note_mtimes: dict[Path, float] = {}
        while self._server:
            try:
                seen = {Path(f["path"]): f["mtime"] for f in self.files()}
                for p, m in seen.items():
                    if p in self._mtimes and self._mtimes[p] != m:
                        self.changed(p)
                    self._mtimes.setdefault(p, m)
                    # notes written by another process (the agent's resolve_note) reach the page too
                    nm = _mtime(notes_path(p))
                    if p in note_mtimes and note_mtimes[p] != nm:
                        self.broadcast({"type": "notes", "path": str(p)})
                    note_mtimes[p] = nm
            except Exception:  # noqa: BLE001 - the watcher must survive odd files
                pass
            time.sleep(0.6)

    # ------------------------------------------------------------------ rendering
    def strip(self, p: Path, v: int | None, layers: list[str] | None) -> tuple[bytes, int]:
        spr, ver = self.sprite_at(p, v)
        key = (p, ver, tuple(layers or ()))
        with self._lock:
            if key in self._strips:
                self._strips.move_to_end(key)
                return self._strips[key], ver
        w, h, n = spr.width, spr.height, spr.frame_count
        sheet = np.zeros((h, w * n, 4), np.uint8)
        names = [la.path for la in spr.layers if la.path in set(layers)] if layers else None
        for f in range(n):
            sheet[:, f * w : (f + 1) * w] = render.render_frame(spr, f, names or None)
        buf = io.BytesIO()
        Image.fromarray(sheet, "RGBA").save(buf, "PNG", optimize=False, compress_level=3)
        data = buf.getvalue()
        with self._lock:
            self._strips[key] = data
            while len(self._strips) > 64:
                self._strips.popitem(last=False)
        return data, ver

    def glitches(self, p: Path, layer: str | None, loop: bool, masters: list[int] | None) -> dict:
        spr, ver = self.sprite_at(p, None)
        key = (p, ver, layer, loop, tuple(masters or ()))
        if key not in self._glitches:
            ms = [render.render_frame(spr, f - 1, [layer] if layer else None) for f in masters] if masters else None
            res = review.find_glitches(spr, layer, None, ms, loop)
            self._glitches[key] = {"per_frame": res["per_frame"], "issues": res["issues"], "kinds": res["kinds"]}
        return self._glitches[key]

    # ------------------------------------------------------------------ events
    def broadcast(self, event: dict) -> None:
        msg = json.dumps(event)
        with self._clients_lock:
            for q in list(self._clients):
                with contextlib.suppress(queue.Full):
                    q.put_nowait(msg)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._clients_lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._clients_lock:
            if q in self._clients:
                self._clients.remove(q)


def note_picture(spr: Sprite, note: dict, scale: int = 0) -> bytes:
    """One PNG per note: for each marked frame (or each selected frame without marks) the
    previous frame, the frame with the user's marks, and the next frame, side by side, zoomed on
    the marked area, so the agent judges the spot against its neighbours as the user did."""
    from .font import render_text

    rows = []
    marked = {int(k): v for k, v in note.get("marks", {}).items() if v}
    frames = sorted(marked) or note.get("frames", [])[:4]
    w, h = spr.width, spr.height
    for f in frames[:6]:
        pts = marked.get(f, [])
        if pts:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            x0, y0 = max(0, min(xs) - 8), max(0, min(ys) - 8)
            x1, y1 = min(w, max(xs) + 9), min(h, max(ys) + 9)
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        cw, ch = x1 - x0, y1 - y0
        s = scale or max(2, min(12, 220 // max(cw, ch)))
        tiles = []
        for g, label in ((f - 1, f"F{f - 1}"), (f, f"F{f}  {note['id']}"), (f + 1, f"F{f + 1}")):
            tile = np.zeros((ch * s + 12, cw * s, 4), np.uint8)
            tile[...] = (250, 250, 252, 255)
            if 1 <= g <= spr.frame_count:
                img = render.render_frame(spr, g - 1)[y0:y1, x0:x1]
                bg = render._checker(ch * s, cw * s, max(4, s * 2))
                acc = bg.astype(float) / 255
                render.composite_onto(acc, render.upscale(img, s))
                area = render.to_uint8(acc)
                if g == f:
                    for x, y in pts:
                        yy, xx = (y - y0) * s, (x - x0) * s
                        cell = area[yy : yy + s, xx : xx + s].astype(np.int16)
                        area[yy : yy + s, xx : xx + s] = (cell + (np.array([255, 40, 140, 255]) - cell) * 0.55).astype(
                            np.uint8
                        )
                tile[12:] = area
            render._stamp(tile, render_text(label, (40, 40, 56, 255)), 2, 2)
            tiles.append(tile)
        gap = np.full((tiles[0].shape[0], 6, 4), 255, np.uint8)
        rows.append(np.concatenate([tiles[0], gap, tiles[1], gap, tiles[2]], 1))
    if not rows:
        rows = [np.full((20, 20, 4), 255, np.uint8)]
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 6), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    buf = io.BytesIO()
    Image.fromarray(np.concatenate(rows, 0), "RGBA").save(buf, "PNG")
    return buf.getvalue()


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return -1.0


def _ping(url: str) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(url + "api/ping", timeout=1) as r:
            return json.loads(r.read()).get("name") == NAME
    except Exception:  # noqa: BLE001
        return False


def _ui() -> bytes:
    return (Path(__file__).with_name("live_ui.html")).read_bytes()


def _handler(live: Live):
    class Handler(BaseHTTPRequestHandler):
        server_version = "StudioLive/1"

        def log_message(self, *args) -> None:  # stdout belongs to the MCP protocol: stay quiet
            return

        # -------------------------------------------------------------- helpers
        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _path(self, q: dict) -> Path:
            raw = (q.get("path") or [""])[0]
            return live.store.resolve(raw, must_exist=True)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def _route(self, method: str) -> None:
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if method == "GET":
                    self._get(u.path, q)
                elif method == "POST" and u.path == "/api/notes":
                    b = self._body()
                    p = live.store.resolve(b.get("path", ""), must_exist=True)
                    cur = live.version(p)
                    note = live.notes.add(p, {**b, "version": cur["v"] if cur else None})
                    live.broadcast({"type": "notes", "path": str(p)})
                    self._json(note)
                elif method == "PATCH" and u.path == "/api/notes":
                    b = self._body()
                    p = live.store.resolve(b.get("path", ""), must_exist=True)
                    note = live.notes.update(
                        p, b["id"], status=b.get("status"), reply=b.get("reply"), text=b.get("text"), tags=b.get("tags")
                    )
                    live.broadcast({"type": "notes", "path": str(p)})
                    self._json(note)
                elif method == "DELETE" and u.path == "/api/notes":
                    p = self._path(q)
                    live.notes.delete(p, (q.get("id") or [""])[0])
                    live.broadcast({"type": "notes", "path": str(p)})
                    self._json({"ok": True})
                else:
                    self._json({"error": "not found"}, 404)
            except StudioError as e:
                self._json({"error": str(e)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # noqa: BLE001 - report, never kill the server
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _get(self, path: str, q: dict) -> None:
            if path in ("/", "/index.html"):
                self._send(200, _ui(), "text/html; charset=utf-8")
            elif path == "/api/ping":
                self._json({"name": NAME, "roots": [str(r) for r in live.store.roots]})
            elif path == "/api/files":
                self._json({"files": live.files()})
            elif path == "/api/sprite":
                p = self._path(q)
                spr, ver = live.sprite_at(p, None)
                self._json(
                    {
                        "path": str(p),
                        "name": p.name,
                        "width": spr.width,
                        "height": spr.height,
                        "frames": spr.frame_count,
                        "durations": spr.durations,
                        "tags": [
                            {"name": t.name, "from": t.start + 1, "to": t.end + 1, "direction": t.direction}
                            for t in spr.tags
                        ],
                        "layers": [{"path": la.path, "visible": la.visible, "kind": la.kind} for la in spr.layers],
                        "v": ver,
                        "versions": [{"v": h["v"], "time": h["time"]} for h in live._versions.get(p, [])],
                    }
                )
            elif path == "/api/strip":
                p = self._path(q)
                v = int(q["v"][0]) if q.get("v") else None
                layers = [x for x in (q.get("layers") or [""])[0].split(",") if x] or None
                data, ver = live.strip(p, v, layers)
                self._send(200, data, "image/png", {"X-Version": str(ver)})
            elif path == "/api/glitches":
                p = self._path(q)
                layer = (q.get("layer") or [""])[0] or None
                masters = [int(x) for x in (q.get("masters") or [""])[0].split(",") if x] or None
                self._json(live.glitches(p, layer, (q.get("loop") or ["0"])[0] == "1", masters))
            elif path == "/api/notes":
                p = self._path(q)
                self._json({"notes": live.notes.read(p)})
            elif path == "/api/events":
                self._events()
            else:
                self._json({"error": "not found"}, 404)

        def _events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = live.subscribe()
            try:
                self.wfile.write(b": hello\n\n")
                self.wfile.flush()
                while live._server:
                    try:
                        msg = q.get(timeout=15)
                        self.wfile.write(f"data: {msg}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                live.unsubscribe(q)

        def do_GET(self) -> None:  # noqa: N802
            self._route("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._route("POST")

        def do_PATCH(self) -> None:  # noqa: N802
            self._route("PATCH")

        def do_DELETE(self) -> None:  # noqa: N802
            self._route("DELETE")

    return Handler


def main() -> None:
    """Run the viewer on its own: python -m aseprite_mcp_studio.live [--port N] [root ...]."""
    import argparse

    from .store import Store

    ap = argparse.ArgumentParser(description="Studio Live viewer")
    ap.add_argument("roots", nargs="*", help="folders to browse (default ASE_STUDIO_ROOTS or the working directory)")
    ap.add_argument("--port", type=int, default=None)
    a = ap.parse_args()
    live = Live(Store(a.roots or None), port=a.port)
    print(f"Studio Live: {live.start()}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        live.stop()


if __name__ == "__main__":
    main()
