"""Media viewer: a local web app that shows what the session produces.

``open_viewer`` starts a localhost HTTP server (once per process) and opens
it in the browser. The page polls ``/api/items`` and renders every file by
kind: image, video, audio, 3D, HTML, text/code. The session dir is the source
of truth, as it is for the asset sweep; tracked artifacts add prompt/model
captions and cover remote results that haven't downloaded yet.

Files are served at ``/fs/<absolute path>`` so an HTML page in the session
dir can load its sibling files by relative URL.
"""

import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote, urlparse

from typing_extensions import TypedDict

from okonomi.session import favorite_hits, favorite_refs, norm_ref
from okonomi.tui.assets import set_favorite


_TEXT_EXTS = {
    ".md", ".markdown", ".rst", ".txt", ".log",
    ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml", ".ini", ".csv",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".rb", ".sh", ".css",
}
# <model-viewer> only renders glTF; other 3D formats fall through to a download link.
_3D_EXTS = {".glb", ".gltf"}
_KINDS = {"image", "video", "audio", "3d", "html", "text"}
_SKIP_NAMES = {"traces.jsonl"}
_CHUNK = 1 << 20


class ViewerResult(TypedDict, total=False):
    url: str
    error: str


def _kind(name: str, fallback: str = "file") -> str:
    suffix = Path(name.split("?", 1)[0]).suffix.lower()
    if suffix in (".html", ".htm"):
        return "html"
    if suffix in _3D_EXTS:
        return "3d"
    if suffix in _TEXT_EXTS:
        return "text"
    mime, _ = mimetypes.guess_type(name.split("?", 1)[0])
    major = mime.split("/", 1)[0] if mime else ""
    if major in _KINDS:
        return major
    return fallback if fallback in _KINDS else "file"


def _fs_url(p: Path) -> str:
    return "/fs" + quote(str(p))


def _items(session: Any) -> dict:
    """Viewer payload: every non-hidden file in the session's artifacts dir
    plus tracked artifacts that live elsewhere or are still remote-only,
    oldest first. Favorites from other sessions ride along flagged
    ``elsewhere`` so the ♥ filter is the one gallery of every favorite; the
    other filters leave them out."""
    root = session.artifacts_dir.resolve()
    by_path: dict[str, Any] = {}
    extra: list[Any] = []
    for a in session.state.artifacts:
        if a.path:
            by_path[str(Path(a.path).resolve())] = a
        elif a.url:
            extra.append(a)

    paths: set[Path] = set()
    if root.exists():
        for p in root.rglob("*"):
            rel = p.relative_to(root)
            if p.name in _SKIP_NAMES or any(part.startswith(".") for part in rel.parts):
                continue
            paths.add(p)
    paths.update(Path(k) for k in by_path)

    fav_hits = {norm_ref(h["path"] or h["url"]): h for h in favorite_hits()}
    favorites = set(fav_hits)
    elsewhere = {Path(r) for r, h in fav_hits.items() if h["path"]} - paths
    paths |= elsewhere
    tracked_urls = {a.url for a in extra}
    items = []
    for p in paths:
        try:
            if not p.is_file():
                continue
            st = p.stat()
        except OSError:
            continue
        a = by_path.get(str(p))
        name = str(p.relative_to(root)) if p.is_relative_to(root) else p.name
        other = fav_hits.get(str(p)) if p in elsewhere else None
        items.append({
            "id": f"{p}:{st.st_mtime_ns}",
            "name": name,
            "kind": _kind(p.name, a.artifact_type if a else "file"),
            "src": _fs_url(p),
            "ref": str(p),
            "favorite": str(p) in favorites,
            "size": st.st_size,
            "time": st.st_mtime,
            "caption": (a.prompt or a.description) if a else (other["snippet"] if other else ""),
            "model": a.model_id if a else "",
            "elsewhere": p in elsewhere,
        })
    for a in extra:
        items.append({
            "id": a.url,
            "name": Path(urlparse(a.url).path).name or a.url,
            "kind": _kind(a.url, a.artifact_type),
            "src": a.url,
            "ref": a.url,
            "favorite": a.url in favorites,
            "size": None,
            "time": 0,
            "caption": a.prompt or a.description,
            "model": a.model_id,
            "elsewhere": False,
        })
    for ref, h in fav_hits.items():
        if h["path"] or ref in tracked_urls:
            continue
        items.append({
            "id": ref,
            "name": Path(urlparse(ref).path).name or ref,
            "kind": _kind(ref, h["artifact_type"]),
            "src": ref,
            "ref": ref,
            "favorite": True,
            "size": None,
            "time": 0,
            "caption": h["snippet"],
            "model": "",
            "elsewhere": True,
        })
    items.sort(key=lambda i: i["time"])
    return {"title": session.title or session.session_id, "dir": str(root), "items": items}


def _servable(session: Any, p: Path) -> bool:
    if p.is_relative_to(session.artifacts_dir.resolve()) or str(p) in favorite_refs():
        return True
    return any(a.path and Path(a.path).resolve() == p for a in session.state.artifacts)


def _favorite(session: Any, body: dict) -> bool:
    """Toggle a favorite from the page. Only refs the viewer itself lists —
    servable files or tracked remote URLs — can be favorited."""
    ref = str(body.get("ref") or "")
    if ref.startswith(("http://", "https://")):
        tracked = next((a for a in session.state.artifacts if a.url == ref), None)
        if tracked is None and ref not in favorite_refs():
            return False
        artifact_type = tracked.artifact_type if tracked else _kind(ref)
    else:
        p = Path(ref).resolve()
        if not ref or not p.is_file() or not _servable(session, p):
            return False
        ref, artifact_type = str(p), _kind(p.name)
    set_favorite(
        ref,
        artifact_type="" if artifact_type == "file" else artifact_type,
        favorite=bool(body.get("favorite")),
        session=session,
    )
    return True


class _Handler(BaseHTTPRequestHandler):
    server: "_ViewerServer"

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", _PAGE.encode())
        elif path == "/api/items":
            self._send(200, "application/json", json.dumps(_items(self.server.session_fn())).encode())
        elif path.startswith("/fs/"):
            self._send_file(Path(unquote(path[3:])).resolve())
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/favorite":
            self._send(404, "text/plain", b"not found")
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        except ValueError:
            self._send(400, "text/plain", b"bad json")
            return
        if not isinstance(body, dict) or not _favorite(self.server.session_fn(), body):
            self._send(404, "text/plain", b"not found")
            return
        self._send(200, "application/json", b"{}")

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, p: Path) -> None:
        if not p.is_file() or not _servable(self.server.session_fn(), p):
            self._send(404, "text/plain", b"not found")
            return
        size = p.stat().st_size
        start, end, status = 0, size - 1, 200
        # Range support: Safari won't play <video> without it, and the page
        # uses it to fetch only the head of large text files.
        header = self.headers.get("Range", "")
        if header.startswith("bytes="):
            first, _, last = header[6:].split(",", 1)[0].strip().partition("-")
            try:
                if first:
                    start, end = int(first), min(int(last), size - 1) if last else size - 1
                else:
                    start = max(0, size - int(last))
            except ValueError:
                start = size
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = 206

        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(p.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        remaining = end - start + 1
        try:
            with open(p, "rb") as f:
                f.seek(start)
                while remaining > 0:
                    chunk = f.read(min(_CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


class _ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, session_fn: Callable[[], Any]):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.session_fn = session_fn


class ViewerTools:
    """Live media viewer for the current session."""


    def __init__(self, session_fn: Callable[[], Any]):
        # A callable, not the session: /new and resume swap the agent's session.
        self._session_fn = session_fn
        self._server: _ViewerServer | None = None
        self._lock = threading.Lock()

    def open_viewer(self) -> ViewerResult:
        """Open the media viewer in the user's browser.

        The viewer is a local web app that shows every file in the session
        directory (images, video, audio, 3D, HTML pages, code and text) and
        updates live as new outputs land. Call it once, when a task will
        produce outputs the user should watch arrive; it keeps updating for
        the rest of the session, so don't call it again for each new result.

        Returns:
            ``{"url": str}`` on success, or ``{"error": str}`` on failure.
        """
        with self._lock:
            if self._server is None:
                try:
                    self._server = _ViewerServer(self._session_fn)
                except OSError as e:
                    return {"error": f"Could not start the viewer server: {e}"}
                threading.Thread(target=self._server.serve_forever, daemon=True, name="okonomi-viewer").start()
        url = f"http://127.0.0.1:{self._server.server_port}/"
        webbrowser.open(url)
        return {"url": url}


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>okonomi viewer</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script type="module" src="https://unpkg.com/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>
<style>
:root {
  --bg: #0e0e10; --panel: #18181b; --line: #2a2a2f; --text: #e7e7ea; --dim: #8b8b94;
  --accent: #f0a35e; --radius: 10px;
  --mono: ui-monospace, "SF Mono", Menlo, monospace;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 -apple-system, system-ui, sans-serif; }
header { position: sticky; top: 0; z-index: 5; display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
  padding: 12px 20px; background: color-mix(in srgb, var(--bg) 88%, transparent); backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--line); }
h1 { font-size: 15px; margin: 0; font-weight: 600; }
h1 small { color: var(--dim); font-weight: 400; margin-left: 8px; }
.filters { display: flex; gap: 6px; margin-left: auto; flex-wrap: wrap; }
.filters button { background: var(--panel); color: var(--dim); border: 1px solid var(--line); border-radius: 999px;
  padding: 4px 11px; font: inherit; font-size: 12px; cursor: pointer; }
.filters button.on { color: var(--bg); background: var(--accent); border-color: var(--accent); }
#status { width: 8px; height: 8px; border-radius: 50%; background: #4ade80; }
#status.off { background: #ef4444; }
main { padding: 20px; display: grid; gap: 16px; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden;
  display: flex; flex-direction: column; animation: in .35s ease-out; }
.card.wide { grid-column: span 2; }
@keyframes in { from { opacity: 0; transform: translateY(6px); box-shadow: 0 0 0 2px var(--accent); } }
.media { background: #000; display: flex; align-items: center; justify-content: center; min-height: 120px; cursor: zoom-in; }
.media img, .media video { width: 100%; max-height: 360px; object-fit: contain; display: block; }
.media model-viewer { width: 100%; height: 300px; background: #111; }
.media iframe { width: 100%; height: 320px; border: 0; background: #fff; }
.media audio { width: calc(100% - 24px); margin: 40px 12px; }
.media pre { margin: 0; width: 100%; max-height: 320px; overflow: hidden; font: 12px/1.5 var(--mono); padding: 12px;
  background: #0d1117; -webkit-mask-image: linear-gradient(#000 80%, transparent); mask-image: linear-gradient(#000 80%, transparent); }
.media pre code.hljs { padding: 0; background: none; }
pre code { white-space: pre-wrap; overflow-wrap: anywhere; }
.media .file { color: var(--dim); font: 13px var(--mono); padding: 40px 12px; }
.meta { padding: 9px 12px; border-top: 1px solid var(--line); display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.name { font: 12px var(--mono); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.name a { color: inherit; text-decoration: none; }
.sub { color: var(--dim); font-size: 12px; display: flex; gap: 8px; }
.tag { text-transform: uppercase; letter-spacing: .04em; font-size: 10px; color: var(--accent); }
.fav { margin-left: auto; background: none; border: 0; padding: 0 2px; cursor: pointer; font-size: 16px; line-height: 1;
  color: var(--dim); }
.fav:hover { color: var(--text); }
.fav.on { color: #f43f5e; }
.card.favorite { border-color: color-mix(in srgb, #f43f5e 55%, var(--line)); }
.caption { color: var(--dim); font-size: 12px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.empty { grid-column: 1 / -1; text-align: center; color: var(--dim); padding: 80px 0; }
#box { position: fixed; inset: 0; z-index: 10; background: rgba(0,0,0,.92); display: none; flex-direction: column; }
#box.open { display: flex; }
#box .bar { display: flex; align-items: center; gap: 12px; padding: 10px 16px; color: var(--dim); font: 12px var(--mono); }
#box .bar span { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#box .bar button { background: none; border: 0; color: var(--text); font-size: 20px; cursor: pointer; }
#box .stage { flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; padding: 0 16px 16px; }
#box .stage img, #box .stage video { max-width: 100%; max-height: 100%; object-fit: contain; }
#box .stage iframe, #box .stage model-viewer { width: 100%; height: 100%; border: 0; background: #fff; }
#box .stage model-viewer { background: #111; }
#box .stage pre { margin: 0; width: 100%; max-width: 1100px; height: 100%; overflow: auto; font: 13px/1.55 var(--mono);
  background: #0d1117; padding: 16px; border-radius: var(--radius); }
#box .caption { -webkit-line-clamp: unset; padding: 0 16px 14px; max-width: 1100px; align-self: center; }
@media (max-width: 640px) { main { padding: 12px; grid-template-columns: 1fr; } .card.wide { grid-column: auto; } }
</style>
</head>
<body>
<header>
  <div id="status" title="live"></div>
  <h1 id="title">okonomi<small id="count"></small></h1>
  <div class="filters" id="filters"></div>
</header>
<main id="grid"><div class="empty">Waiting for outputs…</div></main>
<div id="box">
  <div class="bar"><span id="box-name"></span><button id="box-fav" class="fav" title="Favorite (f)">♥</button><a id="box-open" target="_blank" style="color:inherit">open ↗</a><button id="box-close" aria-label="Close">×</button></div>
  <div class="stage" id="box-stage"></div>
  <div class="caption" id="box-caption"></div>
</div>
<script>
const KINDS = ["all", "favorites", "image", "video", "audio", "3d", "html", "text", "file"];
const PREVIEW_BYTES = 8000, FULL_BYTES = 400000;
const grid = document.getElementById("grid");
const cards = new Map();
let items = [], filter = "all", order = "";
const pending = new Map();

const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const fmtSize = n => n == null ? "remote" : n < 1024 ? n + " B" : n < 1048576 ? (n / 1024).toFixed(1) + " KB" : (n / 1048576).toFixed(1) + " MB";
const fmtTime = t => t ? new Date(t * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}) : "";
const lang = name => (name.split(".").pop() || "").toLowerCase();

async function fetchText(src, bytes) {
  const r = await fetch(src, {headers: {Range: "bytes=0-" + (bytes - 1)}});
  return r.text();
}

function codeBlock(text, name) {
  const pre = document.createElement("pre"), code = document.createElement("code");
  code.textContent = text;
  const l = lang(name);
  if (window.hljs && hljs.getLanguage(l)) code.className = "language-" + l;
  pre.appendChild(code);
  if (window.hljs) hljs.highlightElement(code);
  return pre;
}

function mediaEl(it, full) {
  switch (it.kind) {
    case "image": { const e = document.createElement("img"); e.src = it.src; e.loading = "lazy"; e.alt = it.name; return e; }
    case "video": { const e = document.createElement("video"); e.src = it.src; e.controls = true; e.preload = "metadata"; e.loop = true; e.playsInline = true; if (!full) e.muted = true; return e; }
    case "audio": { const e = document.createElement("audio"); e.src = it.src; e.controls = true; e.preload = "metadata"; return e; }
    case "3d": { const e = document.createElement("model-viewer"); e.setAttribute("src", it.src); e.setAttribute("camera-controls", ""); e.setAttribute("auto-rotate", ""); return e; }
    case "html": { const e = document.createElement("iframe"); e.src = it.src; e.loading = "lazy"; return e; }
    case "text": {
      const holder = document.createElement("pre");
      fetchText(it.src, full ? FULL_BYTES : PREVIEW_BYTES).then(t => holder.replaceWith(codeBlock(t, it.name))).catch(() => {});
      return holder;
    }
    default: { const e = document.createElement("div"); e.className = "file"; e.textContent = "." + lang(it.name) + " file"; return e; }
  }
}

function card(it) {
  const c = document.createElement("article");
  c.className = "card" + (it.kind === "html" ? " wide" : "");
  const m = document.createElement("div");
  m.className = "media";
  m.appendChild(mediaEl(it, false));
  // Interactive media keeps its own clicks; everything else opens the lightbox.
  m.addEventListener("click", e => { if (!e.target.closest("video, audio, model-viewer")) openBox(it); });
  c.appendChild(m);
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.innerHTML = `<div class="name"><a href="${esc(it.src)}" target="_blank" title="${esc(it.name)}">${esc(it.name)}</a></div>
    <div class="sub"><span class="tag">${esc(it.kind)}</span><span>${fmtSize(it.size)}</span><span>${fmtTime(it.time)}</span>${it.model ? `<span>${esc(it.model)}</span>` : ""}<button class="fav" title="Favorite (f in lightbox)">♥</button></div>
    ${it.caption ? `<div class="caption" title="${esc(it.caption)}">${esc(it.caption)}</div>` : ""}`;
  c.appendChild(meta);
  meta.querySelector(".fav").addEventListener("click", () => toggleFavorite(it.id));
  paintFavorite(c, it.favorite);
  return c;
}

function paintFavorite(c, on) {
  c.classList.toggle("favorite", on);
  c.querySelector(".fav").classList.toggle("on", on);
}

async function toggleFavorite(id) {
  const it = items.find(x => x.id === id);
  if (!it) return;
  it.favorite = !it.favorite;
  pending.set(id, it.favorite);
  paintFavorite(cards.get(id), it.favorite);
  if (boxItem && boxItem.id === id) paintBoxFavorite();
  renderFilters();
  const r = await fetch("/api/favorite", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ref: it.ref, favorite: it.favorite})}).catch(() => null);
  pending.delete(id);
  const cur = items.find(x => x.id === id) || it;
  if (!r || !r.ok) cur.favorite = !it.favorite;
  paintFavorite(cards.get(id), cur.favorite);
  paintBoxFavorite();
  renderFilters();
  render();
}

function render() {
  const shown = items.filter(it => filter === "favorites" ? it.favorite : !it.elsewhere && (filter === "all" || it.kind === filter)).reverse();
  const key = filter + "|" + shown.map(it => it.id).join("|");
  for (const it of items) if (cards.has(it.id)) paintFavorite(cards.get(it.id), it.favorite);
  if (key === order) return;
  order = key;
  const ids = new Set(items.map(it => it.id));
  for (const id of cards.keys()) if (!ids.has(id)) cards.delete(id);
  if (!shown.length) { grid.innerHTML = '<div class="empty">' + (items.length ? "Nothing of this kind yet." : "Waiting for outputs…") + "</div>"; return; }
  grid.querySelector(".empty")?.remove();
  for (const el of [...grid.children]) if (!shown.some(it => cards.get(it.id) === el)) el.remove();
  shown.forEach((it, i) => {
    if (!cards.has(it.id)) cards.set(it.id, card(it));
    const el = cards.get(it.id);
    if (grid.children[i] !== el) grid.insertBefore(el, grid.children[i] || null);
  });
}

function renderFilters() {
  const counts = {all: 0};
  for (const it of items) {
    if (it.favorite) counts.favorites = (counts.favorites || 0) + 1;
    if (it.elsewhere) continue;
    counts.all += 1;
    counts[it.kind] = (counts[it.kind] || 0) + 1;
  }
  document.getElementById("filters").innerHTML = KINDS.filter(k => counts[k])
    .map(k => `<button data-k="${k}" class="${k === filter ? "on" : ""}">${k === "favorites" ? "♥" : k} ${counts[k]}</button>`).join("");
}

document.getElementById("filters").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  filter = b.dataset.k;
  renderFilters();
  render();
});

async function poll() {
  try {
    const r = await fetch("/api/items", {cache: "no-store"});
    const data = await r.json();
    items = data.items;
    for (const it of items) if (pending.has(it.id)) it.favorite = pending.get(it.id);
    document.title = data.title + " · okonomi";
    document.getElementById("title").firstChild.textContent = data.title;
    document.getElementById("count").textContent = items.length + " outputs · " + data.dir;
    document.getElementById("status").className = "";
    renderFilters();
    render();
  } catch (e) {
    document.getElementById("status").className = "off";
  }
  setTimeout(poll, 1500);
}

const box = document.getElementById("box"), stage = document.getElementById("box-stage");
let boxItem = null;
function paintBoxFavorite() { document.getElementById("box-fav").classList.toggle("on", !!items.find(x => x.id === boxItem?.id)?.favorite); }
document.getElementById("box-fav").onclick = () => boxItem && toggleFavorite(boxItem.id);
function openBox(it) {
  boxItem = it;
  paintBoxFavorite();
  stage.replaceChildren(mediaEl(it, true));
  document.getElementById("box-name").textContent = it.name;
  document.getElementById("box-open").href = it.src;
  document.getElementById("box-caption").textContent = [it.model, it.caption].filter(Boolean).join(" — ");
  box.classList.add("open");
}
function closeBox() { box.classList.remove("open"); stage.replaceChildren(); boxItem = null; }
document.getElementById("box-close").onclick = closeBox;
box.addEventListener("click", e => { if (e.target === box || e.target === stage) closeBox(); });
document.addEventListener("keydown", e => {
  if (e.key === "Escape") closeBox();
  if (e.key === "f" && boxItem) toggleFavorite(boxItem.id);
});

poll();
</script>
</body>
</html>
"""
