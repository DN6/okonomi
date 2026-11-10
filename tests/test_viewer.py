"""The media viewer serves every file in the session's artifacts dir (plus
tracked artifacts elsewhere), classified by kind, and refuses anything else."""

from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import quote

import httpx
import pytest

from okonomi.session import Artifact
from okonomi.tools import viewer as viewer_mod
from okonomi.tools.viewer import ViewerTools


@pytest.fixture
def setup(store, tmp_path, monkeypatch):
    monkeypatch.setattr(viewer_mod.webbrowser, "open", lambda url: True)
    session_dir = tmp_path / "session"
    artifacts_dir = session_dir / "artifacts"
    (artifacts_dir / "images").mkdir(parents=True)
    (session_dir / "scratch").mkdir()
    (artifacts_dir / "images" / "cat.png").write_bytes(b"\x89PNG" + b"0" * 100)
    (artifacts_dir / "clip.mp4").write_bytes(b"v" * 1000)
    (artifacts_dir / "song.wav").write_bytes(b"a" * 10)
    (artifacts_dir / "page.html").write_text("<p>hi</p>")
    (artifacts_dir / "mesh.glb").write_bytes(b"glTF")
    (artifacts_dir / ".hidden").write_text("x")
    (session_dir / "scratch" / "make.py").write_text("print('hi')\n")
    (session_dir / "traces.jsonl").write_text("{}")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    tracked = outside / "ref.jpg"
    tracked.write_bytes(b"jpg")
    secret = outside / "secret.txt"
    secret.write_text("nope")
    artifacts = [
        Artifact(artifact_type="image", path=str(artifacts_dir / "images" / "cat.png"), prompt="a cat", model_id="flux"),
        Artifact(artifact_type="image", path=str(tracked)),
        Artifact(artifact_type="video", url="https://example.com/out/abc123"),
    ]
    session = SimpleNamespace(
        session_dir=session_dir,
        artifacts_dir=artifacts_dir,
        session_id="s1",
        title="Cats",
        state=SimpleNamespace(artifacts=artifacts),
        append_stream_events=lambda events: None,
    )
    tools = ViewerTools(lambda: session)
    result = tools.open_viewer()
    yield result["url"].rstrip("/"), session_dir, tracked, secret
    tools._server.shutdown()


def test_items_cover_artifacts_and_tracked_files(setup):
    base, session_dir, tracked, _ = setup
    data = httpx.get(f"{base}/api/items").json()
    kinds = {i["name"]: i["kind"] for i in data["items"]}
    assert kinds == {
        "images/cat.png": "image",
        "clip.mp4": "video",
        "song.wav": "audio",
        "page.html": "html",
        "mesh.glb": "3d",
        "ref.jpg": "image",
        "abc123": "video",
    }, f"scratch, traces and hidden files must not be listed: {kinds}"
    cat = next(i for i in data["items"] if i["name"] == "images/cat.png")
    assert (cat["caption"], cat["model"]) == ("a cat", "flux"), cat
    assert data["title"] == "Cats"


def test_serves_allowed_files_and_refuses_others(setup):
    base, session_dir, tracked, secret = setup
    assert httpx.get(f"{base}/fs{quote(str(session_dir / 'artifacts' / 'page.html'))}").text == "<p>hi</p>"
    assert httpx.get(f"{base}/fs{quote(str(tracked))}").content == b"jpg"
    assert httpx.get(f"{base}/fs{quote(str(secret))}").status_code == 404
    assert httpx.get(f"{base}/fs{quote(str(session_dir / 'scratch' / 'make.py'))}").status_code == 404
    assert httpx.get(f"{base}/fs{quote(str(session_dir / 'traces.jsonl'))}").status_code == 404
    escape = f"{base}/fs{quote(str(session_dir))}/..%2Felsewhere%2Fsecret.txt"
    assert httpx.get(escape).status_code == 404


def test_favorite_toggle_roundtrip(setup, tmp_path):
    base, session_dir, _, secret = setup
    cat = str(session_dir / "artifacts" / "images" / "cat.png")

    def favorites() -> set[str]:
        return {i["ref"] for i in httpx.get(f"{base}/api/items").json()["items"] if i["favorite"]}

    assert favorites() == set()
    assert httpx.post(f"{base}/api/favorite", json={"ref": cat, "favorite": True}).status_code == 200
    assert favorites() == {cat}
    links = list((tmp_path / "favorites").iterdir())
    assert [str(p.resolve()) for p in links] == [cat], links

    assert httpx.post(f"{base}/api/favorite", json={"ref": cat, "favorite": False}).status_code == 200
    assert favorites() == set()
    assert list((tmp_path / "favorites").iterdir()) == []

    assert httpx.post(f"{base}/api/favorite", json={"ref": str(secret), "favorite": True}).status_code == 404
    assert httpx.post(f"{base}/api/favorite", json={"ref": "https://example.com/out/abc123", "favorite": True}).status_code == 200
    assert "https://example.com/out/abc123" in favorites()
