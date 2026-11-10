"""Every write into the mind wiki goes through ``remember``: one schema,
no silent clobbering, a git commit per write, and the file tools refuse
the directory."""

import subprocess
from types import SimpleNamespace

from okonomi import session as session_store
from okonomi.tools.files import FileTools
from okonomi.tools.sessions import SessionsTools


def _tools(session_id: str = "") -> SessionsTools:
    return SessionsTools((lambda: SimpleNamespace(session_id=session_id)) if session_id else None)


def test_remember_writes_the_schema_that_readers_use(store):
    out = _tools("s1").remember(
        title="Lofi Treatment — VHS Still", body="Chroma bleed, scanlines.", tags=["lofi", "vhs"], related=["grain"]
    )

    page = (session_store.OKONOMI_MIND_DIR / "lofi-treatment-vhs-still.md").read_text()
    assert out["slug"] == "lofi-treatment-vhs-still" and out["created"], out
    assert page.startswith("---\ntitle: Lofi Treatment — VHS Still\ntags: [lofi, vhs]\ncreated: "), page
    assert "related: [[grain]]\nsessions: [s1]\n---\n\n# Lofi Treatment — VHS Still\n\nChroma bleed" in page
    (hit,) = [h for h in session_store.browse() if h["source"] == "mind"]
    assert (hit["source"], hit["session_title"], hit["snippet"]) == ("mind", "Lofi Treatment — VHS Still", "[lofi, vhs]")


def test_every_remember_is_a_commit_and_accumulates_sessions(store):
    mind = session_store.OKONOMI_MIND_DIR
    _tools("s1").remember(title="Agni", body="v1")
    _tools("s2").remember(title="Agni", body="v2", slug="agni", update=True)
    _tools("s2").remember(title="Agni", body="v3", slug="agni", update=True)

    assert "sessions: [s1, s2]\n" in (mind / "agni.md").read_text(), "each writing session is listed once"
    log = subprocess.run(["git", "log", "--format=%s"], cwd=mind, capture_output=True, text=True, check=True).stdout
    assert log.splitlines() == ["update agni (s2)", "update agni (s2)", "create agni (s1)"], log
    shown = subprocess.run(["git", "show", "HEAD~2:agni.md"], cwd=mind, capture_output=True, text=True, check=True)
    assert "v1" in shown.stdout, "page history is recoverable from git"


def test_existing_page_needs_update_and_keeps_created(store):
    tools = _tools()
    tools.remember(title="Agni", body="v1")
    page = session_store.OKONOMI_MIND_DIR / "agni.md"
    page.write_text(page.read_text().replace(page.read_text().split("created: ")[1][:10], "2020-01-01"))

    refused = tools.remember(title="Agni", body="v2")
    assert "error" in refused and "v1" in refused["current"], refused
    assert "v1" in page.read_text(), "a refused write must not touch the page"

    tools.remember(title="Agni", body="v2", slug="agni", update=True)
    assert "created: 2020-01-01" in page.read_text() and "v2" in page.read_text(), page.read_text()


def test_file_tools_refuse_mind_directory(store, tmp_path):
    mind = session_store.OKONOMI_MIND_DIR
    (mind / "x.md").write_text("hi")
    tools = FileTools()

    assert "remember" in tools.write_file(str(mind / "y.md"), "new")["error"]
    assert "remember" in tools.edit_file(str(mind / "x.md"), "hi", "bye")["error"]
    assert "error" not in tools.write_file(str(tmp_path / "elsewhere.md"), "ok")
    assert tools.read_file(str(mind / "x.md"))["content"].endswith("hi"), "mind pages stay readable"
