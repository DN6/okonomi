"""Every write into the mind wiki goes through ``remember``: one schema,
no silent clobbering, and the file tools refuse the directory."""

import pytest

from okonomi import session as session_store
from okonomi.tools import files as files_mod
from okonomi.tools.files import FileTools
from okonomi.tools.sessions import SessionsTools


@pytest.fixture
def mind(tmp_path, monkeypatch):
    d = tmp_path / "mind"
    monkeypatch.setattr(session_store, "OKONOMI_MIND_DIR", d)
    monkeypatch.setattr(files_mod, "OKONOMI_MIND_DIR", d)
    return d


def test_remember_writes_the_schema_that_readers_use(mind):
    out = SessionsTools().remember(
        title="Lofi Treatment — VHS Still", body="Chroma bleed, scanlines.", tags=["lofi", "vhs"], related=["grain"]
    )

    page = (mind / "lofi-treatment-vhs-still.md").read_text()
    assert out["slug"] == "lofi-treatment-vhs-still" and out["created"], out
    assert page.startswith("---\ntitle: Lofi Treatment — VHS Still\ntags: [lofi, vhs]\ncreated: "), page
    assert "related: [[grain]]\n---\n\n# Lofi Treatment — VHS Still\n\nChroma bleed" in page, page
    (hit,) = [h for h in session_store.browse() if h["source"] == "mind"]
    assert (hit["source"], hit["session_title"], hit["snippet"]) == ("mind", "Lofi Treatment — VHS Still", "[lofi, vhs]")


def test_existing_page_needs_update_and_keeps_created(mind):
    tools = SessionsTools()
    tools.remember(title="Agni", body="v1")
    page = mind / "agni.md"
    page.write_text(page.read_text().replace(page.read_text().split("created: ")[1][:10], "2020-01-01"))

    refused = tools.remember(title="Agni", body="v2")
    assert "error" in refused and "v1" in refused["current"], refused
    assert "v1" in page.read_text(), "a refused write must not touch the page"

    tools.remember(title="Agni", body="v2", slug="agni", update=True)
    assert "created: 2020-01-01" in page.read_text() and "v2" in page.read_text(), page.read_text()


def test_file_tools_refuse_mind_directory(mind, tmp_path):
    mind.mkdir()
    (mind / "x.md").write_text("hi")
    tools = FileTools()

    assert "remember" in tools.write_file(str(mind / "y.md"), "new")["error"]
    assert "remember" in tools.edit_file(str(mind / "x.md"), "hi", "bye")["error"]
    assert "error" not in tools.write_file(str(tmp_path / "elsewhere.md"), "ok")
    assert tools.read_file(str(mind / "x.md"))["content"].endswith("hi"), "mind pages stay readable"
