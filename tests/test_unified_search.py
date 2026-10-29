"""Unified search: one query reaches sessions, conversations, assets and
the mind wiki, and no source is crowded out by another's raw BM25."""

from types import SimpleNamespace

import pytest

from okonomi import db
from okonomi import session as session_store
from okonomi.events import EventEmitter, EventType
from okonomi.tools.sessions import SessionsTools


SID = "2026-10-20_firetest"


@pytest.fixture
def store(tmp_path, monkeypatch):
    conn = db.open_db(tmp_path / "test.db")
    monkeypatch.setattr(db, "_connection", conn)
    mind = tmp_path / "mind"
    mind.mkdir()
    monkeypatch.setattr(session_store, "OKONOMI_MIND_DIR", mind)
    db.upsert_session(conn, {"session_id": SID, "title": "Fire deities moodboard"})
    yield conn, mind
    conn.close()


def _talk(conn, n: int) -> None:
    rows = [
        {
            "type": EventType.ASSISTANT_MESSAGE,
            "payload": "{}",
            "content": f"fire deities fire deities across cultures, take {i}",
            "timestamp": "",
        }
        for i in range(n)
    ]
    db.append_events(conn, SID, rows)


def test_mind_page_surfaces_despite_many_conversation_hits(store):
    conn, mind = store
    _talk(conn, 20)
    (mind / "theechamundi-fire-comparative.md").write_text(
        "---\ntitle: Theechamundi & the World of Fire Ritual\n---\n\nA comparison of fire deities in other cultures.\n"
    )
    (mind / "rococo.md").write_text("---\ntitle: Rococo\n---\n\nShell motifs and pastel fire.\n")

    results = SessionsTools().recall(query="fire deities", limit=8)["results"]

    mind_hits = [h for h in results if h["source"] == "mind"]
    assert mind_hits, f"mind page missing from {[h['source'] for h in results]}"
    assert mind_hits[0]["session_title"] == "Theechamundi & the World of Fire Ritual", mind_hits[0]
    assert mind_hits[0]["session_id"] == "", "mind hits belong to no session"
    assert mind_hits[0]["path"].endswith("theechamundi-fire-comparative.md"), mind_hits[0]


def test_every_hit_has_the_same_keys(store):
    conn, mind = store
    _talk(conn, 2)
    (mind / "page.md").write_text("fire deities\n")

    hits = session_store.search("fire", limit=20)

    assert {h["source"] for h in hits} == {"event", "session", "mind"}, hits
    for h in hits:
        for key in ("source", "session_id", "session_title", "snippet", "timestamp", "score"):
            assert key in h, f"{h['source']} hit is missing {key!r}"


def _artifact(conn, path: str, prompt: str) -> None:
    db.replace_artifacts(conn, SID, [{"artifact_type": "image", "path": path, "prompt": prompt}])


def test_favorite_identity_matches_across_picker_and_search(store, tmp_path, monkeypatch):
    conn, _ = store
    from okonomi.tui import assets

    monkeypatch.setattr(assets, "FAVORITES_DIR", tmp_path / "favorites")
    img = tmp_path / "assets" / "agni.png"
    img.parent.mkdir()
    img.write_bytes(b"png")
    _artifact(conn, str(img), "agni riding a ram, fire deity")
    agent = SimpleNamespace(state=SimpleNamespace(artifacts=[]), session=SimpleNamespace(session_id=SID))

    assets._favorite_paths([str(tmp_path / "assets" / ".." / "assets" / "agni.png")], agent, EventEmitter())

    hit = next(h for h in session_store.search("agni") if h["source"] == "artifact")
    assert hit["favorite"] is True, hit
    (fav,) = session_store.favorite_hits()
    assert fav["session_id"] == SID and "agni" in fav["snippet"], "an external favorite should join back to its artifact"


def test_half_typed_word_survives_the_stemmer(store):
    conn, _ = store
    db.append_events(
        conn, SID, [{"type": EventType.USER_MESSAGE, "payload": "{}", "content": "theyyam rococo portraits", "timestamp": ""}]
    )
    from okonomi.tui.pickers import _live_query

    hits = session_store.search_events(_live_query("rococo theyy"))

    assert len(hits) == 1, "porter stems the raw prefix `theyy*` to `theyi*`; it must be expanded against the index instead"


def test_short_prefix_not_limited_by_expansion_cap(store, monkeypatch):
    conn, _ = store
    monkeypatch.setattr(db, "_PREFIX_EXPANSIONS", 1)
    db.append_events(
        conn,
        SID,
        [
            {"type": EventType.USER_MESSAGE, "payload": "{}", "content": text, "timestamp": ""}
            for text in ("stone stone", "stone", "still")
        ],
    )

    hits = session_store.search_events("st*")

    assert len(hits) == 3, "a capped expansion must not hide `still` from a two-letter prefix"
