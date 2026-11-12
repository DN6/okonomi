"""The store: artifacts and references, the session event tree, unified
search over everything, and the mind wiki."""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from okonomi import db
from okonomi import session as session_store
from okonomi.agent.agent import AgentDependencies, store_references
from okonomi.context import BRANCH_SUMMARY_PREFIX
from okonomi.events import CONVERSATION_TYPES, EventType
from okonomi.tools.files import FileTools
from okonomi.tools.sessions import SessionsTools
from okonomi.tui import artifacts
from okonomi.tui.pickers import _live_query, _tree_rows, apply_filters, parse_query


ARTIFACT_SID = "2026-11-10_artifacts"
TREE_SID = "2026-07-30_treetest"
FIRE_SID = "2026-10-20_firetest"
SESSION = SimpleNamespace(session_id=FIRE_SID, append_stream_events=lambda events: None)



def test_artifact_ids_survive_saves_and_duplicates_are_migrated_away(store):
    db.upsert_session(store, {"session_id": ARTIFACT_SID})
    db.upsert_artifacts(store, ARTIFACT_SID, [{"artifact_type": "image", "path": "/a.png", "prompt": "v1"}])
    first_id = store.execute("SELECT id FROM artifacts").fetchone()["id"]
    db.upsert_artifacts(
        store, ARTIFACT_SID, [{"artifact_type": "image", "path": "/a.png", "prompt": "v2", "tool_call_id": "c1"}]
    )

    (row,) = store.execute("SELECT id, prompt, tool_call_id FROM artifacts").fetchall()
    assert (row["id"], row["prompt"], row["tool_call_id"]) == (first_id, "v2", "c1"), "same ref → same row, updated"

    store.execute("INSERT INTO artifacts (session_id, path) VALUES (?, '/a.png')", (ARTIFACT_SID,))
    db._migrate(store)
    assert store.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 1, "legacy duplicate rows are removed"


def test_sweep_tracks_outputs_with_their_tool_call(okonomi_agent):
    deps = AgentDependencies(session=okonomi_agent.session)
    deps.session.artifacts_dir.mkdir(parents=True)
    (deps.session.artifacts_dir / "noir_brief.md").write_text("# Noir\n")
    (deps.session.artifacts_dir / "notes.json").write_text("{}")
    (deps.session.scratch_dir / "pylibs" / "scipy").mkdir(parents=True)
    (deps.session.scratch_dir / "pylibs" / "scipy" / "test.png").write_bytes(b"png")
    (deps.session.session_dir / "stray.png").write_bytes(b"png")

    assert okonomi_agent._sync_session_artifacts(deps, tool_call_id="c7") == 1, "only media and briefs under artifacts/"
    (brief,) = deps.state.artifacts
    assert (brief.artifact_type, brief.tool_call_id, Path(brief.path).name) == ("research", "c7", "noir_brief.md")


def test_user_supplied_images_are_kept_as_references(okonomi_agent, tmp_path):
    session = okonomi_agent.session
    session.artifacts_dir.mkdir(parents=True)
    supplied = tmp_path / "moodboard.png"
    supplied.write_bytes(b"png-a")
    generated = session.artifacts_dir / "fox_v1_2.png"
    generated.write_bytes(b"png-b")

    added = store_references(session, [str(supplied), str(generated), str(supplied)])

    (ref,) = added
    assert ref.tool_name == "user" and ref.path == str(session.references_dir / "moodboard.png"), ref
    assert Path(ref.path).read_bytes() == b"png-a"
    assert session.state.artifacts == [ref], "an output the user pointed at is a pick, not a reference"
    assert store_references(session, [str(supplied)]) == [], "a reference is kept once"



@pytest.fixture
def conn(store):
    db.upsert_session(store, {"session_id": TREE_SID})
    db.upsert_session(store, {"session_id": FIRE_SID, "title": "Fire deities moodboard"})
    return store


def _row(text: str, type_: EventType = EventType.USER_MESSAGE) -> dict:
    return {"type": type_, "payload": "{}", "content": text, "timestamp": ""}


def _active_ids(conn) -> list[int]:
    return [r["event_id"] for r in db.fetch_events(conn, TREE_SID)]


def test_checkout_activates_the_root_path_and_later_saves_chain_from_its_leaf(conn):
    # ids 0-3 linear, rewind at 2, ids 4-5 appended → branches 0-1-2-3 (inactive) and 0-1-4-5 (active).
    db.append_events(conn, TREE_SID, [_row(f"e{i}") for i in range(4)])
    db.deactivate_events(conn, TREE_SID, from_event_id=2)
    db.append_events(conn, TREE_SID, [_row("e4"), _row("e5")])
    assert _active_ids(conn) == [0, 1, 4, 5], f"fixture broken: active={_active_ids(conn)}"

    n = db.checkout_events(conn, TREE_SID, 3)
    assert n == 4, f"expected 4 events on path root→3, got {n}"
    assert _active_ids(conn) == [0, 1, 2, 3], f"active after checkout: {_active_ids(conn)}"
    parents = {r["event_id"]: r["parent_id"] for r in db.fetch_events(conn, TREE_SID, active_only=False)}
    assert parents[4] == 1 and parents[2] == 1, f"fork edges changed by checkout: {parents}"

    db.append_events(conn, TREE_SID, [_row("e6")])
    new = next(r for r in db.fetch_events(conn, TREE_SID, active_only=False) if r["event_id"] == 6)
    assert new["parent_id"] == 3, f"append after checkout chained from {new['parent_id']}, expected leaf 3"
    assert _active_ids(conn) == [0, 1, 2, 3, 6]

    # A checkout followed by reload keeps _sync_conversation's count heuristic
    # consistent: the next save appends only the new tail from the checked-out leaf.
    db.checkout_events(conn, TREE_SID, None)
    history = [
        ModelRequest(parts=[UserPromptPart(content="ask one")]),
        ModelResponse(parts=[TextPart(content="answer one")]),
        ModelRequest(parts=[UserPromptPart(content="ask two")]),
        ModelResponse(parts=[TextPart(content="answer two")]),
    ]
    session = session_store.Session(session_id=TREE_SID, state=SimpleNamespace(message_history=history, artifacts=[]))
    session.save()
    assert _active_ids(conn) == [7, 8, 9, 10]

    with db.transaction(conn):
        db.checkout_events(conn, TREE_SID, 8)
    loaded = session.load_messages()
    assert len(loaded) == 2, f"active path should hold 2 messages, got {len(loaded)}"
    summary = ModelRequest(parts=[UserPromptPart(content=f"{BRANCH_SUMMARY_PREFIX}\nabandoned work")])
    session.state.message_history = loaded + [summary]
    session.save()

    rows = db.fetch_events(conn, TREE_SID, types=CONVERSATION_TYPES)
    assert [r["event_id"] for r in rows] == [7, 8, 11], f"active conversation: {[r['event_id'] for r in rows]}"
    assert rows[-1]["parent_id"] == 8, f"summary row chained from {rows[-1]['parent_id']}, expected checked-out leaf 8"
    assert rows[-1]["type"] == EventType.USER_MESSAGE.value
    inactive = [r["event_id"] for r in db.fetch_events(conn, TREE_SID, active_only=False) if not r["active"]]
    assert inactive == [0, 1, 2, 3, 4, 5, 6, 9, 10], f"old branches must stay in the DAG inactive, got {inactive}"


def _user_payload(text: str) -> str:
    return json.dumps({"parts": [{"part_kind": "user-prompt", "content": text}]})


def _ev(eid, parent, type_, content, active, payload="{}"):
    return {
        "event_id": eid,
        "parent_id": parent,
        "type": type_.value,
        "content": content,
        "active": int(active),
        "payload": payload,
    }


def test_tree_rows_forked_session():
    u, a = EventType.USER_MESSAGE, EventType.ASSISTANT_MESSAGE
    events = [
        # root A: fork at node 1 (children 2 and 4), all abandoned
        _ev(0, None, u, "first ask", 0, _user_payload("first ask")),
        _ev(1, 0, a, "answer 1", 0),
        _ev(2, 1, u, "try redis", 0, _user_payload("try redis")),
        _ev(3, 2, a, "redis done", 0),
        _ev(4, 1, u, "try sqlite", 0, _user_payload("try sqlite")),
        _ev(5, 4, a, "sqlite done", 0),
        # hidden tool event on the active-root chain — must not become a row
        _ev(6, None, EventType.USER_MESSAGE, "post-clear ask", 1, _user_payload("post-clear ask")),
        _ev(7, 6, EventType.TOOL_CALL, "tool", 1),
        _ev(8, 7, a, "fresh answer", 1),
    ]
    rows = _tree_rows(events)

    assert [r["event_id"] for r in rows] == [0, 1, 2, 3, 4, 5, 6, 8], f"row order: {[r['event_id'] for r in rows]}"
    # No active child under node 1, so the newest (4) is the trunk and 2 hangs.
    guides = [r["guide"] for r in rows]
    assert guides == ["", "", "├── ", "│   ", "", "", "", ""], f"guides: {guides}"
    assert [r["event_id"] for r in rows if r["branch"]] == [2], "only the hanging branch has a fold head"
    heads = [r["heads"] for r in rows]
    assert heads == [(), (), (), (2,), (), (), (), ()], f"enclosing heads: {heads}"

    by_id = {r["event_id"]: r for r in rows}
    assert by_id[0]["checkout_target"] is None, "root user row must check out to empty path"
    assert by_id[2]["checkout_target"] == 1 and by_id[4]["checkout_target"] == 1
    assert by_id[3]["checkout_target"] == 3, "assistant row checks out to itself"
    assert by_id[2]["prefill"] == "try redis"
    assert by_id[3]["prefill"] == ""
    assert by_id[8]["is_current"], "highest-id active message is the current leaf"
    assert not by_id[5]["is_current"]
    assert not by_id[5]["active"] and by_id[6]["active"]




def _talk(conn, texts: list[str], type_: EventType = EventType.ASSISTANT_MESSAGE) -> None:
    db.append_events(conn, FIRE_SID, [{"type": type_, "payload": "{}", "content": t, "timestamp": ""} for t in texts])


def test_mind_pages_are_indexed_from_files_and_surface_beside_conversations(conn):
    mind = session_store.OKONOMI_MIND_DIR
    page = mind / "agni.md"
    page.write_text("---\ntitle: Agni\ntags: [fire]\n---\n\nVedic fire deity.\n")
    (hit,) = db.search_mind_fts(conn, mind, "vedic")
    assert hit["title"] == "Agni" and ">>>Vedic<<< fire deity." in hit["snippet"], hit

    page.write_text("---\ntitle: Agni\ntags: [fire]\n---\n\nRides a ram.\n")
    os.utime(page, (page.stat().st_atime, page.stat().st_mtime + 5))
    assert db.search_mind_fts(conn, mind, "vedic") == [], "a hand edit drops the stale text"
    assert [h["title"] for h in db.search_mind_fts(conn, mind, "ram")] == ["Agni"]
    page.unlink()
    assert db.list_mind_pages(conn, mind) == [], "a deleted page leaves the index"

    _talk(conn, [f"fire deities fire deities across cultures, take {i}" for i in range(20)])
    (mind / "theechamundi-fire-comparative.md").write_text(
        "---\ntitle: Theechamundi & the World of Fire Ritual\n---\n\nA comparison of fire deities in other cultures.\n"
    )
    results = SessionsTools().recall(query="fire deities", limit=8)["results"]
    mind_hits = [h for h in results if h["source"] == "mind"]
    assert mind_hits, f"mind page missing from {[h['source'] for h in results]}"
    assert mind_hits[0]["session_title"] == "Theechamundi & the World of Fire Ritual", mind_hits[0]
    assert mind_hits[0]["session_id"] == "", "mind hits belong to no session"
    assert mind_hits[0]["path"].endswith("theechamundi-fire-comparative.md"), mind_hits[0]

    hits = session_store.search("fire", limit=20)
    assert {h["source"] for h in hits} == {"event", "session", "mind"}, hits
    for h in hits:
        for key in ("source", "session_id", "session_title", "snippet", "timestamp", "score"):
            assert key in h, f"{h['source']} hit is missing {key!r}"


def test_prefix_queries_survive_the_stemmer_and_the_expansion_cap(conn, monkeypatch):
    _talk(conn, ["theyyam rococo portraits"], EventType.USER_MESSAGE)
    hits = session_store.search_events(_live_query("rococo theyy"))
    assert len(hits) == 1, "porter stems the raw prefix `theyy*` to `theyi*`; it must be expanded against the index instead"

    monkeypatch.setattr(db, "_PREFIX_EXPANSIONS", 1)
    _talk(conn, ["stone stone", "stone", "still"], EventType.USER_MESSAGE)
    hits = session_store.search_events("st*")
    assert len(hits) == 3, "a capped expansion must not hide `still` from a two-letter prefix"


def test_favorites_join_their_artifacts_and_browse_opens_on_this_session(conn, tmp_path):
    first, second = tmp_path / "out" / "agni_v1.png", tmp_path / "out" / "agni_v2.png"
    first.parent.mkdir()
    first.write_bytes(b"png")
    second.write_bytes(b"png")
    db.upsert_artifacts(
        conn,
        FIRE_SID,
        [
            {"artifact_type": "image", "path": str(first), "prompt": "agni riding a ram, fire deity", "timestamp": "2026-07-30T10:00:00"},
            {"artifact_type": "image", "path": str(second), "prompt": "agni, round two", "timestamp": "2026-07-30T10:05:00"},
        ],
    )
    artifacts.set_favorite(str(tmp_path / "out" / ".." / "out" / "agni_v1.png"), artifact_type="image", favorite=True, session=SESSION)

    hit = next(h for h in session_store.search("ram") if h["source"] == "artifact")
    assert hit["favorite"] is True, hit
    (fav,) = session_store.favorite_hits()
    assert fav["session_id"] == FIRE_SID and "agni" in fav["snippet"], "a favorite by path must join back to its artifact"

    own = [h for h in session_store.browse(session_id=FIRE_SID) if h["source"] == "artifact"]
    assert [h["path"] for h in own] == [str(second), str(first)], "this session's artifacts, newest first, favorite not repeated"
    assert [h["favorite"] for h in own] == [False, True], own
    assert session_store.browse()[0]["path"] == str(first), "without a session the view still opens on favorites"


def test_find_qualifiers_split_from_text_and_filter_hits():
    assert parse_query("fox type:artifact kind:image") == ("fox", {"type": "artifact", "kind": "image"})
    assert parse_query("is:favorite")[1] == {"is": "favorite", "type": "artifact"}, "favorite implies artifacts"
    assert parse_query("note:self 10:30")[0] == "note:self 10:30", "unknown keys stay free text"

    hits = [
        {"source": "artifact", "artifact_type": "image", "favorite": True, "session_id": "2026-07-30_a", "session_title": "Fox logo"},
        {"source": "artifact", "artifact_type": "video", "favorite": False, "session_id": "2026-07-30_a", "session_title": "Fox logo"},
        {"source": "session", "artifact_type": "", "favorite": False, "session_id": "2026-07-31_b", "session_title": "Owl poster"},
    ]
    assert [h["artifact_type"] for h in apply_filters(hits, {"type": "artifact"})] == ["image", "video"]
    assert [h["artifact_type"] for h in apply_filters(hits, {"kind": "video"})] == ["video"]
    assert [h["source"] for h in apply_filters(hits, {"is": "favorite"})] == ["artifact"]
    assert [h["session_id"] for h in apply_filters(hits, {"session": "owl"})] == ["2026-07-31_b"]
    assert [h["session_id"] for h in apply_filters(hits, {"session": "2026-07-30"})] == ["2026-07-30_a"] * 2


def _tools(session_id: str = "") -> SessionsTools:
    return SessionsTools((lambda: SimpleNamespace(session_id=session_id)) if session_id else None)


def test_remember_writes_one_schema_refuses_clobbering_and_commits_each_write(store):
    mind = session_store.OKONOMI_MIND_DIR
    out = _tools("s1").remember(
        title="Lofi Treatment — VHS Still", body="Chroma bleed, scanlines.", tags=["lofi", "vhs"], related=["grain"]
    )
    page = (mind / "lofi-treatment-vhs-still.md").read_text()
    assert out["slug"] == "lofi-treatment-vhs-still" and out["created"], out
    assert page.startswith("---\ntitle: Lofi Treatment — VHS Still\ntags: [lofi, vhs]\ncreated: "), page
    assert "related: [[grain]]\nsessions: [s1]\n---\n\n# Lofi Treatment — VHS Still\n\nChroma bleed" in page
    (hit,) = [h for h in session_store.browse() if h["source"] == "mind"]
    assert (hit["source"], hit["session_title"], hit["snippet"]) == ("mind", "Lofi Treatment — VHS Still", "[lofi, vhs]")

    _tools("s1").remember(title="Agni", body="v1")
    agni = mind / "agni.md"
    agni.write_text(agni.read_text().replace(agni.read_text().split("created: ")[1][:10], "2020-01-01"))
    refused = _tools("s2").remember(title="Agni", body="v2")
    assert "error" in refused and "v1" in refused["current"], refused
    assert "v1" in agni.read_text(), "a refused write must not touch the page"

    _tools("s2").remember(title="Agni", body="v2", slug="agni", update=True)
    _tools("s2").remember(title="Agni", body="v3", slug="agni", update=True)
    text = agni.read_text()
    assert "created: 2020-01-01" in text and "v3" in text, text
    assert "sessions: [s1, s2]\n" in text, "each writing session is listed once"
    log = subprocess.run(["git", "log", "--format=%s"], cwd=mind, capture_output=True, text=True, check=True).stdout
    assert log.splitlines() == [
        "update agni (s2)",
        "update agni (s2)",
        "create agni (s1)",
        "create lofi-treatment-vhs-still (s1)",
    ], log
    shown = subprocess.run(["git", "show", "HEAD~2:agni.md"], cwd=mind, capture_output=True, text=True, check=True)
    assert "v1" in shown.stdout, "page history is recoverable from git"


def test_file_tools_refuse_mind_directory(store, tmp_path):
    mind = session_store.OKONOMI_MIND_DIR
    (mind / "x.md").write_text("hi")
    tools = FileTools()

    assert "remember" in tools.write_file(str(mind / "y.md"), "new")["error"]
    assert "remember" in tools.edit_file(str(mind / "x.md"), "hi", "bye")["error"]
    assert "error" not in tools.write_file(str(tmp_path / "elsewhere.md"), "ok")
    assert tools.read_file(str(mind / "x.md"))["content"].endswith("hi"), "mind pages stay readable"
