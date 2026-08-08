"""Session-tree branching: the checkout primitive, tree-row flattening,
branch summaries, and branch-aware saves."""

import json

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from okonomi import db
from okonomi.context import BRANCH_SUMMARY_PREFIX, ContextManager
from okonomi.events import CONVERSATION_TYPES, EventType
from okonomi.tui.pickers import _tree_rows


SID = "2026-07-30_treetest"


@pytest.fixture
def conn(tmp_path):
    c = db.open_db(tmp_path / "test.db")
    db.upsert_session(c, {"session_id": SID})
    yield c
    c.close()


def _row(text: str, type_: EventType = EventType.USER_MESSAGE) -> dict:
    return {"type": type_, "payload": "{}", "content": text, "timestamp": ""}


def _active_ids(conn) -> list[int]:
    return [r["event_id"] for r in db.fetch_events(conn, SID)]


def _forked_session(conn) -> None:
    """ids 0-3 linear, rewind at 2, ids 4-5 appended → branches 0-1-2-3
    (suffix inactive) and 0-1-4-5 (active)."""
    db.append_events(conn, SID, [_row(f"e{i}") for i in range(4)])
    db.deactivate_events(conn, SID, from_event_id=2)
    db.append_events(conn, SID, [_row("e4"), _row("e5")])


# ── db.checkout_events ───────────────────────────────────────────────────


def test_checkout_activates_root_path(conn):
    _forked_session(conn)
    assert _active_ids(conn) == [0, 1, 4, 5], f"fixture broken: active={_active_ids(conn)}"

    n = db.checkout_events(conn, SID, 3)
    assert n == 4, f"expected 4 events on path root→3, got {n}"
    assert _active_ids(conn) == [0, 1, 2, 3], f"active after checkout: {_active_ids(conn)}"

    parents = {r["event_id"]: r["parent_id"] for r in db.fetch_events(conn, SID, active_only=False)}
    assert parents[4] == 1 and parents[2] == 1, f"fork edges changed by checkout: {parents}"


def test_append_after_checkout_chains_from_leaf(conn):
    _forked_session(conn)
    db.checkout_events(conn, SID, 3)
    db.append_events(conn, SID, [_row("e6")])

    rows = db.fetch_events(conn, SID, active_only=False)
    new = next(r for r in rows if r["event_id"] == 6)
    assert new["parent_id"] == 3, f"append after checkout chained from {new['parent_id']}, expected leaf 3"
    assert _active_ids(conn) == [0, 1, 2, 3, 6]


def test_checkout_none_deactivates_all(conn):
    _forked_session(conn)
    n = db.checkout_events(conn, SID, None)
    assert n == 0
    assert _active_ids(conn) == [], f"expected empty active path, got {_active_ids(conn)}"


def test_checkout_unknown_target_raises(conn):
    _forked_session(conn)
    with pytest.raises(ValueError, match="99"):
        db.checkout_events(conn, SID, 99)


# ── pickers._tree_rows ───────────────────────────────────────────────────


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
    assert [r["depth"] for r in rows] == [0, 0, 1, 1, 1, 1, 0, 0], f"depths: {[r['depth'] for r in rows]}"

    by_id = {r["event_id"]: r for r in rows}
    assert by_id[0]["checkout_target"] is None, "root user row must check out to empty path"
    assert by_id[2]["checkout_target"] == 1 and by_id[4]["checkout_target"] == 1
    assert by_id[3]["checkout_target"] == 3, "assistant row checks out to itself"
    assert by_id[2]["prefill"] == "try redis"
    assert by_id[3]["prefill"] == ""
    assert by_id[8]["is_current"], "highest-id active message is the current leaf"
    assert not by_id[5]["is_current"]
    assert not by_id[5]["active"] and by_id[6]["active"]


# ── pickers._abandoned_count + history search filter ─────────────────────


def _forked_events():
    u, a = EventType.USER_MESSAGE, EventType.ASSISTANT_MESSAGE
    return [
        _ev(0, None, u, "first ask", 1, _user_payload("first ask")),
        _ev(1, 0, a, "answer 1", 1),
        _ev(2, 1, u, "try redis", 1, _user_payload("try redis")),
        _ev(3, 2, a, "redis done", 1),
        _ev(4, 1, u, "try sqlite", 0, _user_payload("try sqlite")),
        _ev(5, 4, a, "sqlite done", 0),
    ]


def test_abandoned_count_shared_helper():
    from okonomi.tui.pickers import _abandoned_count

    events = _forked_events()
    parents = {e["event_id"]: e["parent_id"] for e in events}
    conv_types = {t.value for t in CONVERSATION_TYPES}

    # Checking out the active leaf (3) abandons nothing on the active path.
    assert _abandoned_count(events, parents, conv_types, 3) == 0
    # Checking out to node 1 drops active turns 2 and 3 off the path.
    assert _abandoned_count(events, parents, conv_types, 1) == 2
    # Checking out to empty path (None) abandons every active conversation turn.
    assert _abandoned_count(events, parents, conv_types, None) == 4


def test_history_preview_filter_matches_tree_rows():
    """The Ctrl-R picker filters _tree_rows previews with fuzzy_match; a
    query should surface the matching message and carry its checkout_target."""
    from okonomi.utils import fuzzy_match
    from okonomi.tui.pickers import _tree_rows

    rows = _tree_rows(_forked_events())
    hits = [r for r in rows if fuzzy_match("redis", r["preview"].lower())]
    assert [r["event_id"] for r in hits] == [2, 3], f"redis hits: {[r['event_id'] for r in hits]}"
    # user row → edit&resubmit its parent; assistant row → continue from itself
    by_id = {r["event_id"]: r for r in hits}
    assert by_id[2]["checkout_target"] == 1 and by_id[2]["prefill"] == "try redis"
    assert by_id[3]["checkout_target"] == 3 and by_id[3]["prefill"] == ""


# ── ContextManager.summarize_branch ──────────────────────────────────────


def test_summarize_branch_prefix(monkeypatch):
    monkeypatch.setattr("okonomi.context.create_chat_client", lambda: None)
    cm = ContextManager(model_id="test-model")

    monkeypatch.setattr(cm, "_chat", lambda prompt, max_tokens: "did some work on foo.py")
    summary = cm.summarize_branch([ModelRequest(parts=[UserPromptPart(content="hi")])])
    assert summary is not None and summary.startswith(BRANCH_SUMMARY_PREFIX), f"summary: {summary!r}"

    monkeypatch.setattr(cm, "_chat", lambda prompt, max_tokens: "")
    assert cm.summarize_branch([]) is None, "failed LLM call must yield None, not an empty summary"


def test_branch_summary_prefix_is_skipped_on_replay():
    from okonomi.tui.transcript import _SYNTHETIC_REPLAY_PREFIXES
    from okonomi.utils import _SYNTHETIC_PROMPT_PREFIXES

    assert any(BRANCH_SUMMARY_PREFIX.startswith(p) for p in _SYNTHETIC_REPLAY_PREFIXES), (
        f"replay skip tuple {_SYNTHETIC_REPLAY_PREFIXES} misses {BRANCH_SUMMARY_PREFIX[:20]!r}"
    )
    assert any(BRANCH_SUMMARY_PREFIX.startswith(p) for p in _SYNTHETIC_PROMPT_PREFIXES), (
        f"titler skip tuple {_SYNTHETIC_PROMPT_PREFIXES} misses {BRANCH_SUMMARY_PREFIX[:20]!r}"
    )


# ── Session save/load across a checkout ──────────────────────────────────


def test_checkout_reload_sync(conn, tmp_path, monkeypatch):
    """A checkout followed by reload keeps _sync_conversation's count
    heuristic consistent: the next save appends only the new tail, chained
    from the checked-out leaf."""
    from types import SimpleNamespace

    import okonomi.session as session_store

    monkeypatch.setattr(db, "_connection", conn)
    monkeypatch.setattr(session_store, "OKONOMI_SESSIONS_DIR", tmp_path)

    history = [
        ModelRequest(parts=[UserPromptPart(content="ask one")]),
        ModelResponse(parts=[TextPart(content="answer one")]),
        ModelRequest(parts=[UserPromptPart(content="ask two")]),
        ModelResponse(parts=[TextPart(content="answer two")]),
    ]
    session = session_store.Session(session_id=SID, state=SimpleNamespace(message_history=history, artifacts=[]))
    session.save()
    assert _active_ids(conn) == [0, 1, 2, 3]

    with db.transaction(conn):
        db.checkout_events(conn, SID, 1)
    loaded = session.load_messages()
    assert len(loaded) == 2, f"active path should hold 2 messages, got {len(loaded)}"

    summary = ModelRequest(parts=[UserPromptPart(content=f"{BRANCH_SUMMARY_PREFIX}\nabandoned work")])
    session.state.message_history = loaded + [summary]
    session.save()

    rows = db.fetch_events(conn, SID, types=CONVERSATION_TYPES)
    assert [r["event_id"] for r in rows] == [0, 1, 4], f"active conversation: {[r['event_id'] for r in rows]}"
    new = rows[-1]
    assert new["parent_id"] == 1, f"summary row chained from {new['parent_id']}, expected checked-out leaf 1"
    assert new["type"] == EventType.USER_MESSAGE.value
    inactive = [r["event_id"] for r in db.fetch_events(conn, SID, active_only=False) if not r["active"]]
    assert inactive == [2, 3], f"old branch must stay in the DAG inactive, got {inactive}"
