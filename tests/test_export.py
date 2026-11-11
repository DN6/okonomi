"""Preference export: candidate sets, picks by lineage, tiers and pairs are
derived from the event DAG, artifacts and favorites the store already keeps."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from okonomi import db
from okonomi import session as session_store
from okonomi.events import AssetAction, EventType, event_to_payload
from okonomi.export import export_preferences
from okonomi.tui import assets


SID = "2026-11-10_export"


def _tool_call(call_id: str, args: dict, *, response_id: str = "") -> dict:
    payload = {
        "parts": [{"part_kind": "tool-call", "tool_name": "bash", "args": args, "tool_call_id": call_id}],
        "provider_response_id": response_id,
        "run_id": "run1",
    }
    return {"type": EventType.TOOL_CALL, "payload": json.dumps(payload), "content": "", "timestamp": "t"}


def _user(text: str) -> dict:
    return {"type": EventType.USER_MESSAGE, "payload": "{}", "content": text, "timestamp": "t"}


def _action(action: str, refs: list[str]) -> dict:
    return {
        "type": EventType.ASSET_ACTION,
        "payload": json.dumps(event_to_payload(AssetAction(action=action, refs=refs))),
        "content": "",
        "timestamp": "t",
    }


def _image(path: str, call_id: str) -> dict:
    return {"artifact_type": "image", "path": path, "tool_call_id": call_id, "timestamp": "t"}


def _jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def conn(store):
    db.upsert_session(store, {"session_id": SID})
    return store


def test_picks_tiers_and_pairs_follow_lineage(conn, tmp_path):
    a, b, c = (str(tmp_path / f"fox_v1_{i}.png") for i in (1, 2, 3))
    b2 = str(tmp_path / "fox_v2_1.png")
    moodboard = str(tmp_path / "references" / "moodboard.png")
    db.append_events(conn, SID, [
        _user(f"three foxes like {moodboard} please"),
        _tool_call("c1", {"command": f"gen.py --style {moodboard} --out fox_v1_{{1,2,3}}.png"}, response_id="pr1"),
        _user("the second one, warmer"),
        _tool_call("c2", {"command": f"edit.py --ref {b} --out fox_v2_1.png"}),  # event 3
        _action("favorite", [b2]),
    ])
    db.upsert_artifacts(conn, SID, [
        {"artifact_type": "image", "path": moodboard, "tool_name": "user", "timestamp": "t"},
        _image(a, "c1"), _image(b, "c1"), _image(c, "c1"), _image(b2, "c2"),
    ])

    metrics = export_preferences(tmp_path / "out", session_ids=[SID])

    first, second = _jsonl(tmp_path / "out" / "sets.jsonl")
    assert first["chosen"] == [b] and first["reaction"] == "the second one, warmer", first
    assert first["trace_ref"] == {"provider_response_id": "pr1", "run_id": "run1"}, first
    assert first["references"] == [moodboard] and second["references"] == [], "a set records the references it read"
    assert second["derived_from"] == [1] and second["candidates"][0]["tier"] == "love", second
    pairs = _jsonl(tmp_path / "out" / "pairs.jsonl")
    assert [(p["chosen"], p["rejected"]) for p in pairs] == [(b, a), (b, c)], pairs
    tiers = {g["ref"]: g["tier"] for g in _jsonl(tmp_path / "out" / "graded.jsonl")}
    assert tiers == {moodboard: "reference", a: "nope", b: "okay", c: "nope", b2: "love"}, tiers
    assert metrics["pursued_rate"] == 1.0 and metrics["rounds_to_keep"] == 1.0, metrics


def test_untouched_set_is_rejected_and_abandoned_branches_are_kept(conn, tmp_path):
    a, b = (str(tmp_path / f"owl_{i}.png") for i in (1, 2))
    db.append_events(conn, SID, [_user("owls"), _tool_call("c1", {"command": "gen owl_{1,2}.png"})])
    db.upsert_artifacts(conn, SID, [_image(a, "c1"), _image(b, "c1")])
    db.deactivate_events(conn, SID, from_event_id=1)
    db.append_events(conn, SID, [_user("no, cats instead")])

    metrics = export_preferences(tmp_path / "out", session_ids=[SID])

    (s,) = _jsonl(tmp_path / "out" / "sets.jsonl")
    assert s["abandoned"] and s["chosen"] == [] and s["reaction"] == "", s
    assert metrics["pursued_rate"] == 0.0, metrics
    assert metrics["tiers"] == {"reference": 0, "love": 0, "okay": 0, "nope": 2}, metrics


def test_favoriting_records_an_asset_action_on_the_session(store, tmp_path):
    recorded = []
    session = SimpleNamespace(session_id="s1", append_stream_events=recorded.extend)
    pic = tmp_path / "pic.png"
    pic.write_bytes(b"x")

    assets.set_favorite(str(pic), artifact_type="image", favorite=True, session=session)
    assets.set_favorite(str(pic), artifact_type="image", favorite=False, session=session)

    ref = session_store.norm_ref(str(pic))
    assert [(e.action, e.refs) for e in recorded] == [("favorite", [ref]), ("unfavorite", [ref])], recorded
