"""Preference export: turn what the store already records into training data.

No signal is captured specially for this. A *candidate set* is the group
of artifacts one tool call (one model response) produced. A candidate is
*chosen* when something downstream touched it — a later tool call read
its path, the user mentioned or attached it, favorited, opened or copied
it — and *rejected* when it got nothing while it had siblings. The user's
reply after the set is kept verbatim as the reason.

Outputs, one JSON object per line:

  sets.jsonl    every candidate set with per-candidate signals, the sets
                its producing call derived from, and the user-supplied
                references it read
  pairs.jsonl   chosen × rejected pairs within a set (DPO-shaped)
  graded.jsonl  every artifact with a tier: reference (supplied by the
                user, before any generation) / love / okay / nope
  metrics.json  pursued rate and rounds-to-keep across the export
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import db
from .events import EventType
from .session import norm_ref


LOVE_ACTIONS = ("favorite", "open", "copy")


def _tool_calls(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """``(tool_call_id, args as text)`` for every call in a model response."""
    out = []
    for part in payload.get("parts", []):
        if part.get("part_kind") != "tool-call":
            continue
        args = part.get("args")
        out.append((part.get("tool_call_id") or "", args if isinstance(args, str) else json.dumps(args)))
    return out


def _producing_event(artifact: dict[str, Any], call_events: dict[str, int], tool_call_rows: list[dict]) -> int | None:
    if artifact["tool_call_id"] in call_events:
        return call_events[artifact["tool_call_id"]]
    # Legacy artifacts carry no tool_call_id: attribute to the last tool
    # call that completed before the file appeared.
    before = [r["event_id"] for r in tool_call_rows if r["timestamp"] <= artifact["timestamp"]]
    return before[-1] if before else None


def _reaction(events: list[dict], parents: dict[int, int | None], after: int) -> str:
    """The first user message on any branch descending from ``after``."""
    for row in events:
        if row["type"] != EventType.USER_MESSAGE.value or row["event_id"] <= after:
            continue
        node: int | None = parents.get(row["event_id"])
        while node is not None and node > after:
            node = parents.get(node)
        if node == after:
            return row["content"]
    return ""


def session_sets(conn, session_id: str, favorites: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """The session's candidate sets, and the refs of images the user supplied."""
    events = db.fetch_events(conn, session_id, active_only=False)
    parents = {r["event_id"]: r["parent_id"] for r in events}
    tool_call_rows = [r for r in events if r["type"] == EventType.TOOL_CALL.value]
    call_events: dict[str, int] = {}
    call_args: dict[int, str] = {}
    trace_refs: dict[int, dict[str, Any]] = {}
    for row in tool_call_rows:
        payload = json.loads(row["payload"])
        trace_refs[row["event_id"]] = {
            "provider_response_id": payload.get("provider_response_id"),
            "run_id": payload.get("run_id"),
        }
        for call_id, args in _tool_calls(payload):
            call_events[call_id] = row["event_id"]
            call_args[row["event_id"]] = call_args.get(row["event_id"], "") + "\n" + args
    user_rows = [r for r in events if r["type"] == EventType.USER_MESSAGE.value]
    actions: list[tuple[int, str, set[str]]] = []
    for row in events:
        if row["type"] == EventType.ARTIFACT_ACTION.value:
            payload = json.loads(row["payload"])
            actions.append((row["event_id"], payload["action"], set(payload["refs"])))

    grouped: dict[int, list[dict[str, Any]]] = {}
    references: list[str] = []
    seen_refs: set[str] = set()
    for artifact in db.fetch_artifacts(conn, session_id):
        ref = artifact["path"] or artifact["url"]
        if not ref or ref in seen_refs:
            continue
        seen_refs.add(ref)
        # Images the user supplied: picks made before any generation, so
        # they anchor taste rather than compete in a candidate set.
        if artifact["tool_name"] == "user":
            references.append(ref)
            continue
        producer = _producing_event(artifact, call_events, tool_call_rows)
        if producer is None:
            continue
        grouped.setdefault(producer, []).append({**artifact, "ref": ref})

    sets: list[dict[str, Any]] = []
    for producer, artifacts in sorted(grouped.items()):
        candidates = []
        for artifact in artifacts:
            ref = artifact["ref"]
            name = Path(ref).name
            signals = []
            if any(eid > producer and name in args for eid, args in call_args.items()):
                signals.append("used")
            if any(r["event_id"] > producer and name in r["content"] for r in user_rows):
                signals.append("mentioned")
            favorited = norm_ref(ref) in favorites
            for eid, action, refs in actions:
                if eid > producer and norm_ref(ref) in refs:
                    if action == "favorite":
                        favorited = True
                    elif action == "unfavorite":
                        favorited = False
                    elif action in LOVE_ACTIONS:
                        signals.append(action)
            if favorited:
                signals.append("favorite")
            candidates.append({"ref": ref, "artifact_type": artifact["artifact_type"], "signals": signals})
        for c in candidates:
            c["chosen"] = bool(c["signals"])
            if any(s in LOVE_ACTIONS for s in c["signals"]):
                c["tier"] = "love"
            elif not c["signals"] and len(candidates) >= 2:
                c["tier"] = "nope"
            else:
                c["tier"] = "okay"
        # Lineage: the earlier sets whose candidates this set's producing
        # call read — the picks that led here.
        derived_from = [
            p["event_id"] for p in sets
            if any(Path(c["ref"]).name in call_args.get(producer, "") for c in p["candidates"])
        ]
        sets.append({
            "session_id": session_id,
            "event_id": producer,
            "abandoned": not any(r["event_id"] == producer and r["active"] for r in tool_call_rows),
            "trace_ref": trace_refs.get(producer, {}),
            "reaction": _reaction(events, parents, producer),
            "derived_from": derived_from,
            "references": [r for r in references if Path(r).name in call_args.get(producer, "")],
            "candidates": candidates,
            "chosen": [c["ref"] for c in candidates if c["chosen"]],
        })
    return sets, references


def _rounds_to_keep(sets: list[dict[str, Any]]) -> list[int]:
    """For every loved artifact, the number of picks that led to it: the
    length of its set's ``derived_from`` chain within the session."""
    by_key = {(s["session_id"], s["event_id"]): s for s in sets}
    depths = []
    for s in sets:
        if not any(c["tier"] == "love" for c in s["candidates"]):
            continue
        depth, node = 0, s
        while node["derived_from"]:
            depth += 1
            node = by_key[(node["session_id"], node["derived_from"][-1])]
        depths.append(depth)
    return depths


def export_preferences(out_dir: Path, *, session_ids: list[str] | None = None) -> dict[str, Any]:
    conn = db.connection()
    favorites = {norm_ref(f["url"]) for f in db.list_favorites(conn) if f.get("url")}
    session_ids = session_ids or [s["session_id"] for s in db.list_sessions(conn)]
    out_dir.mkdir(parents=True, exist_ok=True)
    all_sets: list[dict[str, Any]] = []
    graded = []
    for session_id in session_ids:
        sets, references = session_sets(conn, session_id, favorites)
        all_sets.extend(sets)
        graded.extend({"session_id": session_id, "ref": r, "tier": "reference", "signals": []} for r in references)

    pairs = []
    for s in all_sets:
        rejected = [c for c in s["candidates"] if not c["chosen"]]
        for c in s["candidates"]:
            graded.append({"session_id": s["session_id"], "ref": c["ref"], "tier": c["tier"], "signals": c["signals"]})
            if not c["chosen"]:
                continue
            for r in rejected:
                pairs.append({
                    "session_id": s["session_id"],
                    "event_id": s["event_id"],
                    "trace_ref": s["trace_ref"],
                    "chosen": c["ref"],
                    "rejected": r["ref"],
                    "reaction": s["reaction"],
                })

    multi = [s for s in all_sets if len(s["candidates"]) >= 2]
    rounds = _rounds_to_keep(all_sets)
    metrics = {
        "sessions": len(session_ids),
        "sets": len(all_sets),
        "pairs": len(pairs),
        "pursued_rate": (sum(1 for s in multi if s["chosen"]) / len(multi)) if multi else None,
        "rounds_to_keep": (sum(rounds) / len(rounds)) if rounds else None,
        "tiers": {t: sum(1 for g in graded if g["tier"] == t) for t in ("reference", "love", "okay", "nope")},
    }
    _write_jsonl(out_dir / "sets.jsonl", all_sets)
    _write_jsonl(out_dir / "pairs.jsonl", pairs)
    _write_jsonl(out_dir / "graded.jsonl", graded)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
