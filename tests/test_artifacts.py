"""Artifacts: one stable row per output across saves, and the sweep that
turns files in the session's artifacts dir into rows stamped with the tool
call that wrote them."""

from pathlib import Path

from okonomi import db
from okonomi.agent.agent import AgentDependencies


SID = "2026-11-10_artifacts"


def test_artifact_ids_survive_saves_and_duplicates_are_migrated_away(store):
    db.upsert_session(store, {"session_id": SID})
    db.upsert_artifacts(store, SID, [{"artifact_type": "image", "path": "/a.png", "prompt": "v1"}])
    first_id = store.execute("SELECT id FROM artifacts").fetchone()["id"]
    db.upsert_artifacts(
        store, SID, [{"artifact_type": "image", "path": "/a.png", "prompt": "v2", "tool_call_id": "c1"}]
    )

    (row,) = store.execute("SELECT id, prompt, tool_call_id FROM artifacts").fetchall()
    assert (row["id"], row["prompt"], row["tool_call_id"]) == (first_id, "v2", "c1"), "same ref → same row, updated"

    store.execute("INSERT INTO artifacts (session_id, path) VALUES (?, '/a.png')", (SID,))
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

    assert okonomi_agent._sync_session_assets(deps, tool_call_id="c7") == 1, "only media and briefs under artifacts/"
    (brief,) = deps.state.artifacts
    assert (brief.artifact_type, brief.tool_call_id, Path(brief.path).name) == ("research", "c7", "noir_brief.md")
