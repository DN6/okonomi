"""One-shot migration: import pre-events-era transcripts into the events table.

Sessions saved before the events migration kept their conversation in
``<session_dir>/messages.jsonl`` (with slug-named directories recorded in
the sessions table's old ``dir_name`` column). This walks every session
that has no conversation events, finds its jsonl, validates it through
the pydantic-ai adapter, and inserts event rows — after which resume and
content search work on old sessions like new ones.

Idempotent: sessions that already have conversation events are skipped.
Dry-run by default; pass --write to persist.

    python scripts/import_legacy_transcripts.py            # preview
    python scripts/import_legacy_transcripts.py --write    # apply
"""

import json
import sys

from okonomi import db
from okonomi.constants import OKONOMI_SESSIONS_DIR
from okonomi.events import CONVERSATION_TYPES
from okonomi.session import _message_row


def _find_messages_file(conn, session_id: str):
    """Locate the legacy jsonl: the old dir_name column when present
    (slug-named folders), falling back to the id-named folder."""
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    candidates = []
    if row is not None and "dir_name" in row.keys() and row["dir_name"]:
        candidates.append(OKONOMI_SESSIONS_DIR / row["dir_name"] / "messages.jsonl")
    candidates.append(OKONOMI_SESSIONS_DIR / session_id / "messages.jsonl")
    for path in candidates:
        if path.exists():
            return path
    return None


def main() -> None:
    write = "--write" in sys.argv
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    conn = db.connection()
    session_ids = [r["session_id"] for r in conn.execute("SELECT session_id FROM sessions").fetchall()]

    imported = skipped = missing = failed = 0
    for sid in session_ids:
        if db.count_events(conn, sid, types=CONVERSATION_TYPES) > 0:
            skipped += 1
            continue
        path = _find_messages_file(conn, sid)
        if path is None:
            missing += 1
            continue
        try:
            raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            messages = list(ModelMessagesTypeAdapter.validate_python(raw))
        except Exception as e:
            print(f"  FAILED {sid}: {type(e).__name__}: {e}")
            failed += 1
            continue
        print(f"  {sid}: {len(messages)} messages from {path.parent.name}/")
        if write:
            with db.transaction(conn):
                db.append_events(conn, sid, [_message_row(m) for m in messages])
        imported += 1

    mode = "WRITE" if write else "DRY-RUN"
    print(
        f"\n{mode}: {imported} imported, {skipped} already migrated, "
        f"{missing} without a jsonl, {failed} failed to parse"
    )
    if not write:
        print("Re-run with --write to apply.")


if __name__ == "__main__":
    main()
