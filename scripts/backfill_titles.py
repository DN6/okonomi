"""Regenerate one-sentence titles for existing sessions.

Dry-run by default: prints OLD -> NEW for every session and writes nothing.
Pass --write to persist. Titles are updated in place; the session directory
name (dir_name) is preserved so nothing is relocated.

    python scripts/backfill_titles.py            # preview
    python scripts/backfill_titles.py --write     # apply
"""

import sys

from okonomi.config import load_env_or_raise
from okonomi.session import SessionManager
from okonomi.utils import generate_title


def main() -> None:
    write = "--write" in sys.argv
    load_env_or_raise()

    sm = SessionManager()
    sessions = sm.list_sessions(limit=500)

    print(f"{len(sessions)} sessions ({'WRITE' if write else 'DRY-RUN'})\n")
    for s in sessions:
        full = sm.load_session(s.session_id)
        if not full or not full.message_history:
            print(f"  skip (no history): {s.title!r}")
            continue
        new_title = generate_title(full.message_history)
        if not new_title:
            print(f"  skip (no title): {s.title!r}")
            continue
        print(f"OLD: {s.title}")
        print(f"NEW: {new_title}\n")
        if write:
            sm.update_title(s.session_id, new_title)

    if not write:
        print("Dry run. Re-run with --write to apply.")


if __name__ == "__main__":
    main()
