"""Regenerate one-sentence titles for existing sessions.

Dry-run by default: prints OLD -> NEW for every session and writes nothing.
Pass --write to persist. Titles are display-only; session directories are
never renamed.

    python scripts/backfill_titles.py            # preview
    python scripts/backfill_titles.py --write     # apply
"""

import sys

from okonomi.config import load_env_or_raise
from okonomi.session import list_sessions
from okonomi.utils import generate_session_title


def main() -> None:
    write = "--write" in sys.argv
    load_env_or_raise()

    sessions = list_sessions(limit=500)

    print(f"{len(sessions)} sessions ({'WRITE' if write else 'DRY-RUN'})\n")
    for s in sessions:
        history = s.load_messages()
        if not history:
            print(f"  skip (no history): {s.title!r}")
            continue
        new_title = generate_session_title(history)
        if not new_title:
            print(f"  skip (no title): {s.title!r}")
            continue
        print(f"OLD: {s.title}")
        print(f"NEW: {new_title}\n")
        if write:
            s.set_title(new_title)

    if not write:
        print("Dry run. Re-run with --write to apply.")


if __name__ == "__main__":
    main()
