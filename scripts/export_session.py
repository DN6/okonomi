"""Export a session from the DB to JSONL.

Default output is the conversation as pydantic-ai message JSON, one per
line — byte-compatible with the old ``messages.jsonl`` format. Pass
``--events`` to dump the full event stream (system notes, titles,
interrupts, rewound rows) with envelope columns instead.

    python scripts/export_session.py <session-id>                # → stdout
    python scripts/export_session.py <session-id> --out dump.jsonl
    python scripts/export_session.py <session-id> --events
"""

import argparse
import json
import sys

from okonomi import db
from okonomi.events import CONVERSATION_TYPES
from okonomi.session import Session


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a session to JSONL")
    parser.add_argument("session_id", help="Exact session id")
    parser.add_argument("--out", help="Output path (default: stdout)")
    parser.add_argument(
        "--events",
        action="store_true",
        help="Full event stream with envelope columns (including rewound rows) instead of messages",
    )
    args = parser.parse_args()

    if Session.load(args.session_id) is None:
        print(f"Unknown session: {args.session_id}", file=sys.stderr)
        sys.exit(1)

    conn = db.connection()
    if args.events:
        rows = db.fetch_events(conn, args.session_id, active_only=False)
        lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    else:
        rows = db.fetch_events(conn, args.session_id, types=CONVERSATION_TYPES)
        lines = [r["payload"] for r in rows]

    text = "\n".join(lines) + ("\n" if lines else "")
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"{len(lines)} lines → {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
