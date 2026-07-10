"""Consolidate split session directories caused by the dir_name regen bug.

Background: ``save_session`` used to construct fresh ``SessionMetadata``
without preserving ``dir_name``, so every passive title regeneration
caused the next save to land in a brand-new directory keyed off the
latest title slug. Result: one logical session ended up split across
multiple folders on disk.

This script:

1. Scans every directory under the storage root.
2. Reads ``session.json`` and groups dirs by ``session_id``.
3. For each group with more than one folder, picks the most recently
   updated as the canonical one.
4. Merges the others into it: copies assets (preserving ``assets/<type>``
   subpaths), unions the ``artifacts`` lists, deduplicates, and rewrites
   any ``path`` strings that point to the now-vanished folders.
5. Updates the index to drop the merged-away entries.
6. Deletes the merged-away directories.

By default runs as a dry-run — pass ``--apply`` to actually move files.
Always run with ``--dry-run`` first; this touches your sessions folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_meta(session_json: Path) -> dict[str, Any] | None:
    try:
        with open(session_json) as f:
            return json.load(f)
    except Exception:
        return None


def _scan(storage: Path) -> dict[str, list[Path]]:
    """Return {session_id: [dir_paths]}."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for entry in sorted(storage.iterdir()):
        if not entry.is_dir():
            continue
        sj = entry / "session.json"
        if not sj.exists():
            continue
        data = _load_meta(sj)
        if not data:
            continue
        sid = data.get("metadata", {}).get("session_id")
        if sid:
            groups[sid].append(entry)
    return groups


def _pick_canonical(dirs: list[Path]) -> Path:
    """Most recently updated session.json wins."""

    def key(d: Path) -> str:
        data = _load_meta(d / "session.json") or {}
        meta = data.get("metadata", {}) or {}
        return meta.get("updated_at") or meta.get("created_at") or ""

    return max(dirs, key=key)


def _rewrite_path(p, old_root: Path, new_root: Path):
    """If ``p`` is a string pointing inside ``old_root``, rewrite to ``new_root``.

    Older data sometimes has dict-shaped path fields (raw fal envelopes
    that leaked through). Leave those untouched.
    """
    if not isinstance(p, str) or not p:
        return p
    try:
        rel = Path(p).resolve().relative_to(old_root.resolve())
    except (ValueError, OSError):
        return p
    return str(new_root / rel)


def _merge_artifacts(canonical_arts: list, other_arts: list) -> list:
    """Union by (artifact_type, url, path). Preserves order from canonical first.

    Non-string url/path values (legacy dict-shaped data) get keyed via
    repr to avoid TypeError, so they still dedupe correctly across runs.
    """

    def _key(a):
        def _str(v):
            return v if isinstance(v, str) else repr(v)

        return (_str(a.get("artifact_type", "")), _str(a.get("url", "")), _str(a.get("path", "")))

    seen = set()
    merged = []
    for a in list(canonical_arts) + list(other_arts):
        key = _key(a)
        if key in seen:
            continue
        seen.add(key)
        merged.append(a)
    return merged


def consolidate(storage: Path, apply: bool) -> int:
    groups = _scan(storage)
    splits = {sid: dirs for sid, dirs in groups.items() if len(dirs) > 1}
    if not splits:
        print(f"Scanned {sum(len(v) for v in groups.values())} session dirs across {len(groups)} ids.")
        print("No splits found — nothing to consolidate.")
        return 0

    print(f"Found {len(splits)} session id(s) split across multiple folders:\n")

    actions: list[tuple[Path, Path]] = []  # (dir_to_merge_in, canonical_dir)

    for sid, dirs in splits.items():
        canonical = _pick_canonical(dirs)
        others = [d for d in dirs if d != canonical]
        print(f"session_id = {sid}")
        print(f"  CANONICAL: {canonical.name}")
        for d in others:
            print(f"  → MERGE  : {d.name}")
            actions.append((d, canonical))
        print()

    if not apply:
        print("(dry run — pass --apply to actually merge)")
        return 0

    # --- Apply ---
    index_path = storage / "index.json"
    index = []
    if index_path.exists():
        try:
            with open(index_path) as f:
                index = json.load(f)
        except Exception:
            index = []
    index_by_dir = {e.get("dir_name", ""): e for e in index}

    canonicals_touched: set[Path] = set()

    for src, canonical in actions:
        canonicals_touched.add(canonical)
        # Move every file under src/ into canonical/, preserving subpath.
        for source_path in src.rglob("*"):
            if source_path.is_dir():
                continue
            rel = source_path.relative_to(src)
            target = canonical / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                # Don't clobber. Skip silently — canonical is authoritative.
                continue
            shutil.move(str(source_path), str(target))

        # Merge artifact records from src/session.json into canonical
        src_data = _load_meta(src / "session.json") or {}
        # We've already moved src/session.json into canonical/session.json
        # if canonical had no session.json (it should — but be defensive).
        # Re-read the merged session.json from src, even though it's been
        # moved. Use src_data captured above.
        canon_json_path = canonical / "session.json"
        canon_data = _load_meta(canon_json_path) or {}
        canon_arts = canon_data.setdefault("artifacts", [])
        src_arts = src_data.get("artifacts", []) or []

        # Rewrite any paths in src_arts that point into src to point into canonical.
        for a in src_arts:
            a["path"] = _rewrite_path(a.get("path", ""), src, canonical)

        canon_data["artifacts"] = _merge_artifacts(canon_arts, src_arts)

        # Also rewrite paths in canon_arts (in case the canonical itself has
        # any stale paths into the merged-away folder).
        for a in canon_data["artifacts"]:
            a["path"] = _rewrite_path(a.get("path", ""), src, canonical)

        with open(canon_json_path, "w") as f:
            json.dump(canon_data, f, indent=2)

        # Drop src from the index.
        index_by_dir.pop(src.name, None)

        # Remove the now-empty (or near-empty) src dir.
        try:
            shutil.rmtree(src)
            print(f"merged + removed: {src.name}")
        except Exception as e:
            print(f"warning: could not remove {src.name}: {e}")

    # Write index back.
    if index_path.exists():
        with open(index_path, "w") as f:
            json.dump(list(index_by_dir.values()), f, indent=2)

    print(f"\nDone. Merged into {len(canonicals_touched)} canonical session(s).")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--storage",
        default=str(Path.home() / ".okonomi" / "sessions"),
        help="Sessions storage root (default: ~/.okonomi/sessions)",
    )
    p.add_argument("--apply", action="store_true", help="Actually move files. Default is dry-run.")
    args = p.parse_args()
    storage = Path(args.storage).expanduser()
    if not storage.exists():
        print(f"Storage dir not found: {storage}", file=sys.stderr)
        return 1
    return consolidate(storage, apply=args.apply)


if __name__ == "__main__":
    sys.exit(main())
