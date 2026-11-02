"""Read / write / edit standalone tools for local files.

Bounded reads (offset + limit), line-numbered output, and read-before-write
staleness enforcement so the model can't stomp a file it hasn't inspected.

Returns ``dict`` shapes throughout (``{"content": ...}`` on success,
``{"error": ...}`` on failure) to match the rest of okonomi's tool
surface.
"""

from pathlib import Path

from typing_extensions import TypedDict

from okonomi.config import OKONOMI_MIND_DIR


_DEFAULT_READ_LIMIT = 500
_MAX_READ_LIMIT = 2_000
_MAX_LINE_CHARS = 2_000  # truncate any single line that exceeds this
_MAX_FILE_BYTES = 5_000_000  # refuse to open files larger than 5MB
_BINARY_SNIFF_BYTES = 8_192


class ReadResult(TypedDict, total=False):
    content: str
    path: str
    line_count: int
    truncated: bool
    error: str


class WriteResult(TypedDict, total=False):
    path: str
    bytes_written: int
    created: bool
    error: str


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _mind_write_refusal(p: Path) -> dict | None:
    # Mind pages share one frontmatter schema that recall and /mind read;
    # ``remember`` is the single writer that keeps it, so file tools refuse.
    if p.is_relative_to(OKONOMI_MIND_DIR.expanduser().resolve()):
        return {"error": f"{p} is a mind wiki page. Use remember(title, body, ..., slug=..., update=True) to write it."}
    return None


def _looks_binary(data: bytes) -> bool:
    return b"\x00" in data[:_BINARY_SNIFF_BYTES]


class FileTools:
    """Local file operations: read, write, edit."""


    def __init__(self):
        # Track per-path mtime captured at read time. write_file / edit_file
        # require the model to have read the file at its current version
        # before mutating it, so concurrent edits aren't silently stomped.
        self._read_mtimes: dict[str, float] = {}

    def read_file(self, path: str, offset: int = 1, limit: int = _DEFAULT_READ_LIMIT) -> ReadResult:
        """Read a text file with pagination and line numbers.

        Output is formatted ``{line_no:>6}\\t{line}`` (tab-separated),
        matching Claude Code so the model can reference file_path:line_no.
        Records the file's mtime so a subsequent ``write_file`` /
        ``edit_file`` on the same path can confirm freshness.

        Args:
            path: Absolute or ``~``-expandable file path.
            offset: 1-based first line to include (default 1).
            limit: Max number of lines to return (default 500, max 2000).

        Returns:
            ``{"content": str, "path": str, "line_count": int, "truncated": bool}``
            on success, or ``{"error": str}`` on failure.
        """
        try:
            p = _resolve(path)
        except Exception as e:
            return {"error": f"Invalid path: {e}"}
        if not p.exists():
            return {"error": f"No such file: {p}"}
        if not p.is_file():
            return {"error": f"Not a regular file: {p}"}

        try:
            size = p.stat().st_size
        except OSError as e:
            return {"error": f"stat failed: {e}"}
        if size > _MAX_FILE_BYTES:
            return {
                "error": (
                    f"File is {size:,} bytes; refusing to read more than "
                    f"{_MAX_FILE_BYTES:,}. Use bash(head/tail/sed) for a slice."
                )
            }

        try:
            with open(p, "rb") as f:
                head = f.read(_BINARY_SNIFF_BYTES)
            if _looks_binary(head):
                return {"error": f"File looks binary (NUL byte in first {_BINARY_SNIFF_BYTES} bytes): {p}"}
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return {"error": f"Read failed: {e}"}

        try:
            self._read_mtimes[str(p)] = p.stat().st_mtime
        except OSError:
            pass

        lines = text.splitlines()
        line_count = len(lines)

        offset = max(1, offset)
        limit = max(1, min(int(limit), _MAX_READ_LIMIT))
        start = offset - 1
        end = min(start + limit, line_count)
        if start >= line_count and line_count > 0:
            return {
                "content": "",
                "path": str(p),
                "line_count": line_count,
                "truncated": False,
                "error": f"offset {offset} is past the end ({line_count} lines)",
            }

        out_lines = []
        for i in range(start, end):
            raw = lines[i]
            if len(raw) > _MAX_LINE_CHARS:
                raw = raw[:_MAX_LINE_CHARS] + f"... [truncated {len(raw) - _MAX_LINE_CHARS} chars]"
            out_lines.append(f"{i + 1:>6}\t{raw}")

        return {
            "content": "\n".join(out_lines),
            "path": str(p),
            "line_count": line_count,
            "truncated": end < line_count,
        }

    def write_file(self, path: str, content: str) -> WriteResult:
        """Write ``content`` to ``path``, replacing any existing contents.

        Safety: if the file already exists, the model must have called
        ``read_file`` on it at its current mtime first — otherwise the
        write is refused. New files can be written without a prior read.
        Parent directories are created on demand.

        Args:
            path: Destination path. Parent directories will be created.
            content: Full file contents (str). Pass an empty string to
                truncate.

        Returns:
            ``{"path", "bytes_written", "created"}`` on success, or
            ``{"error"}`` on failure.
        """
        try:
            p = _resolve(path)
        except Exception as e:
            return {"error": f"Invalid path: {e}"}
        if refusal := _mind_write_refusal(p):
            return refusal

        created = not p.exists()
        if not created:
            if p.is_dir():
                return {"error": f"Path is a directory: {p}"}
            staleness = self._check_fresh(p)
            if staleness is not None:
                return {"error": staleness}

        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            data = content.encode("utf-8")
            p.write_bytes(data)
            self._read_mtimes[str(p)] = p.stat().st_mtime
        except OSError as e:
            return {"error": f"Write failed: {e}"}

        return {"path": str(p), "bytes_written": len(data), "created": created}

    def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> WriteResult:
        """Replace ``old_string`` with ``new_string`` inside an existing file.

        Safety: requires a prior ``read_file`` on the same path at its
        current mtime. If ``replace_all`` is False (default),
        ``old_string`` must appear exactly once in the file — otherwise
        the edit is refused to avoid ambiguous matches.

        Args:
            path: Existing file path.
            old_string: Exact substring to replace. Whitespace and
                indentation must match.
            new_string: Replacement.
            replace_all: When True, replace every occurrence. When False
                (default), require a unique match.

        Returns:
            ``{"path", "bytes_written", "created": False}`` on success,
            or ``{"error"}`` on failure.
        """
        try:
            p = _resolve(path)
        except Exception as e:
            return {"error": f"Invalid path: {e}"}
        if refusal := _mind_write_refusal(p):
            return refusal
        if not p.exists():
            return {"error": f"No such file: {p}"}
        if not p.is_file():
            return {"error": f"Not a regular file: {p}"}

        staleness = self._check_fresh(p)
        if staleness is not None:
            return {"error": staleness}

        try:
            text = p.read_text(encoding="utf-8")
        except OSError as e:
            return {"error": f"Read failed: {e}"}
        except UnicodeDecodeError as e:
            return {"error": f"File is not UTF-8 text: {e}"}

        if old_string == new_string:
            return {"error": "old_string and new_string are identical — nothing to do"}
        if old_string not in text:
            return {"error": "old_string not found in file"}

        if replace_all:
            new_text = text.replace(old_string, new_string)
        else:
            count = text.count(old_string)
            if count > 1:
                return {
                    "error": (
                        f"old_string occurs {count} times; pass replace_all=True "
                        "or supply more surrounding context to make it unique"
                    )
                }
            new_text = text.replace(old_string, new_string, 1)

        try:
            data = new_text.encode("utf-8")
            p.write_bytes(data)
            self._read_mtimes[str(p)] = p.stat().st_mtime
        except OSError as e:
            return {"error": f"Write failed: {e}"}

        return {"path": str(p), "bytes_written": len(data), "created": False}

    def _check_fresh(self, p: Path) -> str | None:
        """Return an error string if ``p`` wasn't read at its current mtime."""
        key = str(p)
        seen = self._read_mtimes.get(key)
        if seen is None:
            return (
                f"Must call read_file({key!r}) before writing/editing — "
                "the read tracks the current version so concurrent edits "
                "aren't silently stomped."
            )
        try:
            now = p.stat().st_mtime
        except OSError as e:
            return f"stat failed: {e}"
        if now > seen + 1e-6:
            return (
                f"File changed since last read (mtime moved). "
                f"Re-read with read_file({key!r}) before editing."
            )
        return None
