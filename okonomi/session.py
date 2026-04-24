"""Session management for Okonomi.

This module provides session storage and retrieval functionality to enable
resuming from previous runs, viewing past outputs, and searching through
generation history.
"""

import fcntl
import json
import re
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .config import config


_ASSET_EXTENSIONS = {
    "images": {".png", ".jpg", ".jpeg", ".webp", ".gif"},
    "videos": {".mp4", ".webm", ".mov"},
    "html": {".html", ".htm"},
}


def _slugify(text: str, max_len: int = 50) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    text = text.strip("-")
    return text[:max_len].rstrip("-")


def asset_subdir_for_file(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    for subdir, extensions in _ASSET_EXTENSIONS.items():
        if ext in extensions:
            return subdir
    return "other"


@dataclass
class GeneratedOutput:
    """A single generated output (image or video)."""

    asset_url: str
    asset_type: str  # "image" or "video"
    prompt: str = ""
    tool_name: str = ""
    model_id: str = ""
    timestamp: str = ""
    local_path: str = ""  # Local file path after download

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedOutput":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class Artifact:
    """A file or resource created during a session."""

    path: str
    artifact_type: str  # "image", "video", "research", "skill", "download", "other"
    description: str = ""
    timestamp: str = ""
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class SessionMetadata:
    """Metadata for a session (stored in index for quick lookups)."""

    session_id: str
    task_input: str
    status: str  # "completed", "error", "abandoned"
    created_at: str
    updated_at: str
    num_outputs: int = 0
    num_artifacts: int = 0
    tools_used: List[str] = field(default_factory=list)
    models_used: List[str] = field(default_factory=list)
    has_input_assets: bool = False
    token_usage: Optional[Dict[str, int]] = None
    title: str = ""
    dir_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMetadata":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @property
    def display_name(self) -> str:
        if self.title:
            return self.title
        if self.task_input:
            return self.task_input[:60]
        return self.session_id[:8]


@dataclass
class Session:
    """Complete session data."""

    metadata: SessionMetadata
    generated_outputs: List[GeneratedOutput] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    input_assets: List[str] = field(default_factory=list)
    steps_summary: str = ""
    errors: List[str] = field(default_factory=list)
    message_history: Optional[List[Any]] = None

    def add_artifact(self, path: str, artifact_type: str, description: str = "", url: str = "") -> None:
        self.artifacts.append(Artifact(
            path=path,
            artifact_type=artifact_type,
            description=description,
            timestamp=datetime.now().isoformat(),
            url=url,
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "generated_outputs": [o.to_dict() for o in self.generated_outputs],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "input_assets": self.input_assets,
            "steps_summary": self.steps_summary,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        outputs = [GeneratedOutput.from_dict(o) for o in data.get("generated_outputs", [])]
        artifacts = [Artifact.from_dict(a) for a in data.get("artifacts", [])]
        return cls(
            metadata=SessionMetadata.from_dict(data["metadata"]),
            generated_outputs=outputs,
            artifacts=artifacts,
            input_assets=data.get("input_assets", []),
            steps_summary=data.get("steps_summary", ""),
            errors=data.get("errors", []),
        )


class SessionManager:
    """Manages session storage and retrieval with file locking and caching."""

    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize SessionManager.

        Args:
            storage_dir: Directory to store sessions. Uses config if not specified,
                        defaults to ~/.okonomi/sessions if config is also empty.
        """
        if storage_dir is None:
            # Try to get from config
            config_dir = config.context.sessions.storage_dir
            if config_dir:
                storage_dir = Path(config_dir)
            else:
                storage_dir = Path.home() / ".okonomi" / "sessions"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Create index file for quick session lookups
        self.index_file = self.storage_dir / "index.json"
        self._ensure_index()

        # Index caching
        self._index_cache: Optional[List[Dict[str, Any]]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(seconds=30)

    @contextmanager
    def _file_lock(self, file_path: Path, exclusive: bool = True) -> Generator[None, None, None]:
        """Acquire a file lock for safe concurrent access.

        Args:
            file_path: Path to the file to lock
            exclusive: If True, acquire exclusive lock; otherwise shared lock
        """
        lock_path = file_path.with_suffix(file_path.suffix + ".lock")
        lock_file = open(lock_path, "w")

        try:
            lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock_file.fileno(), lock_type)
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def _is_cache_valid(self) -> bool:
        """Check if the index cache is still valid."""
        if self._index_cache is None or self._cache_timestamp is None:
            return False
        return datetime.now() - self._cache_timestamp < self._cache_ttl

    def _invalidate_cache(self) -> None:
        """Invalidate the index cache."""
        self._index_cache = None
        self._cache_timestamp = None

    def _ensure_index(self) -> None:
        """Ensure the index file exists."""
        if not self.index_file.exists():
            self._write_index([])

    def _read_index(self) -> List[Dict[str, Any]]:
        """Read the session index with caching and file locking."""
        # Return cached index if still valid
        if self._is_cache_valid():
            return self._index_cache

        with self._file_lock(self.index_file, exclusive=False):
            try:
                with open(self.index_file, "r") as f:
                    self._index_cache = json.load(f)
                    self._cache_timestamp = datetime.now()
                    return self._index_cache
            except json.JSONDecodeError:
                # Handle corrupted index - recover from session files
                return self._recover_index()

    def _write_index(self, index: List[Dict[str, Any]]) -> None:
        """Write the session index with file locking and atomic write."""
        self._invalidate_cache()

        with self._file_lock(self.index_file, exclusive=True):
            # Write to temp file first, then atomic rename
            temp_file = self.index_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(index, f, indent=2)
            temp_file.rename(self.index_file)

    def _recover_index(self) -> List[Dict[str, Any]]:
        """Recover index by scanning session folders."""
        index = []
        for session_dir in self.storage_dir.iterdir():
            if not session_dir.is_dir():
                continue
            session_file = session_dir / "session.json"
            if not session_file.exists():
                continue
            try:
                with open(session_file, "r") as f:
                    data = json.load(f)
                    if "metadata" in data:
                        index.append(data["metadata"])
            except (json.JSONDecodeError, KeyError):
                continue

        index.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        self._write_index(index)
        self._index_cache = index
        self._cache_timestamp = datetime.now()
        return index

    def _update_index(self, metadata: SessionMetadata) -> None:
        """Update the index with new session metadata."""
        index = self._read_index()

        # Remove existing entry if present
        index = [entry for entry in index if entry["session_id"] != metadata.session_id]

        # Add new entry
        index.append(metadata.to_dict())

        # Sort by created_at (newest first)
        index.sort(key=lambda x: x["created_at"], reverse=True)

        self._write_index(index)

    def create_session_id(self) -> str:
        """Generate a new session ID: YYYY-MM-DD_shortUUID."""
        date_prefix = datetime.now().strftime("%Y-%m-%d")
        short_uuid = uuid.uuid4().hex[:8]
        return f"{date_prefix}_{short_uuid}"

    def _dir_name_for_session(self, session_id: str, title: str = "") -> str:
        """Derive a directory name from session ID and optional title."""
        if title:
            slug = _slugify(title)
            if slug:
                date_part = session_id[:10] if len(session_id) > 10 else datetime.now().strftime("%Y-%m-%d")
                return f"{date_part}_{slug}"
        return session_id

    def set_title(self, session_id: str, title: str) -> bool:
        """Set a title for a session, renaming its directory."""
        old_dir = self.get_session_dir(session_id)
        if not old_dir.exists():
            return False

        new_dir_name = self._dir_name_for_session(session_id, title)
        new_dir = self.storage_dir / new_dir_name

        if new_dir.exists() and new_dir != old_dir:
            return False

        if old_dir != new_dir:
            old_dir.rename(new_dir)

        session_file = new_dir / "session.json"
        if session_file.exists():
            with open(session_file, "r") as f:
                data = json.load(f)
            data["metadata"]["title"] = title
            data["metadata"]["dir_name"] = new_dir_name
            with open(session_file, "w") as f:
                json.dump(data, f, indent=2)

        index = self._read_index()
        for entry in index:
            if entry["session_id"] == session_id:
                entry["title"] = title
                entry["dir_name"] = new_dir_name
                break
        self._write_index(index)
        return True

    def create_session(self, task_input: str) -> Session:
        """Create a new session with initialized metadata.

        Args:
            task_input: The initial task/request

        Returns:
            A new Session object
        """
        now = datetime.now().isoformat()
        metadata = SessionMetadata(
            session_id=self.create_session_id(),
            task_input=task_input,
            status="in_progress",
            created_at=now,
            updated_at=now,
        )
        return Session(metadata=metadata)

    def get_session_dir(self, session_id: str) -> Path:
        """Get the directory path for a session.

        Checks the index for a dir_name first (supports titled/renamed dirs),
        falls back to using session_id as the directory name.
        """
        index = self._read_index()
        for entry in index:
            if entry["session_id"] == session_id and entry.get("dir_name"):
                return self.storage_dir / entry["dir_name"]
        return self.storage_dir / session_id

    def get_asset_dir(self, session_id: str, asset_type: str = "") -> Path:
        """Get the asset subdirectory for a session, creating it if needed."""
        session_dir = self.get_session_dir(session_id)
        if asset_type:
            d = session_dir / "assets" / asset_type
        else:
            d = session_dir / "assets"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_asset_path(self, session_id: str, filename: str) -> Path:
        """Get the full path for an asset file, routed to the correct subdirectory."""
        subdir = asset_subdir_for_file(filename)
        return self.get_asset_dir(session_id, subdir) / filename

    def save_session(self, session: Session) -> None:
        """Save a session to disk.

        Writes session.json for metadata/outputs and messages.jsonl for
        the conversation history (one JSON object per message).
        """
        sid = session.metadata.session_id
        if not session.metadata.dir_name:
            session.metadata.dir_name = self._dir_name_for_session(sid, session.metadata.title)

        session_dir = self.storage_dir / session.metadata.dir_name
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "assets").mkdir(exist_ok=True)
        session_file = session_dir / "session.json"

        def _safe_default(obj):
            try:
                return str(obj)
            except Exception:
                return f"<{type(obj).__name__}>"

        with open(session_file, "w") as f:
            json.dump(session.to_dict(), f, indent=2, default=_safe_default)

        if session.message_history:
            from pydantic_ai.messages import ModelMessagesTypeAdapter

            messages_file = session_dir / "messages.jsonl"
            serialized = json.loads(ModelMessagesTypeAdapter.dump_json(session.message_history))
            with open(messages_file, "w") as f:
                for msg in serialized:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        self._update_index(session.metadata)

    def load_session(self, session_id: str) -> Optional[Session]:
        """Load a session from disk.

        Reads session.json and, if present, messages.jsonl to restore
        the full conversation history.

        Args:
            session_id: ID of the session to load

        Returns:
            Session if found, None otherwise
        """
        session_dir = self.get_session_dir(session_id)
        session_file = session_dir / "session.json"

        if not session_file.exists():
            return None

        with open(session_file, "r") as f:
            data = json.load(f)

        session = Session.from_dict(data)

        messages_file = session_dir / "messages.jsonl"
        if messages_file.exists():
            try:
                from pydantic_ai.messages import ModelMessagesTypeAdapter

                raw_messages = []
                with open(messages_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            raw_messages.append(json.loads(line))
                session.message_history = ModelMessagesTypeAdapter.validate_json(json.dumps(raw_messages))
            except Exception:
                pass

        return session

    def list_sessions(self, limit: Optional[int] = None, status: Optional[str] = None) -> List[SessionMetadata]:
        """List all sessions.

        Args:
            limit: Maximum number of sessions to return
            status: Filter by status (completed, interrupted, max_steps_error)

        Returns:
            List of session metadata, sorted by timestamp (newest first)
        """
        index = self._read_index()

        # Filter by status if specified
        if status:
            index = [entry for entry in index if entry.get("status") == status]

        # Apply limit
        if limit:
            index = index[:limit]

        return [SessionMetadata.from_dict(entry) for entry in index]

    def get_latest_session(self) -> Optional[Session]:
        """Get the most recent session.

        Returns:
            Latest session if any exist, None otherwise
        """
        sessions = self.list_sessions(limit=1)

        if not sessions:
            return None

        return self.load_session(sessions[0].session_id)

    def resolve_session(self, query: str) -> Optional[SessionMetadata]:
        """Resolve a query to a session. Tries: exact ID, prefix, title, fuzzy."""
        index = self._read_index()
        q = query.strip()

        for entry in index:
            if entry["session_id"] == q:
                return SessionMetadata.from_dict(entry)

        for entry in index:
            if entry["session_id"].startswith(q):
                return SessionMetadata.from_dict(entry)

        q_lower = q.lower()
        for entry in index:
            title = entry.get("title", "")
            if title and q_lower in title.lower():
                return SessionMetadata.from_dict(entry)

        results = self.search_sessions(q, limit=1)
        return results[0] if results else None

    def search_sessions(self, query: str, limit: Optional[int] = None) -> List[SessionMetadata]:
        """Search sessions by task content, session ID, title, or tools/models used."""
        index = self._read_index()
        words = query.lower().split()
        if not words:
            return []

        scored: list[tuple[int, dict]] = []
        for entry in index:
            searchable = " ".join([
                entry.get("session_id", ""),
                entry.get("task_input", ""),
                entry.get("title", ""),
                " ".join(entry.get("tools_used", [])),
                " ".join(entry.get("models_used", [])),
            ]).lower()

            if all(w in searchable for w in words):
                score = sum(searchable.count(w) for w in words)
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [entry for _, entry in scored]

        if limit:
            results = results[:limit]

        return [SessionMetadata.from_dict(entry) for entry in results]

    def prune_empty_sessions(self, min_age_hours: int = 24) -> int:
        """Delete sessions with no outputs, no artifacts, and no extra files.

        Args:
            min_age_hours: Only prune sessions older than this (default 24h).
        """
        index = self._read_index()
        cutoff = datetime.now() - timedelta(hours=min_age_hours)
        to_delete = []

        for entry in index:
            if entry.get("num_outputs", 0) > 0 or entry.get("num_artifacts", 0) > 0:
                continue

            created = entry.get("created_at", "")
            if created:
                try:
                    if datetime.fromisoformat(created) > cutoff:
                        continue
                except ValueError:
                    pass

            session_dir = self.get_session_dir(entry["session_id"])
            if not session_dir.exists():
                to_delete.append(entry["session_id"])
                continue
            files = {f.name for f in session_dir.rglob("*") if f.is_file() and not f.name.startswith(".")}
            files -= {"session.json", "messages.jsonl", "index.json.lock"}
            if not files:
                to_delete.append(entry["session_id"])

        for sid in to_delete:
            session_dir = self.get_session_dir(sid)
            if session_dir.exists():
                shutil.rmtree(session_dir)

        if to_delete:
            remaining = [e for e in index if e["session_id"] not in set(to_delete)]
            self._write_index(remaining)
            self._invalidate_cache()

        return len(to_delete)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its artifacts.

        Args:
            session_id: ID of the session to delete

        Returns:
            True if deleted, False if not found
        """
        session_dir = self.get_session_dir(session_id)

        if not session_dir.exists():
            return False

        # Delete entire session folder
        shutil.rmtree(session_dir)

        # Update index
        index = self._read_index()
        index = [entry for entry in index if entry["session_id"] != session_id]
        self._write_index(index)

        return True

    def get_session_outputs(self, session_id: str, asset_type: Optional[str] = None) -> List[GeneratedOutput]:
        """Get all outputs from a session.

        Args:
            session_id: ID of the session
            asset_type: Filter by type ("image" or "video"), or None for all

        Returns:
            List of generated outputs
        """
        session = self.load_session(session_id)

        if not session:
            return []

        if asset_type:
            return [o for o in session.generated_outputs if o.asset_type == asset_type]

        return session.generated_outputs

    def search_outputs(
        self,
        query: Optional[str] = None,
        asset_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search outputs across all sessions.

        Args:
            query: Search query for task/prompt content
            asset_type: Filter by type ("image" or "video"), or None for all
            limit: Maximum number of results

        Returns:
            List of outputs with session metadata
        """
        results = []

        for session_meta in self.list_sessions():
            session = self.load_session(session_meta.session_id)
            if not session:
                continue

            # Filter by query if specified
            if query and query.lower() not in session.metadata.task_input.lower():
                continue

            for output in session.generated_outputs:
                # Filter by asset type
                if asset_type and output.asset_type != asset_type:
                    continue

                results.append(
                    {
                        "session_id": session.metadata.session_id,
                        "task_input": session.metadata.task_input,
                        "output": output.to_dict(),
                        "created_at": session.metadata.created_at,
                    }
                )

        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x["created_at"], reverse=True)

        if limit:
            results = results[:limit]

        return results
