"""Session management for Okonomi.

This module provides session storage and retrieval functionality to enable
resuming from previous runs, viewing past outputs, and searching through
generation history.
"""

import fcntl
import json
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator, List, Optional, Dict, Any

from .config import config


@dataclass
class GeneratedOutput:
    """A single generated output (image or video)."""

    asset_url: str
    asset_type: str  # "image" or "video"
    prompt: str = ""
    tool_name: str = ""
    model_id: str = ""
    loras: List[tuple] = field(default_factory=list)
    timestamp: str = ""
    local_path: str = ""  # Local file path after download

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedOutput":
        # Handle older sessions without local_path
        if "local_path" not in data:
            data["local_path"] = ""
        return cls(**data)


@dataclass
class SessionMetadata:
    """Metadata for a session (stored in index for quick lookups)."""

    session_id: str
    task_input: str
    status: str  # "completed", "error", "abandoned"
    created_at: str
    updated_at: str
    num_outputs: int = 0
    tools_used: List[str] = field(default_factory=list)
    models_used: List[str] = field(default_factory=list)
    has_input_assets: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMetadata":
        return cls(**data)


@dataclass
class Session:
    """Complete session data."""

    metadata: SessionMetadata
    generated_outputs: List[GeneratedOutput] = field(default_factory=list)
    input_assets: List[str] = field(default_factory=list)  # URLs/paths of input images/videos
    steps_summary: str = ""  # LLM-generated summary of actions + user feedback
    loras_used: List[tuple] = field(default_factory=list)  # [(lora_url, scale), ...]
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "generated_outputs": [o.to_dict() for o in self.generated_outputs],
            "input_assets": self.input_assets,
            "steps_summary": self.steps_summary,
            "loras_used": self.loras_used,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        outputs = [GeneratedOutput.from_dict(o) for o in data.get("generated_outputs", [])]
        return cls(
            metadata=SessionMetadata.from_dict(data["metadata"]),
            generated_outputs=outputs,
            input_assets=data.get("input_assets", []),
            steps_summary=data.get("steps_summary", ""),
            loras_used=data.get("loras_used", []),
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
            config_dir = config.memory.sessions.storage_dir
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
        """Generate a new session ID."""
        return str(uuid.uuid4())

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

        Args:
            session_id: ID of the session

        Returns:
            Path to the session directory
        """
        return self.storage_dir / session_id

    def save_session(self, session: Session) -> None:
        """Save a session to disk.

        Creates a folder for the session with session.json inside.

        Args:
            session: Session to save
        """
        session_dir = self.get_session_dir(session.metadata.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        session_file = session_dir / "session.json"

        with open(session_file, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

        # Update index
        self._update_index(session.metadata)

    def load_session(self, session_id: str) -> Optional[Session]:
        """Load a session from disk.

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

        return Session.from_dict(data)

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

    def search_sessions(self, query: str, limit: Optional[int] = None) -> List[SessionMetadata]:
        """Search sessions by task content.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching session metadata
        """
        index = self._read_index()

        # Simple text search in task_input field
        query_lower = query.lower()
        matches = [entry for entry in index if query_lower in entry.get("task_input", "").lower()]

        # Apply limit
        if limit:
            matches = matches[:limit]

        return [SessionMetadata.from_dict(entry) for entry in matches]

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
