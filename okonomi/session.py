"""Session management for Okonomi.

This module provides session storage and retrieval functionality to enable
resuming from previous runs, viewing past outputs, and searching through
generation history.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from .config import config


@dataclass
class SessionMetadata:
    """Metadata for a session."""

    session_id: str
    task: str
    timestamp: str
    status: str  # "completed", "interrupted", "max_steps_error"
    num_outputs: int = 0
    model_id: str = ""
    total_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMetadata":
        return cls(**data)


@dataclass
class Session:
    """Complete session data including agent state and outputs."""

    metadata: SessionMetadata
    generated_outputs: List[Dict[str, Any]] = field(default_factory=list)
    agent_memory_state: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "generated_outputs": self.generated_outputs,
            "agent_memory_state": self.agent_memory_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            metadata=SessionMetadata.from_dict(data["metadata"]),
            generated_outputs=data.get("generated_outputs", []),
            agent_memory_state=data.get("agent_memory_state"),
        )


class SessionManager:
    """Manages session storage and retrieval."""

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

    def _ensure_index(self) -> None:
        """Ensure the index file exists."""
        if not self.index_file.exists():
            self._write_index([])

    def _read_index(self) -> List[Dict[str, Any]]:
        """Read the session index."""
        with open(self.index_file, "r") as f:
            return json.load(f)

    def _write_index(self, index: List[Dict[str, Any]]) -> None:
        """Write the session index."""
        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)

    def _update_index(self, metadata: SessionMetadata) -> None:
        """Update the index with new session metadata."""
        index = self._read_index()

        # Remove existing entry if present
        index = [entry for entry in index if entry["session_id"] != metadata.session_id]

        # Add new entry
        index.append(metadata.to_dict())

        # Sort by timestamp (newest first)
        index.sort(key=lambda x: x["timestamp"], reverse=True)

        self._write_index(index)

    def create_session_id(self) -> str:
        """Generate a new session ID."""
        return str(uuid.uuid4())

    def save_session(self, session: Session) -> None:
        """Save a session to disk.

        Args:
            session: Session to save
        """
        session_file = self.storage_dir / f"{session.metadata.session_id}.json"

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
        session_file = self.storage_dir / f"{session_id}.json"

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

        # Simple text search in task field
        query_lower = query.lower()
        matches = [entry for entry in index if query_lower in entry.get("task", "").lower()]

        # Apply limit
        if limit:
            matches = matches[:limit]

        return [SessionMetadata.from_dict(entry) for entry in matches]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: ID of the session to delete

        Returns:
            True if deleted, False if not found
        """
        session_file = self.storage_dir / f"{session_id}.json"

        if not session_file.exists():
            return False

        # Delete file
        session_file.unlink()

        # Update index
        index = self._read_index()
        index = [entry for entry in index if entry["session_id"] != session_id]
        self._write_index(index)

        return True

    def get_session_images(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all images from a session.

        Args:
            session_id: ID of the session

        Returns:
            List of image data with URLs and metadata
        """
        session = self.load_session(session_id)

        if not session:
            return []

        images = []
        for output in session.generated_outputs:
            if "image_url" in output or "image" in output:
                images.append(
                    {
                        "image_url": output.get("image_url") or output.get("image"),
                        "prompt": output.get("prompt", ""),
                        "tool_name": output.get("tool_name", ""),
                        "concept_id": output.get("concept_id", ""),
                        "visual_critique_score": output.get("visual_critique_score", 0),
                    }
                )

        return images

    def search_images(
        self, query: Optional[str] = None, min_score: float = 0.0, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search images across all sessions.

        Args:
            query: Search query for task/prompt content
            min_score: Minimum visual critique score
            limit: Maximum number of results

        Returns:
            List of images with metadata
        """
        all_images = []

        # Get all sessions
        sessions = self.list_sessions()

        for session_meta in sessions:
            session = self.load_session(session_meta.session_id)
            if not session:
                continue

            # Filter by query if specified
            if query and query.lower() not in session.metadata.task.lower():
                continue

            # Get images from this session
            for output in session.generated_outputs:
                score = output.get("visual_critique_score", 0)

                if score < min_score:
                    continue

                if "image_url" in output or "image" in output:
                    all_images.append(
                        {
                            "session_id": session.metadata.session_id,
                            "session_task": session.metadata.task,
                            "image_url": output.get("image_url") or output.get("image"),
                            "prompt": output.get("prompt", ""),
                            "tool_name": output.get("tool_name", ""),
                            "concept_id": output.get("concept_id", ""),
                            "visual_critique_score": score,
                            "timestamp": session.metadata.timestamp,
                        }
                    )

        # Sort by score (highest first)
        all_images.sort(key=lambda x: x.get("visual_critique_score", 0), reverse=True)

        # Apply limit
        if limit:
            all_images = all_images[:limit]

        return all_images
