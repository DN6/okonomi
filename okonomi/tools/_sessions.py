"""Session management tools for Okonomi.

Provides tools for listing, searching, and viewing past sessions and images.
"""

from typing import Optional

from okonomi.session import SessionManager
from okonomi.tools._metadata import ToolMetadataMixin


class SessionTools(ToolMetadataMixin):
    __namespace__ = "sessions"
    __description__ = "Tools for managing and searching past generation sessions and images"

    def __init__(self, config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.manager = SessionManager()

    async def list_sessions(self, limit: int = 10, status: Optional[str] = None) -> dict:
        """List past generation sessions.

        Args:
            limit: Maximum number of sessions to return (default: 10).
            status: Filter by status - "completed", "interrupted", or "max_steps_error".

        Returns:
            Dict with list of sessions containing metadata (id, task, timestamp, status, num_outputs).
        """
        try:
            sessions = self.manager.list_sessions(limit=limit, status=status)

            results = [
                {
                    "session_id": s.session_id,
                    "task": s.task,
                    "timestamp": s.timestamp,
                    "status": s.status,
                    "num_outputs": s.num_outputs,
                    "model_id": s.model_id,
                    "total_steps": s.total_steps,
                }
                for s in sessions
            ]

            return {"sessions": results}

        except Exception as e:
            return {"error": str(e)}

    async def get_session_details(self, session_id: str) -> dict:
        """Get detailed information about a specific session.

        Args:
            session_id: ID of the session to retrieve.

        Returns:
            Dict with complete session data including metadata, outputs, and memory state.
        """
        try:
            session = self.manager.load_session(session_id)

            if not session:
                return {"error": "Session not found"}

            return session.to_dict()

        except Exception as e:
            return {"error": str(e)}

    async def search_sessions(self, query: str, limit: int = 10) -> dict:
        """Search past sessions by task description.

        Args:
            query: Search query to match against task descriptions.
            limit: Maximum number of results to return (default: 10).

        Returns:
            Dict with list of matching sessions with metadata.
        """
        try:
            sessions = self.manager.search_sessions(query=query, limit=limit)

            results = [
                {
                    "session_id": s.session_id,
                    "task": s.task,
                    "timestamp": s.timestamp,
                    "status": s.status,
                    "num_outputs": s.num_outputs,
                    "model_id": s.model_id,
                }
                for s in sessions
            ]

            return {"sessions": results}

        except Exception as e:
            return {"error": str(e)}

    async def view_session_images(self, session_id: str) -> dict:
        """View all images generated in a specific session.

        Args:
            session_id: ID of the session.

        Returns:
            Dict with list of images containing URLs, prompts, and scores.
        """
        try:
            images = self.manager.get_session_images(session_id)

            if not images:
                return {"message": "No images found in this session", "images": []}

            return {"images": images}

        except Exception as e:
            return {"error": str(e)}

    async def search_images(self, query: Optional[str] = None, min_score: float = 0.0, limit: int = 20) -> dict:
        """Search for images across all sessions.

        Args:
            query: Optional search query to match against session tasks and prompts.
            min_score: Minimum visual critique score (0-10, default: 0).
            limit: Maximum number of results (default: 20).

        Returns:
            Dict with list of images with metadata, sorted by score (highest first).
        """
        try:
            images = self.manager.search_images(query=query, min_score=min_score, limit=limit)

            if not images:
                return {"message": "No images found matching criteria", "images": []}

            return {"images": images}

        except Exception as e:
            return {"error": str(e)}

    async def get_latest_session(self) -> dict:
        """Get the most recent generation session.

        Returns:
            Dict with the latest session's metadata and outputs, or error if none exist.
        """
        try:
            session = self.manager.get_latest_session()

            if not session:
                return {"error": "No sessions found"}

            return session.to_dict()

        except Exception as e:
            return {"error": str(e)}

    async def delete_session(self, session_id: str) -> dict:
        """Delete a session and all its data.

        Args:
            session_id: ID of the session to delete.

        Returns:
            Dict with success/error message.
        """
        try:
            success = self.manager.delete_session(session_id)

            if success:
                return {"status": "success", "message": f"Session {session_id} deleted"}
            else:
                return {"status": "error", "message": "Session not found"}

        except Exception as e:
            return {"error": str(e)}
