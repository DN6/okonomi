"""Session management tools for Okonomi.

Provides tools for listing and loading past sessions. For searching
session content, use the terminal search tool to grep through
~/.okonomi/sessions/.
"""

from typing import Optional

from okonomi.session import SessionManager
from okonomi.tools._metadata import ToolMetadataMixin


class SessionTools(ToolMetadataMixin):
    __namespace__ = "sessions"
    __description__ = "Tools for listing and loading past generation sessions"

    def __init__(self, config):
        self.config = config
        self.manager = SessionManager()

    def get_session_details(self, session_id: str) -> dict:
        """Get detailed information about a specific session.

        Args:
            session_id: ID of the session to retrieve.

        Returns:
            Dict with complete session data including metadata, outputs, and summary.
        """
        try:
            session = self.manager.load_session(session_id)
            if not session:
                return {"error": "Session not found"}
            return session.to_dict()
        except Exception as e:
            return {"error": str(e)}

    def get_session_summary(self, session_id: str) -> dict:
        """Get the steps summary for a specific session.

        Args:
            session_id: ID of the session.

        Returns:
            Dict with the session's task, status, and steps summary.
        """
        try:
            session = self.manager.load_session(session_id)
            if not session:
                return {"error": "Session not found"}
            return {
                "session_id": session_id,
                "task_input": session.metadata.task_input,
                "status": session.metadata.status,
                "steps_summary": session.steps_summary or "No summary available.",
            }
        except Exception as e:
            return {"error": str(e)}

    def get_session_images(self, session_id: str) -> dict:
        """Get all images generated in a specific session.

        Args:
            session_id: ID of the session.

        Returns:
            Dict with list of images containing URLs, prompts, and model info.
        """
        try:
            session = self.manager.load_session(session_id)
            if not session:
                return {"error": "Session not found"}

            images = [o.to_dict() for o in session.generated_outputs if o.asset_type == "image"]
            return {"images": images}
        except Exception as e:
            return {"error": str(e)}

    def get_session_videos(self, session_id: str) -> dict:
        """Get all videos generated in a specific session.

        Args:
            session_id: ID of the session.

        Returns:
            Dict with list of videos containing URLs, prompts, and model info.
        """
        try:
            session = self.manager.load_session(session_id)
            if not session:
                return {"error": "Session not found"}

            videos = [o.to_dict() for o in session.generated_outputs if o.asset_type == "video"]
            return {"videos": videos}
        except Exception as e:
            return {"error": str(e)}
