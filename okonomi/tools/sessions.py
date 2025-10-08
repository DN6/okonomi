"""Session management MCP tools for Okonomi.

Provides tools for listing, searching, and viewing past sessions and images.
"""

import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from ..config import config
from ..session import SessionManager

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

mcp_logger = logging.getLogger("mcp")
mcp_logger.setLevel(logging.WARNING)

os.environ["MCP_TIMEOUT"] = str(config.tools.mcp.timeout)

mcp = FastMCP("session_tools")


@mcp.tool()
def list_sessions(limit: int = 10, status: Optional[str] = None) -> str:
    """List past generation sessions.

    Args:
        limit: Maximum number of sessions to return (default: 10).
        status: Filter by status - "completed", "interrupted", or "max_steps_error".

    Returns:
        JSON list of sessions with metadata (id, task, timestamp, status, num_outputs).
    """
    manager = SessionManager()
    sessions = manager.list_sessions(limit=limit, status=status)

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

    return json.dumps(results)


@mcp.tool()
def get_session_details(session_id: str) -> str:
    """Get detailed information about a specific session.

    Args:
        session_id: ID of the session to retrieve.

    Returns:
        JSON with complete session data including metadata, outputs, and memory state.
    """
    manager = SessionManager()
    session = manager.load_session(session_id)

    if not session:
        return json.dumps({"error": "Session not found"})

    return json.dumps(session.to_dict())


@mcp.tool()
def search_sessions(query: str, limit: int = 10) -> str:
    """Search past sessions by task description.

    Args:
        query: Search query to match against task descriptions.
        limit: Maximum number of results to return (default: 10).

    Returns:
        JSON list of matching sessions with metadata.
    """
    manager = SessionManager()
    sessions = manager.search_sessions(query=query, limit=limit)

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

    return json.dumps(results)


@mcp.tool()
def view_session_images(session_id: str) -> str:
    """View all images generated in a specific session.

    Args:
        session_id: ID of the session.

    Returns:
        JSON list of images with URLs, prompts, and scores.
    """
    manager = SessionManager()
    images = manager.get_session_images(session_id)

    if not images:
        return json.dumps({"message": "No images found in this session"})

    return json.dumps(images)


@mcp.tool()
def search_images(query: Optional[str] = None, min_score: float = 0.0, limit: int = 20) -> str:
    """Search for images across all sessions.

    Args:
        query: Optional search query to match against session tasks and prompts.
        min_score: Minimum visual critique score (0-10, default: 0).
        limit: Maximum number of results (default: 20).

    Returns:
        JSON list of images with metadata, sorted by score (highest first).
    """
    manager = SessionManager()
    images = manager.search_images(query=query, min_score=min_score, limit=limit)

    if not images:
        return json.dumps({"message": "No images found matching criteria"})

    return json.dumps(images)


@mcp.tool()
def get_latest_session() -> str:
    """Get the most recent generation session.

    Returns:
        JSON with the latest session's metadata and outputs, or error if none exist.
    """
    manager = SessionManager()
    session = manager.get_latest_session()

    if not session:
        return json.dumps({"error": "No sessions found"})

    return json.dumps(session.to_dict())


@mcp.tool()
def delete_session(session_id: str) -> str:
    """Delete a session and all its data.

    Args:
        session_id: ID of the session to delete.

    Returns:
        JSON with success/error message.
    """
    manager = SessionManager()
    success = manager.delete_session(session_id)

    if success:
        return json.dumps({"status": "success", "message": f"Session {session_id} deleted"})
    else:
        return json.dumps({"status": "error", "message": "Session not found"})


if __name__ == "__main__":
    mcp.run()
