import pytest

from okonomi import db, mcp
from okonomi import session as session_mod
from okonomi.agent.agent import OkonomiAgent


@pytest.fixture
def okonomi_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("OKONOMI_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(db, "_connection", db.open_db(tmp_path / "test.db"))
    monkeypatch.setattr(session_mod, "OKONOMI_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(mcp, "OKONOMI_MCP_CONFIG", tmp_path / "mcp.json")
    return OkonomiAgent()
