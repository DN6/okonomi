import pytest

from okonomi import db, mcp
from okonomi import session as session_mod
from okonomi.agent.agent import OkonomiAgent
from okonomi.tools import files as files_mod
from okonomi.tools import skills as skills_mod
from okonomi.tui import artifacts as artifacts_mod
from okonomi.tui import commands as commands_mod


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A throwaway okonomi home: temp database, sessions, mind, skills,
    favorites and MCP config all under ``tmp_path``, so no test reads or
    writes ``~/.okonomi``. Yields the open connection."""
    monkeypatch.setenv("OKONOMI_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    conn = db.open_db(tmp_path / "okonomi.db")
    monkeypatch.setattr(db, "_connection", conn)
    monkeypatch.setattr(session_mod, "OKONOMI_SESSIONS_DIR", tmp_path / "sessions")
    mind = tmp_path / "mind"
    mind.mkdir()
    monkeypatch.setattr(session_mod, "OKONOMI_MIND_DIR", mind)
    monkeypatch.setattr(files_mod, "OKONOMI_MIND_DIR", mind)
    monkeypatch.setattr(skills_mod, "USER_SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(commands_mod, "USER_SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(artifacts_mod, "FAVORITES_DIR", tmp_path / "favorites")
    monkeypatch.setattr(mcp, "OKONOMI_MCP_CONFIG", tmp_path / "mcp.json")
    yield conn
    conn.close()


@pytest.fixture
def okonomi_agent(store):
    return OkonomiAgent()
