"""Shared filesystem constants, resolved once at import.

``$OKONOMI_HOME`` overrides the root (default ``~/.okonomi``).
"""

import os
from pathlib import Path


OKONOMI_HOME = Path(os.getenv("OKONOMI_HOME", str(Path.home() / ".okonomi")))
OKONOMI_SESSIONS_DIR = OKONOMI_HOME / "sessions"
OKONOMI_RESEARCH_DIR = OKONOMI_HOME / "research"
OKONOMI_SKILLS_DIR = OKONOMI_HOME / "skills"
OKONOMI_FAVOURITES_DIR = OKONOMI_HOME / "favourites"
OKONOMI_LOVED_DIR = OKONOMI_HOME / "loved"
