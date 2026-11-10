"""Section-structure enforcement on skill writes.

Okonomi skills load section by section (load_skill lists ## H2 slugs,
load_skill_section fetches one). These guard that create/write paths
reject structures that break that loading (duplicate slugs → unreachable
sections) and warn on structures that load everything inline.
"""

from okonomi.tools import skills
from okonomi.tools.skills import SkillTools, _check_skill_structure


def _mk(body: str) -> str:
    return f"---\nname: demo\ndescription: d\n---\n{body}"


def test_no_sections_with_long_body_warns():
    body = "x" * (skills.MAX_INTRO_CHARS + 1)
    err, warnings = _check_skill_structure(_mk(body))
    assert err is None
    assert len(warnings) == 1
    assert "no '## '" in warnings[0]


def test_well_formed_skill_has_no_error_or_warning():
    body = "Short preamble.\n\n## Setup\ndo x\n\n## Usage\ndo y\n"
    err, warnings = _check_skill_structure(_mk(body))
    assert err is None
    assert warnings == []


def test_create_skill_rejects_duplicate_slugs(store):
    result = SkillTools().create_skill("dupe", _mk("## Overview\na\n\n## Overview\nb\n"))
    assert result["success"] is False
    assert "Duplicate" in result["error"]
    assert not (skills.USER_SKILLS_DIR / "dupe").exists()
