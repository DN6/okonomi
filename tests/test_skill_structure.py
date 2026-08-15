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


def test_duplicate_slugs_are_a_hard_error():
    body = "## Overview\nfirst\n\n## Overview\nsecond\n"
    err, warnings = _check_skill_structure(_mk(body))
    assert err is not None
    assert "overview" in err
    assert warnings == []


def test_heading_collision_across_levels_is_an_error():
    body = "## Workflow\ntop\n\n### Workflow\nnested\n"
    err, _ = _check_skill_structure(_mk(body))
    assert err is not None
    assert "workflow" in err


def test_no_sections_with_long_body_warns():
    body = "x" * (skills.MAX_INTRO_CHARS + 1)
    err, warnings = _check_skill_structure(_mk(body))
    assert err is None
    assert len(warnings) == 1
    assert "no '## '" in warnings[0]


def test_long_preamble_before_first_section_warns():
    body = "y" * (skills.MAX_INTRO_CHARS + 1) + "\n\n## Real\ncontent\n"
    err, warnings = _check_skill_structure(_mk(body))
    assert err is None
    assert len(warnings) == 1
    assert "before the first" in warnings[0]


def test_well_formed_skill_has_no_error_or_warning():
    body = "Short preamble.\n\n## Setup\ndo x\n\n## Usage\ndo y\n"
    err, warnings = _check_skill_structure(_mk(body))
    assert err is None
    assert warnings == []


def test_tiny_sectionless_skill_is_allowed_silently():
    body = "Just a couple lines, nothing to disclose.\n"
    err, warnings = _check_skill_structure(_mk(body))
    assert err is None
    assert warnings == []


def test_create_skill_rejects_duplicate_slugs(tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "USER_SKILLS_DIR", tmp_path)
    tool = SkillTools()
    result = tool.create_skill(
        "dupe", _mk("## Overview\na\n\n## Overview\nb\n")
    )
    assert result["success"] is False
    assert "Duplicate" in result["error"]
    assert not (tmp_path / "dupe").exists()


def test_create_skill_succeeds_with_warnings(tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "USER_SKILLS_DIR", tmp_path)
    tool = SkillTools()
    result = tool.create_skill("noheadings", _mk("z" * (skills.MAX_INTRO_CHARS + 1)))
    assert result["success"] is True
    assert result["warnings"]
    assert (tmp_path / "noheadings" / "SKILL.md").exists()


def test_create_skill_clean_has_no_warnings_key(tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "USER_SKILLS_DIR", tmp_path)
    tool = SkillTools()
    result = tool.create_skill("clean", _mk("Preamble.\n\n## A\nx\n\n## B\ny\n"))
    assert result["success"] is True
    assert "warnings" not in result
