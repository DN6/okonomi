"""Slash-command invocation of skills: /<skill-name> sugar + completion.

Skill frontmatter names may contain spaces (e.g. "FLUX LoRA Recipes"), while
slash commands split on the first space — so invocation goes via a slug and
must forward the real name to the loader. These guard that mapping.
"""

from types import SimpleNamespace

from okonomi.tools import skills
from okonomi.tui import commands
from okonomi.tui.commands import Action, handle


class _StubSkills:
    def __init__(self):
        self.loaded = None

    def _load_full(self, name):
        self.loaded = name
        return {"content": f"BODY OF {name}", "name": name}


def _stub_agent() -> SimpleNamespace:
    return SimpleNamespace(skills=_StubSkills())


def _null_emitter() -> SimpleNamespace:
    return SimpleNamespace(emit=lambda *a, **k: None)


def _install_user_skill(name: str) -> None:
    skill_dir = skills.USER_SKILLS_DIR / name.lower().replace(" ", "-")
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\nintro\n")


def test_spaced_skill_dispatches_by_slug_and_forwards_real_name(store):
    _install_user_skill("FLUX LoRA Recipes")
    (slug, real, _desc) = next(c for c in commands._skill_commands() if c[1] == "FLUX LoRA Recipes")
    agent = _stub_agent()

    res = handle(f"/{slug} make it warmer", agent, _null_emitter())

    assert res is not None and res.action is Action.FORWARD
    assert agent.skills.loaded == real, "loader must receive the real (spaced) name"
    assert f"BODY OF {real}" in res.payload
    assert "make it warmer" in res.payload, "trailing text must pass through as instruction"


def test_bare_slug_dispatches_without_prefix(store):
    _install_user_skill("demo")
    (slug, real, _desc) = next(c for c in commands._skill_commands() if c[1] == "demo")
    bare = slug.split(":", 1)[1]
    agent = _stub_agent()

    res = handle(f"/{bare}", agent, _null_emitter())

    assert res is not None and res.action is Action.FORWARD
    assert agent.skills.loaded == real, "bare slug must reach the same skill"
