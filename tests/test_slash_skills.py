"""Slash-command invocation of skills: /<skill-name> sugar + completion.

Skill frontmatter names may contain spaces (e.g. "FLUX LoRA Recipes"), while
slash commands split on the first space — so invocation goes via a slug and
must forward the real name to the loader. These guard that mapping.
"""

from okonomi.tui import commands
from okonomi.tui.commands import Action, handle


class _StubSkills:
    def __init__(self):
        self.loaded = None

    def _load_full(self, name):
        self.loaded = name
        return {"content": f"BODY OF {name}", "name": name}


class _StubAgent:
    def __init__(self):
        self.skills = _StubSkills()


class _NullEmitter:
    def emit(self, *a, **k):
        pass


def test_spaced_skill_dispatches_by_slug_and_forwards_real_name():
    cmds = commands._skill_commands()
    spaced = next(((s, n) for s, n, _ in cmds if " " in n), None)
    if spaced is None:
        return  # no spaced-name skill installed; nothing to assert
    slug, real = spaced
    agent = _StubAgent()
    res = handle(f"/{slug} make it warmer", agent, _NullEmitter())
    assert res is not None and res.action is Action.FORWARD
    assert agent.skills.loaded == real, "loader must receive the real (spaced) name"
    assert f"BODY OF {real}" in res.payload
    assert "make it warmer" in res.payload, "trailing text must pass through as instruction"


def test_bare_slug_dispatches_without_prefix():
    cmds = commands._skill_commands()
    if not cmds:
        return  # no skills installed
    slug, real, _desc = cmds[0]
    bare = slug.split(":", 1)[1]
    agent = _StubAgent()
    res = handle(f"/{bare}", agent, _NullEmitter())
    assert res is not None and res.action is Action.FORWARD
    assert agent.skills.loaded == real, "bare slug must reach the same skill"
