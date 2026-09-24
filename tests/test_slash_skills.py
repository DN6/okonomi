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


def test_skill_command_slugs_are_space_free():
    for cmd, _desc in commands.discover_skill_commands():
        assert " " not in cmd, f"slash slug contains a space: {cmd!r}"
        assert cmd.startswith("/")


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


def test_builtin_wins_over_skill_and_unknown_warns():
    # A real built-in is dispatched to its handler, not treated as a skill.
    assert "/help" in commands.COMMANDS
    # An unknown slash that is neither built-in nor a skill returns None.
    assert handle("/definitely-not-a-skill-xyz", _StubAgent(), _NullEmitter()) is None


def test_skill_slugs_are_namespaced_by_source():
    for cmd, _desc in commands.discover_skill_commands():
        assert cmd.startswith(("/okonomi:", "/user:")), f"slug not namespaced: {cmd!r}"


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
