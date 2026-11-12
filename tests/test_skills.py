"""Skills: installing from GitHub and section-structure checks on write."""

import json

import httpx
import pytest

from okonomi.tools import skills
from okonomi.tools.skills import SkillTools, _check_skill_structure, _parse_github_spec, install_skill_from_github


SKILL_MD = "---\nname: demo\ndescription: d\n---\nintro\n\n## Steps\ndo it\n"

# path → (type, content). Mirrors a repo with one skill at skills/demo.
REPO = {
    "skills/demo": [
        ("SKILL.md", "file"),
        ("README.md", "file"),
        ("references", "dir"),
        ("node_modules", "dir"),
    ],
    "skills/demo/references": [("api.md", "file")],
    "skills/demo/node_modules": [("junk.js", "file")],
}
FILES = {
    "skills/demo/SKILL.md": SKILL_MD,
    "skills/demo/README.md": "readme",
    "skills/demo/references/api.md": "api ref",
    "skills/demo/node_modules/junk.js": "junk",
}


def _handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    api = "https://api.github.com/repos/o/r/contents/"
    raw = "https://raw.test/"
    if url.startswith(api):
        path = url[len(api) :].split("?")[0]
        if path in FILES:
            return httpx.Response(200, json={"type": "file", "path": path})
        if path not in REPO:
            return httpx.Response(404)
        entries = [
            {"type": kind, "name": name, "path": f"{path}/{name}", "download_url": f"{raw}{path}/{name}"}
            for name, kind in REPO[path]
        ]
        return httpx.Response(200, content=json.dumps(entries))
    if url.startswith(raw):
        return httpx.Response(200, text=FILES[url[len(raw) :]])
    return httpx.Response(500)


@pytest.fixture
def github(store, monkeypatch):
    real_client = httpx.Client

    def _mock_client(**kwargs):
        kwargs["transport"] = httpx.MockTransport(_handler)
        return real_client(**kwargs)

    monkeypatch.setattr(skills.httpx, "Client", _mock_client)


def test_install_parses_the_spec_and_copies_allowed_subdirs_only(github):
    assert _parse_github_spec("o/r/skills/demo") == ("o", "r", None, "skills/demo")
    assert _parse_github_spec("https://github.com/o/r/tree/main/skills/demo/") == ("o", "r", "main", "skills/demo")
    assert _parse_github_spec("o/r") == ("o", "r", None, "")
    assert _parse_github_spec("o") is None, "owner alone is not a repo"
    assert _parse_github_spec("https://example.com/x/y") is None, "non-GitHub URLs are rejected"

    result = install_skill_from_github("o/r/skills/demo")
    assert result["success"], result
    assert result["name"] == "demo"
    dest = skills.USER_SKILLS_DIR / "demo"
    assert (dest / "SKILL.md").read_text() == SKILL_MD
    assert (dest / "README.md").read_text() == "readme"
    assert (dest / "references" / "api.md").read_text() == "api ref"
    assert not (dest / "node_modules").exists(), "dirs outside ALLOWED_SUBDIRS must be skipped"
    assert not list(skills.USER_SKILLS_DIR.glob(".tmp.*")), "staging dir must be renamed away"


def _mk(body: str) -> str:
    return f"---\nname: demo\ndescription: d\n---\n{body}"


def test_skill_structure_is_checked_on_write(store):
    assert _check_skill_structure(_mk("Short preamble.\n\n## Setup\ndo x\n\n## Usage\ndo y\n")) == (None, [])

    err, warnings = _check_skill_structure(_mk("x" * (skills.MAX_INTRO_CHARS + 1)))
    assert err is None and len(warnings) == 1 and "no '## '" in warnings[0], (err, warnings)

    result = SkillTools().create_skill("dupe", _mk("## Overview\na\n\n## Overview\nb\n"))
    assert result["success"] is False and "Duplicate" in result["error"], result
    assert not (skills.USER_SKILLS_DIR / "dupe").exists()
