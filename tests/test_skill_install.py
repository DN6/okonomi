"""`/skills install` — fetching a skill directory from GitHub into the user
skills dir. The GitHub contents API is mocked via httpx's MockTransport.
"""

import json

import httpx
import pytest

from okonomi.tools import skills
from okonomi.tools.skills import _parse_github_spec, install_skill_from_github


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


def test_parse_shorthand_and_tree_url():
    assert _parse_github_spec("o/r/skills/demo") == ("o", "r", None, "skills/demo")
    assert _parse_github_spec("https://github.com/o/r/tree/main/skills/demo/") == ("o", "r", "main", "skills/demo")
    assert _parse_github_spec("o/r") == ("o", "r", None, "")
    assert _parse_github_spec("o") is None, "owner alone is not a repo"
    assert _parse_github_spec("https://example.com/x/y") is None, "non-GitHub URLs are rejected"


def test_install_copies_skill_and_allowed_subdirs_only(github):
    result = install_skill_from_github("o/r/skills/demo")
    assert result["success"], result
    assert result["name"] == "demo"
    dest = skills.USER_SKILLS_DIR / "demo"
    assert (dest / "SKILL.md").read_text() == SKILL_MD
    assert (dest / "README.md").read_text() == "readme"
    assert (dest / "references" / "api.md").read_text() == "api ref"
    assert not (dest / "node_modules").exists(), "dirs outside ALLOWED_SUBDIRS must be skipped"
    assert not list(skills.USER_SKILLS_DIR.glob(".tmp.*")), "staging dir must be renamed away"
