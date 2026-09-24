"""Env scrub + sandbox wrapper for model-run subprocesses (bash)."""

import sys

import pytest

from okonomi.tools import _sandbox
from okonomi.utils import scrub_env


def test_scrub_env_drops_secret_like_vars():
    env = {"FAKE_API_KEY": "x" * 20, "MY_TOKEN": "y" * 20, "PATH": "/usr/bin", "HOME": "/tmp/h"}
    out = scrub_env(env)
    assert "FAKE_API_KEY" not in out and "MY_TOKEN" not in out
    assert out["PATH"] == "/usr/bin" and out["HOME"] == "/tmp/h"


def test_scrub_env_allowlist():
    env = {"FAL_KEY": "z" * 20, "HF_TOKEN": "w" * 20, "OKONOMI_ENV_ALLOW": "FAL_KEY"}
    out = scrub_env(env)
    assert out["FAL_KEY"] == "z" * 20, "allowlisted key must survive"
    assert "HF_TOKEN" not in out


def test_wrap_command_passthrough_when_disabled(monkeypatch):
    monkeypatch.setenv("OKONOMI_NO_SANDBOX", "1")
    argv = ["bash", "-lc", "echo hi"]
    assert _sandbox.wrap_command(argv) is argv


def test_wrap_command_passthrough_off_darwin(monkeypatch):
    monkeypatch.delenv("OKONOMI_NO_SANDBOX", raising=False)
    monkeypatch.setattr(_sandbox.sys, "platform", "linux")
    argv = ["bash", "-lc", "echo hi"]
    assert _sandbox.wrap_command(argv) is argv


def test_profile_scopes_writes():
    profile = _sandbox._build_profile("/some/cwd")
    assert "(deny file-write*)" in profile
    assert str(_sandbox.OKONOMI_HOME) in profile
    assert '"/some/cwd"' in profile


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS-only")
def test_sandbox_blocks_out_of_scope_write(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.delenv("OKONOMI_NO_SANDBOX", raising=False)
    _sandbox._sandbox_ok = None  # force a fresh probe
    target = tmp_path / "blocked.txt"  # tmp_path is under /private/var/folders → allowed
    argv = _sandbox.wrap_command(["/bin/bash", "-c", f"echo hi > {target}"], cwd=str(tmp_path))
    if argv[0] != "sandbox-exec":
        pytest.skip("sandbox-exec unavailable on this machine")
    # Allowed: within tmp (var/folders subpath)
    assert subprocess.run(argv, capture_output=True).returncode == 0

    # Blocked: a path outside every allowed subpath
    argv2 = _sandbox.wrap_command(["/bin/bash", "-c", "echo hi > /Users/blocked-probe.txt"], cwd=str(tmp_path))
    result = subprocess.run(argv2, capture_output=True)
    assert result.returncode != 0, "write outside scoped roots must fail"
