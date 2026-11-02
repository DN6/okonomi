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


@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="okonomi sandboxes only macOS and Linux")
def test_sandbox_blocks_out_of_scope_write(tmp_path):
    import subprocess
    import uuid
    from pathlib import Path

    _sandbox._probed = False
    inside = tmp_path / "allowed.txt"
    argv = _sandbox.wrap_command(["/bin/bash", "-c", f"echo hi > {inside}"], cwd=str(tmp_path))
    assert subprocess.run(argv, capture_output=True).returncode == 0, "write inside cwd must succeed"

    # $HOME is writable outside the sandbox but is not a scoped root.
    outside = Path.home() / f"okonomi-sandbox-probe-{uuid.uuid4().hex}.txt"
    try:
        argv = _sandbox.wrap_command(["/bin/bash", "-c", f"echo hi > {outside}"], cwd=str(tmp_path))
        result = subprocess.run(argv, capture_output=True)
        assert result.returncode != 0 and not outside.exists(), "write outside scoped roots must fail"
    finally:
        outside.unlink(missing_ok=True)


def test_unsupported_platform_refuses(monkeypatch):
    monkeypatch.setattr(_sandbox.sys, "platform", "win32")
    with pytest.raises(_sandbox.SandboxUnavailable, match="'win32'"):
        _sandbox.wrap_command(["true"])


def test_failed_probe_refuses_with_install_hint(monkeypatch):
    monkeypatch.setattr(_sandbox, "_probed", False)
    monkeypatch.setitem(_sandbox._RUNNERS, sys.platform, (lambda cwd: ["/nonexistent-okonomi-runner"], "install-hint"))
    with pytest.raises(_sandbox.SandboxUnavailable, match="install-hint"):
        _sandbox.wrap_command(["true"])
    assert _sandbox._probed is False, "a failed probe must be retried, not cached as success"


def test_env_file_injects_scrubbed_credential_into_bash(tmp_path, monkeypatch):
    from okonomi.tools.terminal import TerminalTools

    env_file = tmp_path / "creds.env"
    env_file.write_text("FAL_KEY=fal-secret-value-123\n")
    monkeypatch.setenv("OKONOMI_ENV_FILE", str(env_file))

    result = TerminalTools().bash('[ "$FAL_KEY" = fal-secret-value-123 ] && echo injected; echo "$FAL_KEY"')
    assert "injected" in result["output"], f"FAL_KEY not injected: {result!r}"
    assert "fal-secret-value-123" not in result["output"], f"env-file value leaked into output: {result!r}"
    assert "[FAL_KEY redacted]" in result["output"]


def test_env_file_missing_fails_command(tmp_path, monkeypatch):
    from okonomi.tools.terminal import TerminalTools

    monkeypatch.setenv("OKONOMI_ENV_FILE", str(tmp_path / "missing.env"))

    result = TerminalTools().bash("echo should-not-run")
    assert result["exit_code"] != 0, f"command ran without its env file: {result!r}"
    assert "OKONOMI_ENV_FILE" in result["output"]
