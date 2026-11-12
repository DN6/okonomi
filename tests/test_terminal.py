"""The bash tool: env scrub and sandbox, dangerous-command approvals,
and interrupts."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest
from _helpers import fake_agent
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from okonomi.agent.agent import OkonomiAgent
from okonomi.events import ApprovalRequest
from okonomi.tools import _command_guard as guard
from okonomi.tools import _sandbox
from okonomi.tools.terminal import TerminalTools
from okonomi.utils import scrub_env


def test_bash_env_is_scrubbed_and_carries_env_file_and_session_dirs(tmp_path, monkeypatch):
    env = {"FAKE_API_KEY": "x" * 20, "MY_TOKEN": "y" * 20, "PATH": "/usr/bin", "HOME": "/tmp/h"}
    out = scrub_env(env)
    assert "FAKE_API_KEY" not in out and "MY_TOKEN" not in out
    assert out["PATH"] == "/usr/bin" and out["HOME"] == "/tmp/h"

    env_file = tmp_path / "creds.env"
    env_file.write_text("FAL_KEY=fal-secret-value-123\n")
    monkeypatch.setenv("OKONOMI_ENV_FILE", str(env_file))
    terminal = TerminalTools()
    terminal._session_dir_fn = lambda: "/tmp/okonomi-test/artifacts"
    terminal._scratch_dir_fn = lambda: "/tmp/okonomi-test/scratch"
    result = terminal.bash(
        '[ "$FAL_KEY" = fal-secret-value-123 ] && echo injected; echo "$FAL_KEY"; '
        'echo "$OKONOMI_SESSION_DIR|$OKONOMI_SCRATCH_DIR"'
    )
    assert "injected" in result["output"], f"FAL_KEY not injected: {result!r}"
    assert "fal-secret-value-123" not in result["output"], f"env-file value leaked into output: {result!r}"
    assert "[FAL_KEY redacted]" in result["output"]
    assert result["output"].strip().endswith("/tmp/okonomi-test/artifacts|/tmp/okonomi-test/scratch"), result

    monkeypatch.setenv("OKONOMI_ENV_FILE", str(tmp_path / "missing.env"))
    result = TerminalTools().bash("echo should-not-run")
    assert result["exit_code"] != 0, f"command ran without its env file: {result!r}"
    assert "OKONOMI_ENV_FILE" in result["output"]


@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="okonomi sandboxes only macOS and Linux")
def test_sandbox_blocks_out_of_scope_write(tmp_path):
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


def test_unavailable_sandbox_refuses_with_a_reason(monkeypatch):
    monkeypatch.setattr(_sandbox, "_probed", False)
    monkeypatch.setitem(_sandbox._RUNNERS, sys.platform, (lambda cwd: ["/nonexistent-okonomi-runner"], "install-hint"))
    with pytest.raises(_sandbox.SandboxUnavailable, match="install-hint"):
        _sandbox.wrap_command(["true"])
    assert _sandbox._probed is False, "a failed probe must be retried, not cached as success"

    monkeypatch.setattr(_sandbox.sys, "platform", "win32")
    with pytest.raises(_sandbox.SandboxUnavailable, match="'win32'"):
        _sandbox.wrap_command(["true"])


DANGEROUS = "git reset --hard"


@pytest.fixture(autouse=True)
def approvals_file(tmp_path, monkeypatch):
    path = tmp_path / "approvals.json"
    monkeypatch.setattr(guard, "OKONOMI_APPROVALS_FILE", path)
    monkeypatch.delenv("OKONOMI_APPROVALS", raising=False)
    return path


def test_guard_rules(approvals_file, monkeypatch):
    allowed, reason = guard.check_command(DANGEROUS)
    assert not allowed and "needs approval" in reason, f"unapproved dangerous command ran: {reason!r}"
    assert guard.check_command(DANGEROUS, approved=True) == (True, None)

    guard.approve_always(guard.dangerous_matches(DANGEROUS))
    assert guard.check_command(DANGEROUS) == (True, None), f"allowlist not honoured: {approvals_file.read_text()}"

    monkeypatch.setenv("OKONOMI_APPROVALS", "off")
    approvals_file.write_text(json.dumps({"deny": [r"\bgit\s+push\b"]}))
    assert guard.check_command(DANGEROUS) == (True, None), "OKONOMI_APPROVALS=off must disable approval prompts"
    allowed, reason = guard.check_command("git push origin main", approved=True)
    assert not allowed and "deny rule" in reason, f"deny rule bypassed: {reason!r}"


def test_bad_config_raises(approvals_file, monkeypatch):
    approvals_file.write_text(json.dumps({"deny": ["("]}))
    with pytest.raises(ValueError, match=r"Invalid deny regex '\('"):
        guard.load_approval_rules()

    approvals_file.write_text("{}")
    monkeypatch.setenv("OKONOMI_APPROVALS", "smart")
    with pytest.raises(EnvironmentError, match="OKONOMI_APPROVALS"):
        guard.check_command(DANGEROUS)


class _Terminal:
    def __init__(self):
        self.calls: list[tuple[str, bool]] = []

    def bash(self, command: str, _chunks_sink=None, _kill=None, _approved: bool = False) -> dict:
        """Record the call; refuse like the real bash when not approved."""
        allowed, reason = guard.check_command(command, approved=_approved)
        if not allowed:
            return {"output": reason, "exit_code": 126}
        self.calls.append((command, _approved))
        return {"output": "ran", "exit_code": 0}


def _run_bash(fake, terminal: _Terminal, answer=None) -> tuple[dict | str, list[ApprovalRequest]]:
    requests: list[ApprovalRequest] = []

    def _on_event(ev):
        if isinstance(ev, ApprovalRequest):
            requests.append(ev)
            ev.respond(answer)

    fake.emitter.subscribe(_on_event)

    def _model(messages, info: AgentInfo):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="bash", args={"command": DANGEROUS}, tool_call_id="c1")])
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(_model))
    OkonomiAgent._register_tool(fake, agent, terminal)
    for m in agent.run_sync("go").all_messages():
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, ToolReturnPart):
                    return p.content, requests
    raise AssertionError("no tool return")


def test_approval_answers_once_session_and_always(approvals_file):
    fake, terminal = fake_agent(can_prompt=True), _Terminal()
    result, requests = _run_bash(fake, terminal, answer=True)
    assert result["output"] == "ran" and len(requests) == 1, f"once-approval did not run: {result!r}"
    assert requests[0].reason == "git reset --hard (destroys uncommitted changes)"
    _, requests = _run_bash(fake, terminal, answer=True)
    assert len(requests) == 1, "a once-approval must not carry over to the next call"

    fake, terminal = fake_agent(can_prompt=True), _Terminal()
    _run_bash(fake, terminal, answer="__session__")
    result, requests = _run_bash(fake, terminal, answer=False)
    assert result["output"] == "ran" and not requests, f"session approval not reused: {result!r} {requests!r}"

    fake, terminal = fake_agent(can_prompt=True), _Terminal()
    _run_bash(fake, terminal, answer="__always__")
    allow = json.loads(approvals_file.read_text())["allow"]
    assert allow == guard.dangerous_matches(DANGEROUS), f"always-approval not persisted: {allow!r}"


def test_denial_feedback_and_headless_do_not_run():
    for answer, expected in ((False, "denied"), ("use git stash instead", "use git stash instead")):
        fake, terminal = fake_agent(can_prompt=True), _Terminal()
        result, _ = _run_bash(fake, terminal, answer=answer)
        assert expected in result["error"] and not terminal.calls, f"command ran after {answer!r}: {result!r}"

    fake, terminal = fake_agent(can_prompt=False), _Terminal()
    result, requests = _run_bash(fake, terminal)
    assert not requests and "needs approval" in result["output"], f"headless run must fail closed: {result!r}"
    assert not terminal.calls


def test_interrupt_kills_the_foreground_command_and_leaves_background_tasks_running():
    terminal = TerminalTools()
    finished = threading.Event()
    results: dict = {}

    def _on_complete(**kwargs):
        results.update(kwargs)
        finished.set()

    terminal._on_bg_complete = _on_complete
    terminal.bash("sleep 0.6; echo done", background=True)

    threading.Timer(0.3, terminal._kill_foreground).start()
    started = time.monotonic()
    result = terminal.bash("sleep 30", timeout=10)
    elapsed = time.monotonic() - started
    assert result["exit_code"] == 130, f"expected exit_code 130 after interrupt, got {result!r}"
    assert result["output"].endswith("[Interrupted]"), result["output"]
    assert elapsed < 5, f"interrupt took {elapsed:.1f}s; subprocess was not killed"

    assert finished.wait(timeout=5), "background task did not finish"
    assert results["error"] is None, f"background task was killed by interrupt: {results!r}"
    assert results["output"].strip() == "done", results["output"]
