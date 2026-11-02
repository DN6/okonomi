"""Dangerous-command approvals: hard floors and deny rules always block,
dangerous patterns prompt the user (once / session / always / deny /
feedback), and with no one to answer the command is refused."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from okonomi.agent.agent import OkonomiAgent
from okonomi.agent.jobs import JobRegistry
from okonomi.events import ApprovalRequest, EventEmitter
from okonomi.tools import _command_guard as guard


DANGEROUS = "git reset --hard"


@pytest.fixture(autouse=True)
def approvals_file(tmp_path, monkeypatch):
    path = tmp_path / "approvals.json"
    monkeypatch.setattr(guard, "OKONOMI_APPROVALS_FILE", path)
    monkeypatch.delenv("OKONOMI_APPROVALS", raising=False)
    return path


def test_dangerous_command_needs_approval():
    allowed, reason = guard.check_command(DANGEROUS)
    assert not allowed and "needs approval" in reason, f"unapproved dangerous command ran: {reason!r}"
    assert guard.check_command(DANGEROUS, approved=True) == (True, None)


def test_approvals_off_allows_dangerous(monkeypatch):
    monkeypatch.setenv("OKONOMI_APPROVALS", "off")
    assert guard.check_command(DANGEROUS) == (True, None), "OKONOMI_APPROVALS=off must disable approval prompts"


def test_always_allowlist_allows(approvals_file):
    guard.approve_always(guard.dangerous_matches(DANGEROUS))
    assert guard.check_command(DANGEROUS) == (True, None), f"allowlist not honoured: {approvals_file.read_text()}"


def test_deny_rule_blocks_even_when_approved_or_off(approvals_file, monkeypatch):
    approvals_file.write_text(json.dumps({"deny": [r"\bgit\s+push\b"]}))
    monkeypatch.setenv("OKONOMI_APPROVALS", "off")
    allowed, reason = guard.check_command("git push origin main", approved=True)
    assert not allowed and "deny rule" in reason, f"deny rule bypassed: {reason!r}"


def test_invalid_deny_regex_raises(approvals_file):
    approvals_file.write_text(json.dumps({"deny": ["("]}))
    with pytest.raises(ValueError, match=r"Invalid deny regex '\('"):
        guard.load_approval_rules()


def test_unknown_approval_mode_raises(monkeypatch):
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


def _fake_agent_self(can_prompt: bool) -> SimpleNamespace:
    emitter = EventEmitter()
    fake = SimpleNamespace(
        _tool_pool=ThreadPoolExecutor(),
        emitter=emitter,
        jobs=JobRegistry(emitter),
        _sync_session_assets=lambda deps: 0,
        _get_dependencies=lambda: None,
        _turn_cancelled=threading.Event(),
        _detach_now=threading.Event(),
        can_prompt=can_prompt,
        _session_approvals=set(),
    )
    fake._unapproved_patterns = lambda c: OkonomiAgent._unapproved_patterns(fake, c)
    fake._ask_approval = lambda c, p: OkonomiAgent._ask_approval(fake, c, p)
    return fake


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


def test_approve_once_runs_and_asks_again():
    fake, terminal = _fake_agent_self(can_prompt=True), _Terminal()
    result, requests = _run_bash(fake, terminal, answer=True)
    assert result["output"] == "ran" and len(requests) == 1, f"once-approval did not run: {result!r}"
    assert requests[0].reason == "git reset --hard (destroys uncommitted changes)"
    _, requests = _run_bash(fake, terminal, answer=True)
    assert len(requests) == 1, "a once-approval must not carry over to the next call"


def test_approve_session_stops_asking():
    fake, terminal = _fake_agent_self(can_prompt=True), _Terminal()
    _run_bash(fake, terminal, answer="__session__")
    result, requests = _run_bash(fake, terminal, answer=False)
    assert result["output"] == "ran" and not requests, f"session approval not reused: {result!r} {requests!r}"


def test_approve_always_persists(approvals_file):
    fake, terminal = _fake_agent_self(can_prompt=True), _Terminal()
    _run_bash(fake, terminal, answer="__always__")
    allow = json.loads(approvals_file.read_text())["allow"]
    assert allow == guard.dangerous_matches(DANGEROUS), f"always-approval not persisted: {allow!r}"


@pytest.mark.parametrize("answer, expected", [(False, "denied"), ("use git stash instead", "use git stash instead")])
def test_deny_and_feedback_do_not_run(answer, expected):
    fake, terminal = _fake_agent_self(can_prompt=True), _Terminal()
    result, _ = _run_bash(fake, terminal, answer=answer)
    assert expected in result["error"] and not terminal.calls, f"command ran after {answer!r}: {result!r}"


def test_no_ui_refuses_without_prompting():
    fake, terminal = _fake_agent_self(can_prompt=False), _Terminal()
    result, requests = _run_bash(fake, terminal)
    assert not requests and "needs approval" in result["output"], f"headless run must fail closed: {result!r}"
    assert not terminal.calls
