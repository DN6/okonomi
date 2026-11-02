"""Ctrl+C must kill the foreground bash subprocess, not background jobs."""

import threading
import time

import pytest

from okonomi.tools.terminal import TerminalTools


@pytest.fixture
def terminal():
    return TerminalTools()


def test_interrupt_kills_foreground_command(terminal):
    threading.Timer(0.3, terminal._kill_foreground).start()
    started = time.monotonic()
    result = terminal.bash("sleep 30", timeout=10)
    elapsed = time.monotonic() - started

    assert result["exit_code"] == 130, f"expected exit_code 130 after interrupt, got {result!r}"
    assert result["output"].endswith("[Interrupted]"), result["output"]
    assert elapsed < 5, f"interrupt took {elapsed:.1f}s; subprocess was not killed"


def test_interrupt_leaves_background_job_running(terminal):
    finished = threading.Event()
    results: dict = {}

    def _on_complete(**kwargs):
        results.update(kwargs)
        finished.set()

    terminal._on_bg_complete = _on_complete
    terminal.bash("sleep 0.6; echo done", background=True)
    time.sleep(0.2)
    terminal._kill_foreground()

    assert finished.wait(timeout=5), "background job did not finish"
    assert results["error"] is None, f"background job was killed by interrupt: {results!r}"
    assert results["output"].strip() == "done", results["output"]
