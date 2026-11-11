"""The compact transcript prints one line per finished tool call, linked to
a log file holding the full call and output; ``/logs <call id>`` reads
the same file."""

import io

from rich.console import Console

from okonomi.events import EventEmitter, SystemNote, ToolCallDisplay
from okonomi.tui.commands import handle
from okonomi.tui.transcript import ReplTranscript


def _finished_call() -> ToolCallDisplay:
    call = ToolCallDisplay(tool_name="bash", args="ls artifacts", details="ls artifacts", tool_call_id="call-7")
    return call.finish(output="fox_v1_1.png\nfox_v1_2.png\n")


def test_compact_line_links_to_a_log_with_the_full_call(tmp_path):
    transcript = ReplTranscript(log_dir_fn=lambda: tmp_path / "logs")
    buf = io.StringIO()
    transcript.console = Console(file=buf, force_terminal=True, width=200)

    transcript.on_event(_finished_call())

    log = tmp_path / "logs" / "call-7.md"
    assert "```\nls artifacts\n```" in log.read_text() and "fox_v1_2.png" in log.read_text(), log.read_text()
    printed = buf.getvalue()
    assert "⏺ Ran shell command  ls artifacts" in printed, printed
    assert f";file://{log}\x1b\\" in printed, "the summary line must be an OSC 8 link to the log"
    assert "fox_v1_1.png" not in printed, "compact mode keeps the output out of the scrollback"


def test_without_a_log_dir_output_prints_inline(tmp_path):
    transcript = ReplTranscript()
    buf = io.StringIO()
    transcript.console = Console(file=buf, force_terminal=True, width=200)

    transcript.on_event(_finished_call())

    assert "⎿ fox_v1_1.png" in buf.getvalue(), buf.getvalue()


def test_logs_command_reads_a_tool_call_log(okonomi_agent):
    log_dir = okonomi_agent.session.logs_dir
    log_dir.mkdir(parents=True)
    (log_dir / "call-7.md").write_text("# Ran shell command\n\nfull output here\n")
    notes: list[str] = []
    emitter = EventEmitter()

    def _on_event(ev):
        if isinstance(ev, SystemNote):
            notes.append(ev.text)

    emitter.subscribe(_on_event)

    handle("/logs call-7", okonomi_agent, emitter)

    assert notes == ["# Ran shell command\n\nfull output here\n"], notes
