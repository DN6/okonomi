"""The execute_code tool: Python with tool namespaces injected, foreground
and background (bg-exec-N) dispatch, and approval gating. ``CodeExecutor``
is composed into ``OkonomiAgent``."""

import asyncio
import io
import re
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import TYPE_CHECKING

from pydantic_ai import Agent, RunContext

from ..events import SystemNote


if TYPE_CHECKING:
    from .agent import OkonomiAgent


def _extract_bg_label(code: str) -> str:
    """A short human label for a background code block — its first
    ``namespace.method(...)`` call, else "code block"."""
    match = re.search(r"\b(\w+)\.(\w+)\s*\(", code)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    return "code block"


class CodeExecutor:
    """Owns execute_code: a persistent-state Python REPL plus background
    (bg-exec-N) dispatch.

    The model's code runs in ``execution_globals``, which persists across
    turns; ``get_session_dir`` and the asset sweep are the output path.
    Conversational surfaces (emitter, pending tool-call widgets, job
    registry, asset sync) belong to the owning agent and are reached
    through it.
    """

    def __init__(self, agent: "OkonomiAgent") -> None:
        self.agent = agent
        # Live background execute_code jobs keyed by bg_id. Plain threads
        # (not asyncio tasks) so they survive the per-turn event loop
        # teardown in the CLI. Public: the TUI lists in-flight jobs from it.
        self.threads: dict[str, threading.Thread] = {}
        self._counter = 0

    def register_tool(self, agent: Agent) -> None:
        """Register the execute_code tool on the pydantic-ai agent."""
        # Imported here, not at module top: agent.py defines AgentDependencies
        # and imports this module, so a top-level import would be circular.
        # By the time this method runs (agent construction) both are loaded.
        from .agent import AgentDependencies

        @agent.tool
        async def execute_code(
            ctx: RunContext[AgentDependencies],
            code: str,
            background: bool = False,
        ) -> str:
            """Execute Python code in a persistent REPL environment.

            Variables persist across calls — ``execution_globals`` is kept
            between `execute_code` invocations, so state you build up (loaded
            data, intermediate results) is available in later blocks. Use
            `print()` to see output. Import libraries directly (PIL, imageio,
            numpy, subprocess, urllib, ...).

            To interpret an image, include its URL / data URI / local path in
            your reply text — okonomi attaches image references as multimodal
            content automatically.

            Args:
                code: Python code to execute. Use print() for output.
                background: When True, execution is dispatched as a background
                    task and the tool returns immediately with a `bg-exec-N`
                    handle; a completion notice arrives as a new turn. Use for
                    long-running code you don't need to chain from. Defaults
                    to False.

            Returns:
                The stdout output, or an error message. When background=True,
                returns a handle like "Dispatched to background as bg-exec-N".
            """
            # Mutate execution_globals in place — `ctx.deps.state.execution_globals`
            # is the same object as `self.state.execution_globals` on the agent, so
            # rebinding (`= {...}`) would silently break cross-turn variable
            # persistence (the agent would keep the empty original).
            ctx.deps.state.execution_globals.setdefault("__builtins__", __builtins__)
            ctx.deps.state.execution_globals.setdefault("asyncio", asyncio)

            def get_session_dir() -> str:
                """Get the current session's directory path.

                Use this to save files (like stitched videos) to the session folder.

                Returns:
                    Absolute path to the current session directory.

                Example:
                    session_dir = get_session_dir()
                    html_path = f"{session_dir}/sketch.html"
                    with open(html_path, "w") as f:
                        f.write(html)
                """
                session_dir = ctx.deps.session.session_dir
                session_dir.mkdir(parents=True, exist_ok=True)
                return str(session_dir)

            ctx.deps.state.execution_globals["get_session_dir"] = get_session_dir

            self.agent.status = "Running..."

            def _exec_in_thread():
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                try:
                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        exec(code, ctx.deps.state.execution_globals)

                    stdout_output = stdout_capture.getvalue()
                    stderr_output = stderr_capture.getvalue()

                    result = ""
                    if stdout_output:
                        result += stdout_output
                    if stderr_output:
                        result += f"\nStderr:\n{stderr_output}"
                    return result.strip() if result.strip() else None, None
                except Exception:
                    return None, traceback.format_exc()
                finally:
                    self.agent._sync_session_assets(ctx.deps)

            if background:
                self._counter += 1
                bg_id = f"bg-exec-{self._counter}"
                before_artifacts = len(ctx.deps.state.artifacts)

                # Long-lived buffers so /logs can snapshot during and after
                # the run. _run_bg redirects stdout/stderr into these, then
                # reads from them once exec returns.
                bg_stdout = io.StringIO()
                bg_stderr = io.StringIO()

                def _reader(out=bg_stdout, err=bg_stderr) -> str:
                    parts = []
                    s = out.getvalue()
                    if s:
                        parts.append(s)
                    e = err.getvalue()
                    if e:
                        parts.append(f"\n--- stderr ---\n{e}")
                    return "".join(parts) if parts else "(no output yet)"

                self.agent._register_bg_log(
                    task_id=bg_id, label=_extract_bg_label(code), reader=_reader, kind="bg-exec"
                )

                # Hand the ToolCall event from _display_tool_call to the bg
                # thread so it can resolve (ok/error + output) in-place when
                # the code finishes. This gives the user a single foreground
                # widget that represents the bg job throughout its lifetime.
                bound_tool_call = self.agent._pending_tool_calls.pop(0) if self.agent._pending_tool_calls else None
                if bound_tool_call is not None:
                    try:
                        bound_tool_call.args = f"{bound_tool_call.args} → {bg_id} running"
                        self.agent.emitter.emit(bound_tool_call)
                    except Exception:
                        pass

                def _run_bg():
                    error: str | None = None
                    output: str | None = None
                    new_artifacts = 0
                    try:
                        try:
                            with redirect_stdout(bg_stdout), redirect_stderr(bg_stderr):
                                exec(code, ctx.deps.state.execution_globals)
                            output = _reader().strip()
                            if output == "(no output yet)":
                                output = None
                        except Exception:
                            error = traceback.format_exc()
                            output = _reader().strip() or None
                        self.agent._sync_session_assets(ctx.deps)
                        new_artifacts = len(ctx.deps.state.artifacts) - before_artifacts
                        log_entry = self.agent._bg_logs.get(bg_id)
                        if log_entry is not None:
                            log_entry["status"] = "error" if error else "ok"
                            log_entry["ended_at"] = time.time()
                        if error:
                            self.agent.emitter.emit(SystemNote(f"✗ {bg_id} failed:\n{error}", level="error"))
                        else:
                            tail = (
                                f" — {new_artifacts} artifact{'s' if new_artifacts != 1 else ''}"
                                if new_artifacts
                                else ""
                            )
                            self.agent.emitter.emit(SystemNote(f"✓ {bg_id} done{tail}"))
                        # Update the original ToolCall event so the user
                        # sees the bg job's result in-place in history.
                        if bound_tool_call is not None:
                            try:
                                summary_str = (
                                    f"{new_artifacts} artifact{'s' if new_artifacts != 1 else ''}"
                                    if new_artifacts
                                    else "done"
                                )
                                merged_output = f"{bg_id} {summary_str}" + (f"\n\n{output}" if output else "")
                                self.agent.emitter.emit(bound_tool_call.finish(output=merged_output, error=error))
                            except Exception:
                                pass
                    finally:
                        self.threads.pop(bg_id, None)
                        produced_artifacts = list(ctx.deps.state.artifacts[before_artifacts:])
                        try:
                            self.agent._job_finished(
                                bg_id=bg_id,
                                error=error,
                                output=output,
                                new_artifacts=new_artifacts,
                                produced_artifacts=produced_artifacts,
                            )
                        except Exception as cb_err:
                            self.agent.emitter.emit(SystemNote(f"JobFinished emit failed: {cb_err}", level="error"))

                thread = threading.Thread(target=_run_bg, daemon=True, name=f"okonomi-{bg_id}")
                self.threads[bg_id] = thread
                thread.start()

                # Brief wait so fast failures (KeyError, fal arg validation,
                # 4xx, malformed loras) surface as the *actual* tool result
                # instead of letting the model narrate around an already-
                # broken job. 2s catches almost every fast failure without
                # noticeably stalling successful long-running jobs (which
                # take 10–90s on fal endpoints).
                thread.join(timeout=2.0)
                self.agent.status = ""

                log_entry = self.agent._bg_logs.get(bg_id, {})
                if not thread.is_alive() and log_entry.get("status") == "error":
                    reader = log_entry.get("_reader")
                    snapshot = ""
                    if callable(reader):
                        try:
                            out = reader()
                            if isinstance(out, str):
                                snapshot = out.strip()
                        except Exception:
                            snapshot = ""
                    return (
                        f"Background task {bg_id} FAILED during dispatch.\n\n"
                        f"{snapshot[-2000:] if snapshot else '(no output captured)'}\n\n"
                        "The job did not produce media. Do not narrate expected output. "
                        "Surface the error to the user and ask before retrying — "
                        "re-running an expensive call to recover from a parse-side error "
                        "(KeyError / TypeError on `result[...]`) is the worst kind of mistake; "
                        "inspect first."
                    )

                return (
                    f"Dispatched to background as {bg_id}. Execution is in flight.\n\n"
                    "Acknowledge dispatch in ONE short sentence and stop — do NOT narrate "
                    "what the result will look like, the aesthetic it will produce, or any "
                    "other forward-looking claim. You'll be re-invoked with a completion "
                    "notice when it finishes; describe the actual output then."
                )

            try:
                output, error = await asyncio.to_thread(_exec_in_thread)

                # Attach output/error back to the UI's ToolCall event so the
                # user sees failures without having to ask the model what
                # happened. The widget auto-expands on error status.
                self.agent._finish_pending_tool_call(output=output or "", error=error)

                if error:
                    return f"Error executing code:\n{error}"

                if output:
                    from ..context import _strip_large_content

                    cleaned = _strip_large_content(output)
                    if len(cleaned) > 10_000:
                        cleaned = cleaned[:8_000] + "\n...[truncated]...\n" + cleaned[-1_500:]
                    return cleaned

                return "Code executed successfully (no output)"

            except asyncio.CancelledError:
                self.agent.status = ""
                raise
            finally:
                self.agent.status = ""
