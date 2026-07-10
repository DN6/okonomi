"""The execute_code tool: sandboxed Python with tool namespaces injected,
foreground and background (bg-exec-N) dispatch, and approval gating.
``CodeExecutionMixin`` is mixed into ``OkonomiAgent``."""

import asyncio
import io
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout

from pydantic_ai import Agent, RunContext

from ..events import SystemNote
from .wrappers import ToolWrapper, _extract_bg_label


class CodeExecutionMixin:
    """Registers and runs the execute_code tool. Expects the attributes
    set up in ``OkonomiAgent.__init__`` (emitter, approval flow, bg-exec
    registries, expensive patterns)."""

    def _register_code_execution_tool(self, agent: Agent) -> None:
        """Register the code execution tool with the agent."""
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
            """Execute Python code with access to all tool objects.

            The code runs in an environment with these tool objects available:
            - browser: Headless browser rendering (screenshot and video capture of HTML/JS)
            - genmedia: Run any fal.ai endpoint — image, video, audio, 3D — auto-tracks outputs

            For local image / frame work (PIL ops, imageio), just import
            the libraries directly from inside `execute_code`. For image
            interpretation (one or many), include the URLs / data URIs / local
            paths in your conversation reply or in your next user-facing
            message — okonomi extracts image references from text and
            attaches them to the model as multimodal content automatically.

            For past-session info, use the `search` tool (not a programmatic
            namespace) — e.g. `search(pattern="<keyword>", path="~/.okonomi/sessions/")`
            — instead of any sessions.* methods, which no longer exist.

            Args:
                code: Python code to execute. Use print() for output.
                background: When True, approval (if required) still happens in the foreground,
                    but execution is dispatched as a background task and the tool returns
                    immediately. A completion notification is injected as a new turn when the
                    task finishes. Use for long media generations when you don't need the
                    result to chain into the next code block. Defaults to False.

            Returns:
                The stdout output from the code execution, or error message.
                When background=True, returns a handle like "Dispatched as bg-exec-N ..."
            """
            # Mutate execution_globals in place — `ctx.deps.state.execution_globals`
            # is the same object as `self.state.execution_globals` on the agent, so
            # rebinding (`= {...}`) would silently break cross-turn variable
            # persistence (the agent would keep the empty original).
            #
            # Tool wrappers and helpers are reinstalled every call. They
            # close over the current ctx.deps (different on every turn);
            # gating reinstallation on "dict is empty" would mean a session
            # resume with restored user vars never installs them at all.
            # Cheap to re-create, identical to user vars in keyspace, no
            # collisions in practice.
            ctx.deps.state.execution_globals.setdefault("__builtins__", __builtins__)
            ctx.deps.state.execution_globals.setdefault("asyncio", asyncio)
            for namespace, tool_instance in ctx.deps.tools.items():
                ctx.deps.state.execution_globals[namespace] = ToolWrapper(tool_instance, ctx.deps, namespace)

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
                session_dir = ctx.deps.session_manager.session_dir(ctx.deps.state.info.session_id)
                session_dir.mkdir(parents=True, exist_ok=True)
                return str(session_dir)

            ctx.deps.state.execution_globals["get_session_dir"] = get_session_dir

            is_expensive = any(p in code for p in self._expensive_patterns)

            if self.approval_callback and not self.auto_approve and is_expensive:
                self.spinner_text = ""
                decision = await self.approval_callback(code)
                if decision is False:
                    self._finish_pending_tool_call(error="skipped by user")
                    return "Code execution skipped by user."
                if decision == "__auto_approve__":
                    self.auto_approve = True
                elif isinstance(decision, str):
                    self._finish_pending_tool_call(error=f"feedback: {decision}")
                    return f"User feedback before execution: {decision}\n\nRevise the code and try again."

            # Media/expensive code always backgrounds after approval passes.
            if is_expensive:
                background = True

            self.spinner_text = "Running..."

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
                    self._sync_session_assets(ctx.deps)

            if background:
                self._bg_exec_counter += 1
                bg_id = f"bg-exec-{self._bg_exec_counter}"
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

                self._bg_logs[bg_id] = {
                    "kind": "bg-exec",
                    "label": _extract_bg_label(code),
                    "started_at": time.time(),
                    "ended_at": None,
                    "status": "running",
                    "_reader": _reader,
                }

                # Hand the ToolCall event from _display_tool_call to the bg
                # thread so it can resolve (ok/error + output) in-place when
                # the code finishes. This gives the user a single foreground
                # widget that represents the bg job throughout its lifetime.
                bound_tool_call = self._pending_tool_calls.pop(0) if self._pending_tool_calls else None
                if bound_tool_call is not None:
                    try:
                        bound_tool_call.args = f"{bound_tool_call.args} → {bg_id} running"
                        self.emitter.emit(bound_tool_call)
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
                        self._sync_session_assets(ctx.deps)
                        new_artifacts = len(ctx.deps.state.artifacts) - before_artifacts
                        log_entry = self._bg_logs.get(bg_id)
                        if log_entry is not None:
                            log_entry["status"] = "error" if error else "ok"
                            log_entry["ended_at"] = time.time()
                        if error:
                            self.emitter.emit(SystemNote(f"✗ {bg_id} failed:\n{error}", level="error"))
                        else:
                            tail = (
                                f" — {new_artifacts} artifact{'s' if new_artifacts != 1 else ''}"
                                if new_artifacts
                                else ""
                            )
                            self.emitter.emit(SystemNote(f"✓ {bg_id} done{tail}"))
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
                                self.emitter.emit(bound_tool_call.finish(output=merged_output, error=error))
                            except Exception:
                                pass
                    finally:
                        self._bg_execs.pop(bg_id, None)
                        produced_artifacts = list(ctx.deps.state.artifacts[before_artifacts:])
                        if self.on_bg_complete is not None:
                            try:
                                self.on_bg_complete(
                                    bg_id=bg_id,
                                    error=error,
                                    output=output,
                                    new_artifacts=new_artifacts,
                                    produced_artifacts=produced_artifacts,
                                )
                            except Exception as cb_err:
                                self.emitter.emit(SystemNote(f"on_bg_complete callback failed: {cb_err}", level="error"))

                thread = threading.Thread(target=_run_bg, daemon=True, name=f"okonomi-{bg_id}")
                self._bg_execs[bg_id] = thread
                thread.start()

                # Brief wait so fast failures (KeyError, fal arg validation,
                # 4xx, malformed loras) surface as the *actual* tool result
                # instead of letting the model narrate around an already-
                # broken job. 2s catches almost every fast failure without
                # noticeably stalling successful long-running jobs (which
                # take 10–90s on fal endpoints).
                thread.join(timeout=2.0)
                self.spinner_text = ""

                log_entry = self._bg_logs.get(bg_id, {})
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
                self._finish_pending_tool_call(output=output or "", error=error)

                if error:
                    hint = ""
                    # If the code hit a parsing error *after* touching an
                    # expensive tool, remind the model that globals persist
                    # and it should NOT re-run the expensive call.
                    if any(p in code for p in self._expensive_patterns) and any(
                        kind in error for kind in ("KeyError", "AttributeError", "TypeError", "IndexError")
                    ):
                        hint = (
                            "\n\nHINT: The generation call likely succeeded — "
                            "the `result` variable is still in globals. DO NOT "
                            "re-run genmedia.run/browser.render/fal_client. Write a new "
                            "execute_code block with `print(result)` to inspect "
                            "its shape, then fix the access pattern."
                        )
                    return f"Error executing code:\n{error}{hint}"

                if output:
                    from ..context import _strip_large_content

                    cleaned = _strip_large_content(output)
                    if len(cleaned) > 10_000:
                        cleaned = cleaned[:8_000] + "\n...[truncated]...\n" + cleaned[-1_500:]
                    return cleaned

                return "Code executed successfully (no output)"

            except asyncio.CancelledError:
                self.spinner_text = ""
                raise
            finally:
                self.spinner_text = ""
