"""TuiSession — drives the agent from the streaming-scrollback REPL.

Owns a persistent worker loop on a daemon thread; user messages are
serialized through an ``asyncio.Queue`` and consumed one turn at a
time. Cross-thread coordination (cancel, submit, approval reply) marshals
through ``call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from concurrent.futures import TimeoutError as FutureTimeout

from ..agent import OkonomiAgent
from ..events import ApprovalRequest, Event, EventEmitter, JobFinished, SystemNote, ToolCallDisplay
from ..session import SessionStatus, prune_empty_sessions
from ..utils import format_elapsed
from .app import ReplApp
from .assets import handle_asset_action
from .events import Banner, Divider, UserMessage
from .transcript import replay_history


class TuiSession:
    """Streaming-scrollback REPL session for OkonomiAgent."""

    def __init__(self, agent: OkonomiAgent) -> None:
        self.agent = agent
        self.emitter = agent.emitter

        # Persistent worker loop on its own thread. Turns are submitted
        # via run_coroutine_threadsafe, serialized through an asyncio
        # queue consumed by _worker_loop. Cancellation flips a flag
        # on self._current_task via the same loop.
        self._loop = asyncio.new_event_loop()
        self._agent_thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="okonomi-agent")
        self._agent_thread.start()

        self._input_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._current_task: asyncio.Task | None = None
        self._worker_loop_future = asyncio.run_coroutine_threadsafe(self._worker_loop(), self._loop)

        # The agent surfaces approvals and background-job completions as
        # events; this session subscribes and reacts. The inline approval
        # UI is single-slot (one input box), so concurrent requests —
        # e.g. a background agent and the foreground turn — queue and
        # show one at a time.
        self._approval_queue: deque[ApprovalRequest] = deque()
        self._approval_active = False
        self._approval_lock = threading.Lock()
        self.agent.approvals_enabled = True
        self.emitter.subscribe(self._on_agent_event)

        # First-turn reset wipes message_history/artifacts/etc. for a
        # fresh start. Skip it if the agent was already loaded (CLI
        # ``--resume`` / ``--session`` path) — otherwise the first
        # prompt blows away the resumed state.
        self._next_reset: bool = not bool(self.agent.state.message_history)

        # Wall-clock start of the in-flight turn. Used by the status bar
        # to render an elapsed timer ("Thinking… 1m 23s"). None when idle.
        self._turn_started_at: float | None = None

        # Background full-agent tasks (not the agent's per-execute_code
        # bg-exec threads). Keyed by bg-N id.
        self._bg_tasks: dict[str, threading.Thread] = {}
        self._bg_counter = 0

        # The prompt_toolkit streaming-scrollback REPL front-end. Prints the
        # EventEmitter stream into native terminal scrollback and owns the
        # bottom input chrome.
        self._app = ReplApp(
            self.emitter,
            on_submit=self._on_submit,
            on_interrupt=self._handle_interrupt,
            spinner_label_fn=self._spinner_label,
            usage_fn=self._usage_label,
            bg_tasks_fn=self._bg_tasks_list,
            on_double_escape=self._open_tree,
        )

    # --- Agent worker ---

    @property
    def _agent_running(self) -> bool:
        task = self._current_task
        return task is not None and not task.done()

    async def _worker_loop(self) -> None:
        """Pull messages from the asyncio queue, run one turn per message."""
        while True:
            msg = await self._input_queue.get()
            if msg is None:
                break
            reset = self._next_reset
            self._next_reset = False
            await self._run_agent_turn(msg, reset=reset)

    async def _run_agent_turn(self, user_message: str, reset: bool) -> None:
        self._turn_started_at = time.monotonic()
        self._current_task = asyncio.current_task()
        status: SessionStatus = "completed"
        try:
            await self.agent.run_async(user_message, reset=reset)
        except asyncio.CancelledError:
            status = "interrupted"
            self.emitter.emit(SystemNote("Interrupted. History preserved.", level="warn"))
        except Exception as e:  # noqa: BLE001
            import traceback

            status = "error"
            tb = traceback.format_exc()
            self.emitter.emit(SystemNote(f"Agent error: {type(e).__name__}: {e}", level="error"))
            trace_event = self.emitter.emit(ToolCallDisplay("traceback", args="agent failure", details=tb))
            self.emitter.emit(trace_event.finish(error="see traceback above"))
        finally:
            self._current_task = None
            self._turn_started_at = None
            # Off-load to a thread — save does file + DB writes and a
            # pickle dump that can take a moment. Blocking the consumer
            # here would stall queued submits and cancellations
            # marshalled through the same loop.
            await asyncio.to_thread(self._safe_save_session, status=status)

    def _safe_save_session(
        self,
        *,
        status: SessionStatus,
        download_assets: bool = False,
    ) -> None:
        """Save the session swallowing any exception.

        Called from multiple places (per-turn finally, app exit, atexit)
        so it must never raise — a save error must not crash the session.
        ``download_assets`` defaults to False because the agent's eager
        background download pool keeps assets up to date independently.
        """
        try:
            self.agent.save_session(
                status=status,
                download_assets=download_assets,
            )
        except Exception:  # noqa: BLE001
            pass

    def cancel_current_turn(self) -> None:
        """Cancel the in-flight agent turn (Ctrl+C).

        Marshalled through the worker loop so the task-read and the
        cancel-call happen on the same thread that owns the task.
        """

        def _do_cancel() -> None:
            task = self._current_task
            if task is not None and not task.done():
                task.cancel()

        self._loop.call_soon_threadsafe(_do_cancel)

    def _handle_interrupt(self) -> bool:
        """Ctrl+C hook for ReplApp: cancel the in-flight turn if one is
        running. Returns True if a turn was cancelled, False if idle (the
        app then clears the draft)."""
        if self._agent_running:
            self.cancel_current_turn()
            return True
        return False

    # --- Submit / approval / bg-exec hooks ---

    def _on_submit(self, text: str) -> None:
        """Handle user text from the REPL.

        Runs on the prompt_toolkit loop; push work to the worker via queue.
        Intercepts slash commands before they reach the agent.
        """
        from . import commands
        from .commands import Action

        self.emitter.emit(Divider())
        self.emitter.emit(UserMessage(text))

        if commands.is_slash_command(text):
            result = commands.handle(text, self.agent, self.emitter)
            if result is None:
                return  # handled locally, no agent turn
            if result.action is Action.EXIT:
                self._app.exit()
                return
            if result.action is Action.SESSION_PICKER:
                self._app.show_session_picker(self._resume_session)
                return
            if result.action is Action.TREE_PICKER:
                self._open_tree()
                return
            if result.action is Action.ASSET_PICKER:
                self._app.show_asset_picker(
                    list(self.agent.state.artifacts),
                    lambda action, paths: handle_asset_action(action, paths, self.agent, self.emitter),
                )
                return
            if result.action is Action.VERBOSE:
                self._app.toggle_verbose()
                return
            if result.action is Action.BG:
                self._spawn_bg_task(result.payload)
                return
            if result.action is Action.BTW:
                self._inject_side_question(result.payload)
                return

            # Action.FORWARD — send the command's payload to the agent.
            text = result.payload

        self._submit(text)

    def _submit(self, msg: str | None) -> None:
        """Schedule a message onto the worker loop's input queue.

        ``None`` is the shutdown sentinel — the consumer breaks on it.
        """
        self._loop.call_soon_threadsafe(self._input_queue.put_nowait, msg)

    def _inject_side_question(self, question: str) -> None:
        """Cancel the in-flight turn and queue the question as a course
        correction. The agent's run_async path already handles
        ``_was_interrupted`` by prepending the new message with a
        priority marker, so this works with no agent-side changes.
        """
        if self._agent_running:
            self.cancel_current_turn()
            # Tag the message so the agent knows this is a side question
            # rather than a routine continuation. The "interrupted"
            # framing in run_async will pick it up.
            tagged = f"[Side question while you were working]\n\n{question}"
        else:
            tagged = question
        self._submit(tagged)

    def _spawn_bg_task(self, task_input: str) -> None:
        """Run a task in the background with a separate OkonomiAgent.

        Mirrors the prompt_toolkit CLI's _run_bg_task. Outputs merge back
        into the parent agent's session on completion.
        """
        self._bg_counter += 1
        task_id = f"bg-{self._bg_counter}"
        parent = self.agent

        self.emitter.emit(SystemNote(f"Background task {task_id} started: {task_input[:80]}"))

        def _forward(ev: Event) -> None:
            # Approvals and job completions from the bg agent surface on
            # the foreground emitter (the events carry their own resolve
            # handles / payloads); everything else stays isolated.
            if isinstance(ev, (ApprovalRequest, JobFinished)):
                self.emitter.emit(ev)

        # Register in the shared job registry so /jobs and /logs see /bg
        # tasks alongside bg-exec-N and term-N.
        progress = {"status": "running", "detail": ""}

        def _reader() -> str:
            return f"task: {task_input}\nstatus: {progress['status']}" + (
                f"\n{progress['detail']}" if progress["detail"] else ""
            )

        parent._register_bg_log(task_id=task_id, label=task_input[:60], reader=_reader)

        def _run() -> None:
            status = "ok"
            bg_agent = OkonomiAgent(
                model_id=parent.model_id,
                timeout=parent.timeout,
                compress_at_tokens=parent.compress_at_tokens,
                session_id=parent.session.session_id,
            )
            # Isolated emitter — the bg agent's tool calls and streaming
            # text don't show up in the foreground history; only approvals,
            # job completions, and the note below surface to the user.
            bg_agent.emitter = EventEmitter()
            bg_agent.approvals_enabled = True
            bg_agent.emitter.subscribe(_forward)
            try:
                asyncio.run(bg_agent.run_async(task_input, reset=True))
                parent.state.artifacts.extend(bg_agent.state.artifacts)
                n = len(bg_agent.state.artifacts)
                progress["detail"] = f"artifacts produced: {n}"
                self.emitter.emit(SystemNote(f"Background task {task_id} done — {n} artifact{'s' if n != 1 else ''}"))
            except Exception as e:  # noqa: BLE001
                status = "error"
                progress["detail"] = f"error: {e}"
                self.emitter.emit(SystemNote(f"Background task {task_id} failed: {e}", level="error"))
            finally:
                progress["status"] = status
                parent._mark_bg_log_done(task_id=task_id, status=status)
                self._bg_tasks.pop(task_id, None)

        label = task_input[:50] + "…" if len(task_input) > 50 else task_input
        thread = threading.Thread(target=_run, daemon=True, name=label)
        self._bg_tasks[task_id] = thread
        thread.start()

    def _resume_session(self, session_id: str) -> None:
        """Callback from the session picker. Loads a past session and
        replays its conversation into the chat view so the user can
        actually see where they left off (otherwise the scrollback is
        empty even though the model has full context)."""
        if not self.agent.load_session(session_id):
            self.emitter.emit(SystemNote(f"Failed to load session: {session_id}", level="error"))
            return

        # Skip the first-turn reset — it would wipe the freshly-loaded
        # artifacts and message history.
        self._next_reset = False

        replay_history(self.agent.state.message_history, self.emitter)
        title = self.agent.session.title
        label = f"{session_id[:10]} · {title}" if title else session_id
        self.emitter.emit(SystemNote(f"↑ Resumed: {label}"))

    def _open_tree(self) -> None:
        """Open the session-tree navigator (from /tree or double-Esc)."""
        if self._agent_running:
            self.emitter.emit(SystemNote("Finish or interrupt the current turn before browsing the tree.", level="warn"))
            return
        self._app.show_tree_picker(self.agent.session.session_id, self._checkout_branch)

    def _checkout_branch(self, target_event_id: int | None, prefill: str, summarize: bool) -> None:
        """Callback from the tree picker: optionally summarize the branch
        being abandoned, flip the active path in the DB, reload the agent
        from it, replay the transcript, and prefill the editor for
        edit-and-resubmit.

        Only reachable while idle (``_open_tree`` gates on a running
        turn), so no turn-end save can race the flag flip. A message
        already sitting in the input queue is an accepted residual race.
        """
        import json

        from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, UserPromptPart

        from .. import db
        from ..events import CONVERSATION_TYPES

        session_id = self.agent.session.session_id
        conn = db.connection()

        summary = None
        if summarize:
            rows = db.fetch_events(conn, session_id, active_only=False)
            parents = {r["event_id"]: r["parent_id"] for r in rows}
            path: set[int] = set()
            node = target_event_id
            while node is not None:
                path.add(node)
                node = parents.get(node)
            conv_types = {t.value for t in CONVERSATION_TYPES}
            abandoned = [r for r in rows if r["active"] and r["type"] in conv_types and r["event_id"] not in path]
            if abandoned:
                self.emitter.emit(SystemNote("Summarizing abandoned branch…"))
                try:
                    raw = [json.loads(r["payload"]) for r in abandoned]
                    msgs = list(ModelMessagesTypeAdapter.validate_python(raw))
                    summary = self.agent.context_manager.summarize_branch(msgs)
                except Exception:
                    summary = None
                if summary is None:
                    self.emitter.emit(SystemNote("Branch summary failed — continuing without it.", level="warn"))

        with db.transaction(conn):
            db.checkout_events(conn, session_id, target_event_id)

        if not self.agent.load_session(session_id):
            self.emitter.emit(SystemNote(f"Failed to reload session after checkout: {session_id}", level="error"))
            return
        self._next_reset = False

        if summary:
            # Persisted as a plain USER_MESSAGE by the next save's
            # _sync_conversation delta — same pattern as compaction summaries.
            msg = ModelRequest(parts=[UserPromptPart(content=summary)])
            self.agent.state.message_history.append(msg)
            self.agent.state.context.append(msg)

        replay_history(self.agent.state.message_history, self.emitter)
        if prefill:
            self.emitter.emit(SystemNote("↷ Checked out — edit the restored prompt and submit to branch."))
            self._app.prefill_input(prefill)
        else:
            self.emitter.emit(SystemNote("↷ Checked out — continue from here to branch."))

    def _on_agent_event(self, ev: Event) -> None:
        """React to agent events that need more than rendering. Runs
        synchronously on the emitting thread — must never block."""
        if isinstance(ev, ApprovalRequest):
            self._enqueue_approval(ev)
        elif isinstance(ev, JobFinished):
            self._on_job_finished(ev)

    def _enqueue_approval(self, request: ApprovalRequest) -> None:
        """Queue an approval request for the single-slot inline UI; show
        it immediately if nothing else is on screen. The event carries its
        own resolve handle, so requests from background agents (forwarded
        onto this emitter) resolve their own waiters."""
        with self._approval_lock:
            if self._approval_active:
                self._approval_queue.append(request)
                return
            self._approval_active = True
        self._show_approval(request)

    def _show_approval(self, request: ApprovalRequest) -> None:
        def _resolve(decision: bool | str) -> None:
            try:
                if request.respond is not None:
                    request.respond(decision)
            finally:
                with self._approval_lock:
                    nxt = self._approval_queue.popleft() if self._approval_queue else None
                    if nxt is None:
                        self._approval_active = False
                if nxt is not None:
                    self._show_approval(nxt)

        self._app.call_from_thread(self._app.show_approval, request.code, _resolve)

    def _on_job_finished(self, ev: JobFinished) -> None:
        """Inject a synthetic turn so the agent can surface completed media.

        Names the *specific* assets this bg task produced so the agent
        doesn't grab unrelated older items from session state.
        """
        bg_id, error = ev.job_id, ev.error
        if error:
            msg = (
                f"[system] Background task {bg_id} failed:\n{error}\n\n"
                "DO NOT re-run genmedia.run, browser.render, or any "
                "other expensive generation call to recover. If this "
                "is a parsing error (KeyError / AttributeError / TypeError "
                "after the generation line), the generation already "
                "succeeded — the `result` variable is still in globals from "
                "the bg run. Write a new execute_code block with "
                "`print(result)` to inspect its shape and fix the access. "
                "If the generation call itself raised (network/500/timeout), "
                "surface the error to the user and ask before retrying."
            )
            self._submit(msg)
            return

        asset_lines: list[str] = []
        for ref_info in ev.artifact_refs:
            ref = ref_info.get("path") or ref_info.get("url") or ""
            kind = ref_info.get("artifact_type") or "artifact"
            if ref:
                asset_lines.append(f"  - {kind}: {ref}")
        produced_block = "\n".join(asset_lines) if asset_lines else "  (none)"

        msg = (
            f"[system] Background task {bg_id} completed.\n\n"
            f"Assets produced by THIS task:\n{produced_block}\n\n"
            "Use ONLY these specific URLs/paths when presenting to the user. "
            "Do NOT reference older session assets and do NOT regenerate. "
            "The list above is authoritative for what "
            f"{bg_id} just produced."
        )
        self._submit(msg)

    # --- Status-bar callables passed to ReplApp ---

    def _spinner_label(self) -> str:
        """Working-state label with elapsed timer. Always shows liveness
        while a turn is in flight (even when ``agent.status`` is
        briefly empty between tool calls)."""
        txt = self.agent.status or ""
        if not txt and self._agent_running:
            txt = "Working..."
        if not txt:
            return ""
        if self._turn_started_at is None:
            return txt
        return f"{txt}  ({format_elapsed(time.monotonic() - self._turn_started_at)})"

    def _usage_label(self) -> str:
        usage = self.agent.session.tokens
        if not usage.total_tokens:
            return ""
        return str(usage)

    def _bg_tasks_list(self) -> list[tuple[str, str]]:
        """(id, label) pairs for in-flight bg-exec + /bg tasks. Threads
        that have finished but not yet been popped from their dicts are
        filtered (race around thread.is_alive during shutdown)."""
        out: list[tuple[str, str]] = []
        for bg_id, thread in list(self.agent.executor.threads.items()):
            if thread and thread.is_alive():
                label = (self.agent._bg_logs.get(bg_id) or {}).get("label", "")
                out.append((bg_id, label))
        for task_id, thread in list(self._bg_tasks.items()):
            if thread and thread.is_alive():
                out.append((task_id, thread.name))
        return out

    # --- Entry points ---

    async def run(self, initial_message: str = "") -> None:
        import atexit
        import shutil

        # Last-ditch save: atexit fires before interpreter exit even on
        # SIGTERM, an uncaught exception, or a hung front-end.
        atexit.register(self._safe_save_session, status="abandoned")

        try:
            prune_empty_sessions()
        except Exception:  # noqa: BLE001
            pass

        self.emitter.emit(Banner())
        self.emitter.emit(SystemNote(
            "type a message or /help. Ctrl+D to exit. Scroll and select text "
            "natively; /verbose toggles full tool output.",
        ))

        # A preloaded session (CLI --resume / --session) renders its prior
        # conversation into the scrollback, same as the /session picker —
        # otherwise the model has context the user can't see.
        if self.agent.state.message_history:
            replay_history(self.agent.state.message_history, self.emitter)
            title = self.agent.session.title
            sid = self.agent.session.session_id
            label = f"{sid[:10]} · {title}" if title else sid
            self.emitter.emit(SystemNote(f"↑ Resumed: {label}"))

        if shutil.which("genmedia") is None:
            self.emitter.emit(SystemNote(
                "genmedia CLI not found on $PATH. Install with:\n"
                "  curl -fsSL https://genmedia.sh/install | sh\n"
                "Until then, image/video generation via the genmedia "
                "skill won't work — local tools (browser, prompt, "
                "research) still do.",
                level="warn",
            ))

        if initial_message:
            self._on_submit(initial_message)

        try:
            await self._app.run_async()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Tear down the worker loop and persist a final save.

        Cancels any in-flight turn, posts the consumer-stop sentinel,
        stops the loop, joins the thread, drops in-flight asset
        downloads, and saves session metadata (URLs survive in the DB;
        resume retries anything missing)."""
        if self._agent_running:
            self.cancel_current_turn()
        self._submit(None)
        try:
            self._worker_loop_future.result(timeout=2.0)
        except FutureTimeout:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._agent_thread.join(timeout=2.0)
        try:
            self.agent._asset_dl_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
        self._safe_save_session(status="completed", download_assets=False)
