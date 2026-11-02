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
from ..agent.jobs import job_notice
from ..events import ApprovalRequest, Event, EventEmitter, JobFinished, SystemNote, ToolCallDisplay
from ..session import Artifact, SessionStatus, prune_empty_sessions, search_artifacts
from ..utils import format_elapsed
from .app import ReplApp
from .assets import handle_asset_action
from .events import Banner, Divider, UserMessage
from .transcript import replay_history


class TuiSession:
    """Streaming-scrollback REPL session for OkonomiAgent."""

    def __init__(self, agent: OkonomiAgent) -> None:
        self.agent = agent
        self.agent.can_prompt = True
        self.emitter = agent.emitter

        # Persistent worker loop on its own thread. Turns are submitted
        # via call_soon_threadsafe, serialized through an asyncio
        # queue consumed by _worker_loop. Cancellation cancels the
        # running turn's task on the same loop.
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
        self.emitter.subscribe(self._on_agent_event)

        # First-turn reset wipes message_history/artifacts/etc. for a
        # fresh start. Skip it if the agent was already loaded (CLI
        # ``--resume`` / ``--session`` path) — otherwise the first
        # prompt blows away the resumed state.
        self._next_reset: bool = not bool(self.agent.state.message_history)

        # Wall-clock start of the in-flight turn. Used by the status bar
        # to render an elapsed timer ("Thinking… 1m 23s"). None when idle.
        self._turn_started_at: float | None = None

        # Set by cancel_current_turn (Ctrl+C) so the CancelledError handler
        # can tell a user interrupt from a library-leaked cancellation
        # (aiohttp timeouts cancel the calling task; a missed conversion
        # escapes as bare CancelledError and is NOT a user interrupt).
        self._user_cancel: bool = False
        # Messages that cut into a streaming reply; the turn is cancelled
        # and they run next, together with anything it never picked up.
        self._steer_texts: list[str] = []
        # After a user stop the agent stays quiet until the user speaks:
        # job results are held and ride along with their next message.
        self._quiet = False
        self._held_notices: list[str] = []

        # Background full-agent tasks (/bg) are numbered bg-N.
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
            on_history_search=self._open_history,
            on_jobs=self._open_jobs,
            is_busy=lambda: self._agent_running,
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
        stopped = False
        try:
            await self.agent.run_async(user_message, reset=reset)
        except asyncio.CancelledError:
            status = "interrupted"
            # This task is the worker loop itself — reset its cancellation
            # bookkeeping so the caught cancel can't poison later turns
            # (structured-concurrency primitives compare cancelling() counts).
            task = asyncio.current_task()
            if task is not None:
                while task.cancelling():
                    task.uncancel()
            stopped = self._user_cancel
            if stopped:
                self.emitter.emit(SystemNote("Interrupted. History preserved.", level="warn"))
            elif self._steer_texts:
                self.emitter.emit(SystemNote("↳ Redirecting."))
            else:
                self.emitter.emit(
                    SystemNote(
                        "Turn cancelled from inside the provider stack (not Ctrl+C) — "
                        "likely a connection timeout. History preserved.",
                        level="error",
                    )
                )
            self._user_cancel = False
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
            # Messages sent mid-turn that the turn ended before picking up.
            # After a user stop they go back to the input box rather than
            # starting a turn: stop has to mean the agent goes quiet.
            pending = [*self.agent.take_undelivered(), *self._steer_texts]
            self._steer_texts = []
            if stopped:
                self._quiet = True
                if pending:
                    self._app.call_from_thread(self._restore_draft, pending)
            elif pending:
                self._input_queue.put_nowait("\n\n".join(pending))
            # Off-load to a thread — save does file + DB writes that can
            # take a moment. Blocking the consumer here would stall queued
            # submits and cancellations marshalled through the same loop.
            await asyncio.to_thread(self._safe_save_session, status=status)

    def _restore_draft(self, texts: list[str]) -> None:
        draft = " ".join([*texts, self._app.input.text]).strip()
        self._app.prefill_input(draft)
        self.emitter.emit(SystemNote("Your unsent message is back in the input box — Enter to send it."))

    def _safe_save_session(self, *, status: SessionStatus) -> None:
        """Save the session swallowing any exception.

        Called from multiple places (per-turn finally, app exit, atexit)
        so it must never raise — a save error must not crash the session.
        Assets aren't downloaded here: the agent's eager background
        download pool keeps them up to date independently.
        """
        try:
            self.agent.save_session(status=status, download_assets=False)
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
                self._user_cancel = True
                task.cancel()
                # Cancelling the task abandons the executor thread running
                # a sync tool; the subprocess would otherwise live on until
                # its own timeout.
                self.agent.terminal._kill_foreground()

        self._loop.call_soon_threadsafe(_do_cancel)

    def _handle_interrupt(self) -> bool:
        """Ctrl+C hook for ReplApp: cancel the in-flight turn if one is
        running. Returns True if a turn was cancelled, False if idle (the
        app then clears the draft)."""
        if self._agent_running:
            self.cancel_current_turn()
            return True
        return False

    # --- Submit / approval / bg hooks ---

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
            if result.action is Action.FIND_PICKER:
                self._app.show_find_picker(result.payload, result.scope, self._open_hit)
                return
            if result.action is Action.TREE_PICKER:
                self._open_tree()
                return
            if result.action is Action.ASSET_PICKER:
                artifacts = list(self.agent.state.artifacts)
                if result.payload:
                    artifacts = [
                        Artifact(h["artifact_type"], path=h["path"], url=h["url"], prompt=h["prompt"])
                        for h in search_artifacts(result.payload, limit=200)
                    ]
                self._app.show_asset_picker(
                    artifacts,
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

        self._loop.call_soon_threadsafe(self._deliver_on_loop, text)

    def _deliver_on_loop(self, text: str) -> None:
        """Send a user message into the running turn, where it is picked up
        at the turn's next model request; with no turn running, queue it as
        a new turn. Runs on the worker loop, so it can't race the turn's
        start or end.

        If the model is mid-reply (streaming text, no tool running) the
        message cuts in: the turn is cancelled and the message runs next
        with the partial reply as context, instead of waiting for the model
        to finish talking."""
        self._quiet = False
        if self._held_notices:
            text = "\n\n".join([text, *self._held_notices])
            self._held_notices = []
        if self._agent_running and (self._steer_texts or self.agent.streaming_text):
            if not self._steer_texts:
                self._current_task.cancel()
            self._steer_texts.append(text)
            return
        if not (self._agent_running and self.agent.enqueue(text)):
            self._input_queue.put_nowait(text)
        elif self.agent.status.startswith("Running"):
            # Don't make the user wait on a slow tool: it carries on as a
            # background job and the model reads the message now.
            self.agent.detach_running_tools()
            self.emitter.emit(SystemNote("↳ Running tool moved to the background so the agent can read this now."))

    def _deliver_notice(self, notice: str) -> None:
        """Hand a finished job's result to the agent: into the running turn
        if there is one, else as a new turn — unless the user stopped the
        agent, in which case it waits for their next message. Runs on the
        worker loop."""
        if self._quiet:
            self._held_notices.append(notice)
            self.emitter.emit(SystemNote("A background job finished; the agent gets its result with your next message."))
            return
        if not (self._agent_running and self.agent.enqueue(notice)):
            self._input_queue.put_nowait(notice)

    def _submit(self, msg: str | None) -> None:
        """Schedule a message onto the worker loop's input queue.

        ``None`` is the shutdown sentinel — the consumer breaks on it.
        """
        self._loop.call_soon_threadsafe(self._input_queue.put_nowait, msg)

    def _inject_side_question(self, question: str) -> None:
        """Deliver a side question like any message typed mid-turn (cuts in
        on a streaming reply, else joins the running turn), tagged so the
        agent knows it is an aside rather than new instructions."""
        if self._agent_running:
            tagged = f"[Side question while you were working]\n\n{question}"
        else:
            tagged = question
        self._loop.call_soon_threadsafe(self._deliver_on_loop, tagged)

    def _spawn_bg_task(self, task_input: str) -> None:
        """Run a task in the background with a separate OkonomiAgent.

        Outputs merge back into the parent agent's session on completion.
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
        # tasks alongside term-N.
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
            bg_agent.emitter.subscribe(_forward)
            bg_agent.can_prompt = True
            parent.jobs.entries[task_id]["jobs"] = bg_agent.jobs
            try:
                reply = asyncio.run(bg_agent.run_async(task_input, reset=True))
                parent.state.artifacts.extend(bg_agent.state.artifacts)
                n = len(bg_agent.state.artifacts)
                progress["detail"] = f"artifacts produced: {n}" + (f"\n\n{reply}" if reply else "")
                self.emitter.emit(SystemNote(f"Background task {task_id} done — {n} artifact{'s' if n != 1 else ''}"))
            except Exception as e:  # noqa: BLE001
                status = "error"
                progress["detail"] = f"error: {e}"
                self.emitter.emit(SystemNote(f"Background task {task_id} failed: {e}", level="error"))
            finally:
                progress["status"] = status
                parent._mark_bg_log_done(task_id=task_id, status=status)

        label = task_input[:50] + "…" if len(task_input) > 50 else task_input
        threading.Thread(target=_run, daemon=True, name=label).start()

    def _open_hit(self, action: str, hit: dict) -> None:
        """Callback from the find picker. Enter keeps you in okonomi: a
        session or conversation hit resumes that session; an asset or mind
        page is added to the message being typed, the same as dropping the
        file in. Ctrl-O opens an asset or page in its own app instead."""
        ref = hit.get("path") or hit.get("url") or ""
        if action == "open" and ref:
            handle_asset_action("open", [ref], self.agent, self.emitter)
            return
        if hit["source"] in ("session", "event"):
            self._resume_session(hit["session_id"])
            return
        text = self._app.input.text
        self._app.prefill_input(f"{text} {ref} " if text and not text.endswith(" ") else f"{text}{ref} ")

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

    def _open_jobs(self) -> None:
        """Open the conversation + background-jobs navigator (↓ on an
        empty prompt), mid-turn included. A no-op until something has been
        launched."""
        if not self.agent.jobs.entries:
            return
        self._app.show_jobs_picker(self.agent.session.title or self.agent.session.session_id, self.agent.jobs, self._open_tree)

    def _open_history(self) -> None:
        """Open the Ctrl-R conversation search (branch by content)."""
        if self._agent_running:
            self.emitter.emit(SystemNote("Finish or interrupt the current turn before searching history.", level="warn"))
            return
        self._app.show_history_picker(self.agent.session.session_id, self._checkout_branch)

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
        from .pickers import _abandoned

        session_id = self.agent.session.session_id
        conn = db.connection()

        summary = None
        if summarize:
            rows = db.fetch_events(conn, session_id, active_only=False)
            parents = {r["event_id"]: r["parent_id"] for r in rows}
            conv_types = {t.value for t in CONVERSATION_TYPES}
            abandoned = _abandoned(rows, parents, conv_types, target_event_id)
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

        self._app.call_from_thread(self._app.show_approval, request.code, request.reason, _resolve)

    def _on_job_finished(self, ev: JobFinished) -> None:
        """Hand the finished job's result to the agent (see ``_deliver_notice``)."""
        self._loop.call_soon_threadsafe(self._deliver_notice, job_notice(ev))

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

    def _bg_tasks_list(self) -> list[tuple[str, str, str]]:
        """(id, label, elapsed) for running background jobs, read
        from the shared JobRegistry — covers term-N, launched
        agents, and /bg tasks in one place."""
        now = time.time()
        out: list[tuple[str, str, str]] = []
        for job_id, job in list(self.agent.jobs.entries.items()):
            if job.get("status") != "running":
                continue
            out.append((job_id, job.get("label", ""), f"{now - job['started_at']:.0f}s"))
        return out

    # --- Entry points ---

    async def run(self, initial_message: str = "") -> None:
        import atexit

        # Last-ditch save: atexit fires before interpreter exit even on
        # SIGTERM, an uncaught exception, or a hung front-end.
        atexit.register(self._safe_save_session, status="abandoned")

        try:
            prune_empty_sessions()
        except Exception:  # noqa: BLE001
            pass

        self.emitter.emit(Banner())
        self.emitter.emit(SystemNote(
            "type a message or /help. Ctrl+C or Ctrl+D to exit. Scroll and select text "
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
        # Background work dies with the REPL: detached tool calls, term-N
        # jobs, and sub-agents' commands are all bash subprocesses in their
        # own process groups, so they'd outlive us otherwise. Killing them
        # also unblocks the tool-pool threads the interpreter joins at exit.
        from ..tools.terminal import kill_all_processes

        killed = kill_all_processes()
        if killed:
            self.emitter.emit(SystemNote(f"Killed {killed} running background command{'s' if killed != 1 else ''}."))
        self.agent._tool_pool.shutdown(wait=False, cancel_futures=True)
        # MCP connections are held open across turns on the worker loop; close
        # them there so stdio servers exit instead of outliving the REPL.
        try:
            asyncio.run_coroutine_threadsafe(self.agent.close_mcp(), self._loop).result(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass
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
        self._safe_save_session(status="completed")
