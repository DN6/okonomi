"""Background-task registry.

Tracks term-N (terminal.bash
background tasks), and launched agents so ``/tasks`` can
inspect them after they finish, and emits their lifecycle events. Not core
to request/tool/context handling — the agent just holds one instance.
"""

import asyncio
import time
from typing import Any

from ..events import EventEmitter, TaskFinished, TaskStarted


class TaskRegistry:
    """Registry + lifecycle events for background tasks. Entries survive
    completion so ``/tasks`` can read after a task finishes; each entry is
    ``{"kind", "label", "started_at", "ended_at", "status", "_reader"}`` where
    ``_reader`` is a zero-arg callable returning the current snapshot."""

    def __init__(self, emitter: EventEmitter):
        self._emitter = emitter
        self.entries: dict[str, dict[str, Any]] = {}

    def register(self, *, task_id: str, label: str, reader, kind: str | None = None) -> None:
        """Record a starting task (term, launched agent, /bg task) and emit
        ``TaskStarted``. ``kind`` defaults to the task_id prefix so ``/tasks``
        renders the right column without each call site threading it through."""
        prefix = kind or task_id.split("-", 1)[0]
        self.entries[task_id] = {
            "kind": prefix,
            "label": label,
            "started_at": time.time(),
            "ended_at": None,
            "status": "running",
            "_reader": reader,
        }
        self._emitter.emit(TaskStarted(task_id=task_id, kind=prefix, label=label))

    def mark_done(self, *, task_id: str, status: str) -> None:
        """Mark a registered task (e.g. term-N) finished."""
        entry = self.entries.get(task_id)
        if entry is not None:
            entry["status"] = status
            entry["ended_at"] = time.time()

    def finished(
        self,
        *,
        bg_id: str,
        error: str | None = None,
        output: str | None = None,
        produced_artifacts: list | None = None,
    ) -> None:
        """Emit ``TaskFinished`` for a completed background task (term
        blocks, term commands and launched agents all route here)."""
        refs = [{"artifact_type": a.artifact_type, "path": a.path, "url": a.url} for a in produced_artifacts or []]
        self._emitter.emit(
            TaskFinished(
                task_id=bg_id,
                kind=self.entries.get(bg_id, {}).get("kind", ""),
                status="error" if error else "ok",
                error=error,
                output=output,
                artifact_refs=refs,
            )
        )


# Tasks whose result the owning agent must be told about in a follow-up
# turn. Shell tasks (term-N) are excluded: a running one must not hold a
# one-shot run or a sub-agent open.
REPORTED_KINDS = ("agent", "tool")


def task_notice(ev: TaskFinished) -> str:
    """The follow-up message an agent receives when one of its tasks finishes."""
    if ev.kind == "agent":
        return agent_notice(ev)
    if ev.kind == "tool":
        return tool_notice(ev)
    return term_notice(ev)


def _produced_block(ev: TaskFinished) -> str:
    """The task's artifacts as an indented list, or ``(none)``."""
    artifact_lines: list[str] = []
    for ref_info in ev.artifact_refs:
        ref = ref_info.get("path") or ref_info.get("url") or ""
        kind = ref_info.get("artifact_type") or "artifact"
        if ref:
            artifact_lines.append(f"  - {kind}: {ref}")
    return "\n".join(artifact_lines) if artifact_lines else "  (none)"


async def run_until_tasks_report(agent, message: str) -> str | None:
    """Run ``message`` as a turn, then one more turn per agent or detached
    tool call the turn started, until all have reported. For callers with
    no input loop to deliver completion notices later (one-shot runs,
    launched sub-agents); returning after the first turn would drop their
    results. Returns the reply to the last turn."""
    loop = asyncio.get_running_loop()
    notices: asyncio.Queue[TaskFinished] = asyncio.Queue()

    def _collect(ev) -> None:
        if isinstance(ev, TaskFinished) and ev.kind in REPORTED_KINDS:
            loop.call_soon_threadsafe(notices.put_nowait, ev)

    agent.emitter.subscribe(_collect)
    reply = await agent.run_async(message, reset=True)
    # Counted by task id rather than by "still running": a task is marked
    # done just before its notice is emitted.
    reported: set[str] = set()
    while True:
        launched = {task_id for task_id, entry in agent.tasks.entries.items() if entry["kind"] in REPORTED_KINDS}
        if launched <= reported:
            return reply
        ev = await notices.get()
        reported.add(ev.task_id)
        reply = await agent.run_async(task_notice(ev))


def term_notice(ev: TaskFinished) -> str:
    """The message for a finished background shell task (term-N). Names the
    *specific* artifacts it produced so the agent doesn't grab unrelated older
    items from session state, and carries the command's output."""
    output = f"\n\nOutput:\n{ev.output}" if ev.output else ""
    if ev.error:
        return (
            f"[system] Background task {ev.task_id} failed:\n{ev.error}{output}\n\n"
            "DO NOT re-run an expensive generation call to recover. "
            "If this is a parsing error after the generation line, the "
            "generation likely succeeded — check $OKONOMI_SESSION_DIR "
            "for its output files and `/tasks` for the run's stdout "
            "before re-running anything. If the generation call itself "
            "raised (network/500/timeout), surface the error to the "
            "user and ask before retrying."
        )

    produced_block = _produced_block(ev)
    return (
        f"[system] Background task {ev.task_id} completed.{output}\n\n"
        f"Artifacts produced by THIS task:\n{produced_block}\n\n"
        "Use ONLY these specific URLs/paths when presenting to the user. "
        "Do NOT reference older session artifacts and do NOT regenerate. "
        "The list above is authoritative for what "
        f"{ev.task_id} just produced."
    )


def tool_notice(ev: TaskFinished) -> str:
    """The message an agent receives when one of its tool calls that was
    moved to the background finishes. Carries the tool's result, which
    never reached the model as a tool return."""
    if ev.error:
        return f"[system] Background tool call {ev.task_id} failed:\n{ev.error}"
    return f"[system] Background tool call {ev.task_id} finished. Its result:\n{ev.output}"


def agent_notice(ev: TaskFinished) -> str:
    """The message a parent agent receives when an agent it launched
    settles. Shared by every consumer that turns the event into a parent
    turn, so the parent reads the same text whichever one delivered it."""
    if ev.error:
        return f"[system] Background agent {ev.task_id} failed before it finished:\n{ev.error}"

    produced_block = _produced_block(ev)

    if ev.output:
        closing = f"Its closing message:\n{ev.output}"
    else:
        closing = "It left no closing message."
    return (
        f"[system] Background agent {ev.task_id} finished and will do no further work.\n\n"
        f"{closing}\n\n"
        f"Artifacts it produced:\n{produced_block}"
    )
