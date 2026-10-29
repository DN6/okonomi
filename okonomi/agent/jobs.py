"""Background-job registry.

Tracks term-N (terminal.bash
background jobs), and launched agents so ``/jobs`` and ``/logs <id>`` can
inspect them after they finish, and emits their lifecycle events. Not core
to request/tool/context handling — the agent just holds one instance.
"""

import time
from typing import Any

from ..events import EventEmitter, JobFinished, JobStarted


class JobRegistry:
    """Registry + lifecycle events for background jobs. Entries survive
    completion so ``/logs <id>`` can read after a job finishes; each entry is
    ``{"kind", "label", "started_at", "ended_at", "status", "_reader"}`` where
    ``_reader`` is a zero-arg callable returning the current snapshot."""

    def __init__(self, emitter: EventEmitter):
        self._emitter = emitter
        self.entries: dict[str, dict[str, Any]] = {}

    def register(self, *, task_id: str, label: str, reader, kind: str | None = None) -> None:
        """Record a starting job (term, launched agent, /bg task) and emit
        ``JobStarted``. ``kind`` defaults to the task_id prefix so ``/jobs``
        renders the right column without each call site threading it through."""
        prefix = kind or (task_id.split("-", 1)[0] if "-" in task_id else "task")
        self.entries[task_id] = {
            "kind": prefix,
            "label": label,
            "started_at": time.time(),
            "ended_at": None,
            "status": "running",
            "_reader": reader,
        }
        self._emitter.emit(JobStarted(job_id=task_id, kind=prefix, label=label))

    def mark_done(self, *, task_id: str, status: str) -> None:
        """Mark a registered job (e.g. term-N) finished."""
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
        new_artifacts: int = 0,
        produced_artifacts: list | None = None,
    ) -> None:
        """Emit ``JobFinished`` for a completed background job (term
        blocks, term commands and launched agents all route here)."""
        refs = [{"artifact_type": a.artifact_type, "path": a.path, "url": a.url} for a in produced_artifacts or []]
        self._emitter.emit(
            JobFinished(
                job_id=bg_id,
                kind=self.entries.get(bg_id, {}).get("kind", ""),
                status="error" if error else "ok",
                error=error,
                output=output,
                new_artifacts=new_artifacts,
                artifact_refs=refs,
            )
        )


# Jobs whose result the owning agent must be told about in a follow-up
# turn. Shell jobs (term-N) are excluded: a running one must not hold a
# one-shot run or a sub-agent open.
REPORTED_KINDS = ("agent", "tool")


def job_notice(ev: JobFinished) -> str:
    """The follow-up message an agent receives when one of its jobs finishes."""
    if ev.kind == "agent":
        return agent_notice(ev)
    if ev.kind == "tool":
        return tool_notice(ev)
    return term_notice(ev)


def term_notice(ev: JobFinished) -> str:
    """The message for a finished background shell job (term-N). Names the
    *specific* assets it produced so the agent doesn't grab unrelated older
    items from session state, and carries the command's output."""
    output = f"\n\nOutput:\n{ev.output}" if ev.output else ""
    if ev.error:
        return (
            f"[system] Background task {ev.job_id} failed:\n{ev.error}{output}\n\n"
            "DO NOT re-run an expensive generation call to recover. "
            "If this is a parsing error after the generation line, the "
            "generation likely succeeded — check $OKONOMI_SESSION_DIR "
            "for its output files and `/logs` for the run's stdout "
            "before re-running anything. If the generation call itself "
            "raised (network/500/timeout), surface the error to the "
            "user and ask before retrying."
        )

    asset_lines: list[str] = []
    for ref_info in ev.artifact_refs:
        ref = ref_info.get("path") or ref_info.get("url") or ""
        kind = ref_info.get("artifact_type") or "artifact"
        if ref:
            asset_lines.append(f"  - {kind}: {ref}")
    produced_block = "\n".join(asset_lines) if asset_lines else "  (none)"
    return (
        f"[system] Background task {ev.job_id} completed.{output}\n\n"
        f"Assets produced by THIS task:\n{produced_block}\n\n"
        "Use ONLY these specific URLs/paths when presenting to the user. "
        "Do NOT reference older session assets and do NOT regenerate. "
        "The list above is authoritative for what "
        f"{ev.job_id} just produced."
    )


def tool_notice(ev: JobFinished) -> str:
    """The message an agent receives when one of its tool calls that was
    moved to the background finishes. Carries the tool's result, which
    never reached the model as a tool return."""
    if ev.error:
        return f"[system] Background tool call {ev.job_id} failed:\n{ev.error}"
    return f"[system] Background tool call {ev.job_id} finished. Its result:\n{ev.output}"


def agent_notice(ev: JobFinished) -> str:
    """The message a parent agent receives when an agent it launched
    settles. Shared by every consumer that turns the event into a parent
    turn, so the parent reads the same text whichever one delivered it."""
    if ev.error:
        return f"[system] Background agent {ev.job_id} failed before it finished:\n{ev.error}"

    asset_lines: list[str] = []
    for ref_info in ev.artifact_refs:
        ref = ref_info.get("path") or ref_info.get("url") or ""
        kind = ref_info.get("artifact_type") or "artifact"
        if ref:
            asset_lines.append(f"  - {kind}: {ref}")
    produced_block = "\n".join(asset_lines) if asset_lines else "  (none)"

    if ev.output:
        closing = f"Its closing message:\n{ev.output}"
    else:
        closing = "It left no closing message."
    return (
        f"[system] Background agent {ev.job_id} finished and will do no further work.\n\n"
        f"{closing}\n\n"
        f"Assets it produced:\n{produced_block}"
    )
