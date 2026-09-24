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
        blocks and term commands both route here)."""
        refs = [{"artifact_type": a.artifact_type, "path": a.path, "url": a.url} for a in produced_artifacts or []]
        self._emitter.emit(
            JobFinished(
                job_id=bg_id,
                status="error" if error else "ok",
                error=error,
                output=output,
                new_artifacts=new_artifacts,
                artifact_refs=refs,
            )
        )
