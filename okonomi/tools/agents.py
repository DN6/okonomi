"""Background agent tool for launching sub-agents.

Allows the main agent to delegate tasks to background agents that run
concurrently. Each sub-agent gets its own conversation context but shares
the session directory for outputs.
"""

import asyncio
import threading
from dataclasses import dataclass
from datetime import datetime

from ..events import SystemNote


@dataclass
class BackgroundTask:
    task_id: str
    description: str
    status: str = "running"
    started_at: str = ""
    outputs: int = 0
    error: str = ""


class AgentTools:
    """Tool for launching and managing background sub-agents."""

    def __init__(self):
        self._tasks: dict[str, BackgroundTask] = {}
        self._counter = 0
        self._parent_agent = None

    def _bind(self, parent_agent) -> None:
        """Bind to the parent agent (called during agent setup)."""
        self._parent_agent = parent_agent

    def launch(self, task: str, description: str = "") -> str:
        """Launch a background sub-agent to work on a task concurrently.

        The sub-agent runs independently with its own conversation context
        but shares the session. Generated outputs are merged back when done.

        Use this to parallelize independent work — e.g., generating multiple
        images or variations simultaneously.

        Args:
            task: The task/prompt for the sub-agent to execute.
            description: Short label for tracking (defaults to first 40 chars of task).

        Returns:
            Status message with the task ID for tracking.
        """
        if not self._parent_agent:
            return "Error: agent tools not initialized."

        self._counter += 1
        task_id = f"agent-{self._counter}"
        label = description or task[:40]

        bg_task = BackgroundTask(
            task_id=task_id,
            description=label,
            started_at=datetime.now().strftime("%H:%M:%S"),
        )
        self._tasks[task_id] = bg_task

        parent = self._parent_agent

        approvals_enabled = parent.approvals_enabled

        # Register in the parent agent's shared bg-log registry so /jobs
        # and /logs see launched agents alongside bg-exec-N and term-N.
        # The reader snapshots the BackgroundTask record (status, error,
        # artifact count) — full transcript capture would require routing
        # the bg agent's emitter through here, which the launch site
        # currently doesn't have set up. Status summary is enough to
        # answer "is it still running, did it crash, how much did it
        # produce" — the bulk of /logs queries.
        def _reader(bg=bg_task) -> str:
            parts = [
                f"task: {bg.description}",
                f"started: {bg.started_at}",
                f"status: {bg.status}",
            ]
            if bg.outputs:
                parts.append(f"artifacts produced: {bg.outputs}")
            if bg.error:
                parts.append(f"error: {bg.error}")
            return "\n".join(parts)

        if hasattr(parent, "_register_bg_log"):
            try:
                parent._register_bg_log(task_id=task_id, label=label, reader=_reader)
            except Exception:
                pass

        def _run():
            from okonomi.agent import OkonomiAgent

            final_status = "ok"
            try:
                agent = OkonomiAgent(
                    model_id=parent.model_id,
                    timeout=parent.timeout,
                    compress_at_tokens=parent.compress_at_tokens,
                    session_id=parent.session.session_id,
                )
                # Sub-agent approval requests forward to the parent's
                # emitter — the event carries its own resolve handle, so
                # whoever answers there resolves the sub-agent's wait.
                if approvals_enabled:
                    agent.approvals_enabled = True
                    from okonomi.events import ApprovalRequest

                    agent.emitter.subscribe(
                        lambda ev: parent.emitter.emit(ev) if isinstance(ev, ApprovalRequest) else None
                    )

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(agent.run_async(task, reset=True))
                    parent.state.artifacts.extend(agent.state.artifacts)
                    bg_task.outputs = len(agent.state.artifacts)
                    bg_task.status = "done"
                    parent.emitter.emit(SystemNote(
                        f"✓ Agent done: {label} — {bg_task.outputs} artifact{'s' if bg_task.outputs != 1 else ''}"
                    ))
                except Exception as e:
                    bg_task.status = "failed"
                    bg_task.error = str(e)
                    final_status = "error"
                    parent.emitter.emit(SystemNote(f"✗ Agent failed: {label} — {e}", level="error"))
                finally:
                    from okonomi.cli import _shutdown_loop

                    _shutdown_loop(loop)
            except Exception as e:
                bg_task.status = "failed"
                bg_task.error = str(e)
                final_status = "error"
                parent.emitter.emit(SystemNote(f"✗ Agent error: {label} — {e}", level="error"))
            finally:
                if hasattr(parent, "_mark_bg_log_done"):
                    try:
                        parent._mark_bg_log_done(task_id=task_id, status=final_status)
                    except Exception:
                        pass

        t = threading.Thread(target=_run, daemon=True, name=f"okonomi-{task_id}")
        t.start()

        parent.emitter.emit(SystemNote(f"Launched background agent: {label} [{task_id}]"))
        return f"Background agent launched: {task_id} ({label}). It will run concurrently — check status with list_agents()."

    def list_agents(self) -> str:
        """List all background agents and their current status.

        Returns:
            Summary of all launched agents with their status.
        """
        if not self._tasks:
            return "No background agents have been launched."

        lines = []
        for task_id, t in self._tasks.items():
            status = t.status
            if status == "done":
                status = f"done ({t.outputs} outputs)"
            elif status == "failed":
                status = f"failed: {t.error}"
            lines.append(f"- {task_id}: {t.description} [{status}] (started {t.started_at})")

        return "Background agents:\n" + "\n".join(lines)
