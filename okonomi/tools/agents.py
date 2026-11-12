"""Background agent tool for launching sub-agents.

Allows the main agent to delegate tasks to background agents that run
concurrently. Each sub-agent gets its own conversation context but shares
the session directory for outputs.
"""

import asyncio
import threading
from dataclasses import dataclass
from datetime import datetime

from ..config import subagent_models
from ..events import SystemNote
from ..utils import shutdown_loop


@dataclass
class BackgroundTask:
    task_id: str
    description: str
    status: str = "running"
    started_at: str = ""
    outputs: int = 0
    error: str = ""
    output: str = ""


class AgentTools:
    """Tool for launching and managing background sub-agents."""

    def __init__(self):
        self._tasks: dict[str, BackgroundTask] = {}
        self._counter = 0
        self._parent_agent = None

    def _bind(self, parent_agent) -> None:
        """Bind to the parent agent (called during agent setup)."""
        self._parent_agent = parent_agent

    def launch(self, task: str, description: str = "", model: str = "") -> str:
        """Launch a background sub-agent to work on a task concurrently.

        The sub-agent runs independently with its own conversation context
        but shares the session. Generated outputs are merged back when done.
        Its closing message arrives in a system note when it finishes, so
        there is no need to poll for it.

        Use this to parallelize independent work — e.g., generating multiple
        images or variations simultaneously.

        Args:
            task: The task/prompt for the sub-agent to execute.
            description: Short label for tracking (defaults to first 40 chars of task).
            model: Model id for the sub-agent, to hand simple work to a cheaper
                model. Must be one of the sub-agent models listed in your
                instructions; leave empty to use your own model.

        Returns:
            Status message with the task ID for tracking.
        """
        if not self._parent_agent:
            return "Error: agent tools not initialized."

        model_id = self._parent_agent.model_id
        if model:
            allowed = subagent_models()
            if not allowed:
                return (
                    f"Error: model '{model}' is not available. No sub-agent models are configured "
                    "(OKONOMI_SUBAGENT_MODELS is unset), so sub-agents run on your own model — "
                    "call launch_agent again without `model`."
                )
            if model not in allowed:
                return (
                    f"Error: model '{model}' is not available. Choose one of: {', '.join(allowed)} — "
                    f"or omit `model` to use your own ({model_id})."
                )
            model_id = model

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

        # Register in the parent agent's shared bg-log registry so /tasks
        # sees launched agents alongside term-N shell tasks.
        # The reader snapshots the BackgroundTask record (status, error,
        # artifact count) — full transcript capture would require routing
        # the bg agent's emitter through here, which the launch site
        # currently doesn't have set up. Status summary is enough to
        # answer "is it still running, did it crash, how much did it
        # produce" — the bulk of /tasks lookups.
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
            if bg.output:
                parts.append(f"\n{bg.output}")
            return "\n".join(parts)

        if hasattr(parent, "_register_bg_log"):
            try:
                parent._register_bg_log(task_id=task_id, label=label, reader=_reader)
            except Exception:
                pass

        def _run():
            from okonomi.agent import OkonomiAgent
            from okonomi.agent.tasks import run_until_tasks_report

            final_status = "ok"
            reply = None
            produced = []
            try:
                agent = OkonomiAgent(
                    model_id=model_id,
                    timeout=parent.timeout,
                    compress_at_tokens=parent.compress_at_tokens,
                    session_id=parent.session.session_id,
                )
                parent.tasks.entries[task_id]["tasks"] = agent.tasks

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    reply = loop.run_until_complete(run_until_tasks_report(agent, task))
                    bg_task.output = reply or ""
                    produced = list(agent.state.artifacts)
                    parent.state.artifacts.extend(produced)
                    bg_task.outputs = len(produced)
                    if agent.last_turn_error:
                        bg_task.status = "failed"
                        bg_task.error = agent.last_turn_error
                        final_status = "error"
                        parent.emitter.emit(
                            SystemNote(f"✗ Agent failed: {label} — {agent.last_turn_error}", level="error")
                        )
                    else:
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
                    shutdown_loop(loop)
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
                parent._task_finished(
                    bg_id=task_id,
                    error=bg_task.error or None,
                    output=reply,
                    produced_artifacts=produced,
                )

        t = threading.Thread(target=_run, daemon=True, name=f"okonomi-{task_id}")
        t.start()

        parent.emitter.emit(SystemNote(f"Launched background agent: {label} [{task_id}]"))
        return (
            f"Background agent launched: {task_id} ({label}) on {model_id}. It runs concurrently; its "
            "closing message will arrive in a system note when it finishes."
        )

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
