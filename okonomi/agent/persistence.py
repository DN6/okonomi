"""Session persistence: save/load state, pickle user globals.

``SessionStore`` owns nothing agent-shaped — it takes the managers it
needs at construction and the live state per call, so it can be used and
tested without an OkonomiAgent.
"""

import pickle
import types
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from ..context import ContextManager
from ..session import OkonomiState, SessionManager, SessionStatus


def strip_unanswered_tool_calls(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Remove trailing messages with unprocessed tool calls.

    After cancellation the history may end with a ModelResponse containing
    ToolCallParts that were never answered with ToolReturnParts.  Passing
    such history back to the model triggers:
      "Cannot provide a new user prompt when the message history
       contains unprocessed tool calls."

    We walk backwards and strip any trailing ModelResponse that has tool
    calls, plus any ModelRequest that only contains tool returns (since
    those are orphaned without their preceding response).
    """
    trimmed = list(messages)
    while trimmed:
        last = trimmed[-1]
        if isinstance(last, ModelResponse):
            has_tool_calls = any(isinstance(p, ToolCallPart) for p in last.parts)
            if has_tool_calls:
                trimmed.pop()
                continue
        if isinstance(last, ModelRequest):
            only_tool_returns = all(isinstance(p, ToolReturnPart) for p in last.parts)
            if only_tool_returns:
                trimmed.pop()
                continue
        break
    return trimmed


class SessionStore:
    """Persists and restores ``OkonomiState``.

    ``tool_namespaces`` are the execute_code namespace keys (browser,
    genmedia, ...) — skipped when pickling user globals because they're
    reinstalled fresh on every execute_code call.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        context_manager: ContextManager,
        *,
        tool_namespaces: set[str],
    ) -> None:
        self.session_manager = session_manager
        self.context_manager = context_manager
        self._tool_namespaces = set(tool_namespaces)

    def save(
        self,
        state: OkonomiState,
        *,
        model_id: str,
        status: SessionStatus = "completed",
        download_assets: bool = True,
        skip_summary: bool = False,
    ) -> None:
        """Save ``state``.

        Args:
            state: The live agent state to persist.
            model_id: Recorded on the session metadata.
            status: Status of the session (completed, error, abandoned)
            download_assets: When True (default), synchronously download
                any tracked outputs that don't yet have a ``local_path``.
                Set False on shutdown / atexit paths so quitting doesn't
                wait on slow network downloads — the eager bg pool has
                been pulling these all along, and any still-missing files
                will be retried on the next session resume.
            skip_summary: Skip the LLM steps-summary call. The summary
                hits OpenRouter and can take 10–60s on long histories;
                shutdown paths set this True so exit returns promptly.
                The previous per-turn save already wrote a fresh summary.
        """
        session_dir = self.session_manager.session_dir(state.info.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        if download_assets:
            # Local import — agent.py imports this module at its top level,
            # so importing back at module scope would be circular.
            from .agent import download_session_assets

            download_session_assets(state.artifacts, session_dir)

        # Stamp the run-owned fields onto the live info; SessionManager
        # carries forward identity (title / dir_name / created_at) and
        # persists everything else as-is. skip_summary leaves the prior
        # turn's summary in place (a completed save already wrote one).
        state.info.status = status
        state.info.model = model_id
        if not skip_summary:
            state.info.steps_summary = self.context_manager.generate_steps_summary(state.message_history)

        # Titling lives outside the agent: SessionManager.save_session fires a
        # small-model titler in the background. The agent never generates titles.
        self.session_manager.save_session(state)
        self._serialize_user_globals(state)

    def _serialize_user_globals(self, state: OkonomiState) -> None:
        """Save picklable user-bound variables from ``execution_globals`` to
        ``<session_dir>/globals.pkl``. Tool wrappers, modules, functions,
        and anything that fails to pickle are silently skipped — the goal
        is to recover the most-used patterns (URLs, paths, lists of
        descriptions, intermediate dicts), not arbitrary live objects.
        """
        if not state.execution_globals:
            return

        SKIP_KEYS = {"__builtins__", "asyncio", "get_session_dir"}
        SKIP_KEYS |= self._tool_namespaces  # tool wrappers

        candidates: dict[str, Any] = {}
        for key, value in state.execution_globals.items():
            if key in SKIP_KEYS or key.startswith("__"):
                continue
            if isinstance(value, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType)):
                continue
            # ToolWrappers and Tool instances both carry _tool_instance
            # or _set_deps — drop anything that smells tool-ish.
            if hasattr(value, "_tool_instance") or hasattr(value, "_set_deps"):
                continue
            candidates[key] = value

        if not candidates:
            return

        try:
            session_dir = self.session_manager.session_dir(state.info.session_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            path = session_dir / "globals.pkl"
            try:
                with open(path, "wb") as f:
                    pickle.dump(candidates, f, protocol=pickle.HIGHEST_PROTOCOL)
                return
            except Exception:
                pass
            # One of the values is unpicklable — fall back to per-key.
            clean: dict[str, Any] = {}
            for key, value in candidates.items():
                try:
                    pickle.dumps(value)
                except Exception:
                    continue
                clean[key] = value
            with open(path, "wb") as f:
                pickle.dump(clean, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    def _restore_user_globals(self, state: OkonomiState) -> None:
        """Counterpart to ``_serialize_user_globals``. Reads
        ``<session_dir>/globals.pkl`` if present and merges into
        ``state.execution_globals``. Tool wrappers + helpers will be
        reinstalled on the next ``execute_code`` call."""
        try:
            session_dir = self.session_manager.session_dir(state.info.session_id)
            globals_path = session_dir / "globals.pkl"
            if not globals_path.exists():
                return
            with open(globals_path, "rb") as f:
                snapshot = pickle.load(f)
            if isinstance(snapshot, dict):
                state.execution_globals.update(snapshot)
        except Exception:
            pass

    def load(self, session_id: str) -> OkonomiState | None:
        """Load a saved session into a fully-prepared ``OkonomiState``.

        Restores the full message history when available so the model
        sees its prior tool calls, intermediate reasoning, and produced
        artifacts directly. The next-turn ``run_async`` already runs
        prune + compress on the restored history, so oversized sessions
        get trimmed there instead of being stripped here.

        For legacy sessions without ``messages.jsonl`` (older saves), we
        fall back to a synthesized summary message — better than nothing
        but the agent will know much less about what happened.

        Returns None if the session doesn't exist. The caller installs
        the returned state on the agent.
        """
        state = self.session_manager.load_session(session_id)

        if state is None:
            return None

        # Raw log restored as-is (append-only audit record). Working set
        # is a sanitized view — strip orphan tool calls at the tail from
        # an interrupted save. Compression on the working set fires on
        # the next turn if oversized.
        if state.message_history:
            context = strip_unanswered_tool_calls(list(state.message_history))
            if not context:
                # Sanitization wiped everything (rare). Fall back to summary.
                context = self.context_manager.build_resume_context(
                    session_id=session_id,
                    task=state.info.task_input,
                    steps_summary=state.info.steps_summary,
                    artifacts=state.artifacts,
                )
        else:
            # Legacy session — no jsonl. Synthesize a summary message.
            context = self.context_manager.build_resume_context(
                session_id=session_id,
                task=state.info.task_input,
                steps_summary=state.info.steps_summary,
                artifacts=state.artifacts,
            )
            state.message_history = list(context)

        state.context = context

        # Restore Python locals the model bound during the prior session.
        # Tool wrappers / helpers are re-injected on the next execute_code
        # call, so we only need user-bound vars here.
        self._restore_user_globals(state)

        return state
