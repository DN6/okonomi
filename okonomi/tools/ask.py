"""``ask_user``: a question the model puts to the user mid-turn.

The agent injects ``ask_fn`` (``OkonomiAgent._ask_user``), which emits an
``AskUserRequest`` and blocks until the TUI answers, the same way dangerous
commands wait on an ``ApprovalRequest``.
"""

from typing import Callable


class AskTools:
    def __init__(self, ask_fn: Callable[[str, list[str]], dict]):
        self._ask_fn = ask_fn

    def ask_user(self, question: str, options: list[str] | None = None) -> dict:
        """Ask the user a question and wait for the answer.

        Use it before work whose outcome depends on taste or intent you don't
        have: which direction to pursue, which reference to follow, what the
        piece is for, which of several candidates to develop. Give ``options``
        whenever you can propose concrete choices (2-4, each a short phrase);
        the user can still answer in their own words. Ask one question per
        call. Don't ask what you can infer from the conversation or recall.

        Returns ``{"answer": str, "option": int | None}`` — ``option`` is the
        1-based index when the answer matched one of ``options``. If the user
        dismisses the question or no one can answer in this run, you get an
        ``error``: proceed on your best assumption and say what you assumed.
        """
        return self._ask_fn(question, options or [])
