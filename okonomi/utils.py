import os
import re


def create_chat_client():
    from openai import OpenAI

    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))


_TITLE_STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "it",
    "that",
    "this",
    "my",
    "me",
    "can",
    "you",
    "please",
    "i",
    "want",
    "would",
    "like",
    "make",
    "create",
    "generate",
    "do",
    "some",
    "using",
    "use",
    "let",
    "lets",
    "also",
}


def generate_title(task_input: str) -> str:
    """Extract a 3-word title from the first line of a user message."""
    text = task_input.strip()
    if not text:
        return ""
    text = text.split("\n")[0]
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    meaningful = [w for w in words if w not in _TITLE_STOP_WORDS and len(w) > 1]
    title_words = meaningful[:3] or words[:3]
    return " ".join(title_words) if title_words else ""


def generate_title_from_messages(messages: list, fallback: str = "") -> str:
    """Pick a title from the most substantive user message in the conversation.

    Scans all user-prompt parts in the message history, skips synthetic
    system-injected messages (resume markers, bg-exec completion notes,
    skill-creation nudges), and chooses the longest remaining user
    prompt by meaningful-word count. Falls back to ``generate_title(fallback)``
    if no usable candidate is found.
    """
    try:
        from pydantic_ai.messages import ModelRequest, UserPromptPart
    except Exception:
        return generate_title(fallback)

    candidates: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if not isinstance(part, UserPromptPart):
                continue
            content = part.content if isinstance(part.content, str) else ""
            text = content.strip()
            if not text:
                continue
            # Skip our own synthetic injections.
            if text.startswith(("[Resumed", "[system]", "[The user interrupted")):
                continue
            candidates.append(text)

    if not candidates:
        return generate_title(fallback)

    # Score by count of meaningful (non-stopword, length>1) tokens.
    def score(text: str) -> int:
        words = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return sum(1 for w in words if w not in _TITLE_STOP_WORDS and len(w) > 1)

    best = max(candidates, key=score)
    if score(best) == 0:
        return generate_title(fallback)
    return generate_title(best)
