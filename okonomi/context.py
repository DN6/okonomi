"""Context window management for Okonomi.

Compresses message history when it grows past a token threshold by
summarizing the middle turns with an LLM and keeping the head and tail
intact.
"""

import logging
import re

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from .utils import create_chat_client


logger = logging.getLogger(__name__)

SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION] Earlier turns in this conversation were "
    "compacted to save context space. The summary below describes work "
    "already completed; the current session state may still reflect it "
    "(files may already be changed). Continue from where things left "
    "off — don't repeat work."
)

_CHARS_PER_TOKEN = 4
_SUMMARY_MAX_TOKENS = 4000
_TRUNCATE_AT_CHARS = 3000  # per-part cap in serialization for summarizer
_PRUNED_PLACEHOLDER = "[Old tool output cleared to save context space]"
_IMAGE_PART_TOKEN_ESTIMATE = 1500  # flat per-image charge; vision bills 1.5–3K

_BASE64_PATTERN = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]{100,}")
_HTML_PREFIX_PATTERN = re.compile(r"^\s*(<(!DOCTYPE|html)\b)", re.IGNORECASE)


def _strip_base64(text: str) -> str:
    def _replace(m):
        kb = len(m.group(0)) * 3 // 4 // 1024
        return f"[embedded image, ~{kb}KB]"

    return _BASE64_PATTERN.sub(_replace, text)


def _strip_large_content(text: str) -> str:
    text = _strip_base64(text)
    if len(text) > 500 and _HTML_PREFIX_PATTERN.match(text):
        return f"[HTML content, {len(text)} chars — saved to file]"
    return text


def _content_chars(content) -> int:
    """Char count for a part's content, multimodal-aware.

    pydantic-ai multimodal parts hold a list of [str, BinaryContent,
    ImageUrl, ...]. ``str(list)`` would render raw image bytes as
    escape-sequence text and bloat the count by megabytes per image —
    walk the list and charge a flat per-image token estimate instead.
    """
    if isinstance(content, str):
        return len(_strip_base64(content))
    if isinstance(content, (bytes, bytearray)):
        return _IMAGE_PART_TOKEN_ESTIMATE * _CHARS_PER_TOKEN
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, str):
                total += len(_strip_base64(item))
            elif hasattr(item, "data") or hasattr(item, "url"):
                total += _IMAGE_PART_TOKEN_ESTIMATE * _CHARS_PER_TOKEN
            else:
                total += len(_strip_base64(str(item)))
        return total
    return len(_strip_base64(str(content)))


def count_tokens(messages: list[ModelMessage]) -> int:
    """Rough token estimate for a pydantic-ai message list."""
    total = 0
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if hasattr(part, "content"):
                    total += _content_chars(part.content)
                if isinstance(part, ToolReturnPart):
                    total += len(part.tool_name) + 10
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    total += len(_strip_base64(part.content))
                elif isinstance(part, ToolCallPart):
                    total += len(part.tool_name) + len(_strip_base64(str(part.args))) + 10
    return total // _CHARS_PER_TOKEN


def _truncate(text: str, limit: int = _TRUNCATE_AT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    return text[:head] + "\n...[truncated]...\n" + text[-(limit - head) :]


def _serialize_for_summary(messages: list[ModelMessage]) -> str:
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, (TextPart, UserPromptPart)):
                    lines.append(f"[USER]: {_truncate(str(part.content))}")
                elif isinstance(part, ToolReturnPart):
                    lines.append(f"[TOOL RESULT {part.tool_name}]: {_truncate(str(part.content))}")
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    lines.append(f"[ASSISTANT]: {_truncate(part.content)}")
                elif isinstance(part, ToolCallPart):
                    lines.append(f"[ASSISTANT TOOL CALL]: {part.tool_name}({_truncate(str(part.args), 500)})")
    return "\n\n".join(lines)


class ContextManager:
    """Compresses conversation history when it exceeds ``compress_at_tokens``.

    Algorithm: prune old tool results → keep first ``protect_first_n`` +
    last ``protect_last_n`` messages → summarize the middle into a
    single synthetic user turn.
    """

    def __init__(
        self,
        *,
        model_id: str,
        compress_at_tokens: int = 170_000,
        protect_first_n: int = 3,
        protect_last_n: int = 8,
    ):
        self.model_id = model_id
        self.compress_at_tokens = compress_at_tokens
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.compression_count = 0
        self._client = create_chat_client()
        logger.info("Context manager: compress_at=%d", compress_at_tokens)

    def compress(self, messages: list[ModelMessage]) -> list[ModelMessage] | None:
        """Compress middle turns into a summary if over the threshold.

        Returns the compressed message list on success, or ``None`` if
        no compression happened (under threshold, too few messages,
        empty middle, or LLM call failed).
        """
        before = count_tokens(messages)
        if before < self.compress_at_tokens:
            return None

        n = len(messages)
        if n <= self.protect_first_n + self.protect_last_n + 1:
            return None

        messages, pruned = self._prune_old_tool_results(messages)
        if pruned:
            logger.info("Pre-compression: pruned %d old tool result(s)", pruned)

        head_end = self.protect_first_n
        tail_start = n - self.protect_last_n
        middle = messages[head_end:tail_start]
        if not middle:
            return None

        logger.info(
            "Compressing: %d tokens >= %d, summarizing %d middle turns",
            before,
            self.compress_at_tokens,
            len(middle),
        )

        summary = self._summarize(middle)
        if not summary:
            logger.warning("Summary generation failed; aborting compression")
            return None

        compressed = list(messages[:head_end])
        compressed.append(ModelRequest(parts=[UserPromptPart(content=summary)]))
        compressed.extend(messages[tail_start:])

        self.compression_count += 1
        after = count_tokens(compressed)
        logger.info("Compressed: %d → %d messages (~%d tokens saved)", n, len(compressed), before - after)
        if self.compression_count >= 3:
            print(
                f"\033[33mSession compressed {self.compression_count}× — "
                "consider /clear for a fresh start or /save to preserve this work.\033[0m"
            )
        return compressed

    def generate_steps_summary(self, messages: list[ModelMessage]) -> str:
        """Short LLM summary of session actions for saved metadata."""
        if not messages:
            return ""
        prompt = (
            "Summarize this image/video generation session in 2-4 sentences. "
            "Focus on tools used, key prompts, iterations, and final outcomes.\n\n"
            f"Conversation:\n{_serialize_for_summary(messages)}"
        )
        return self._chat(prompt, max_tokens=256)

    def build_resume_context(
        self,
        session_id: str,
        task: str,
        steps_summary: str,
        artifacts: list | None = None,
    ) -> list[ModelMessage]:
        """Synthesize a single resume message for sessions without messages.jsonl."""
        parts = [f"[Resumed session {session_id[:8]}]", f"\nPrevious task: {task}"]
        if steps_summary:
            parts.append(f"\n## Summary of previous work\n{steps_summary}")
        if artifacts:
            media_types = {"image", "video", "audio", "3d"}
            media = [a for a in artifacts if a.artifact_type in media_types]
            other = [a for a in artifacts if a.artifact_type not in media_types]
            if media:
                parts.append("\n## Generated media")
                for i, a in enumerate(media, 1):
                    line = f"{i}. {a.artifact_type}: {a.path or a.url}"
                    if a.prompt:
                        line += f"\n   Prompt: {a.prompt}"
                    if a.tool_name:
                        line += f"\n   Tool: {a.tool_name}"
                    parts.append(line)
            if other:
                parts.append("\n## Artifacts")
                for a in other:
                    line = f"- [{a.artifact_type}] {a.path or a.url}"
                    if a.description:
                        line += f": {a.description}"
                    parts.append(line)
        return [ModelRequest(parts=[UserPromptPart(content="\n".join(parts))])]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _chat(self, prompt: str, max_tokens: int) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return ""

    def _summarize(self, messages: list[ModelMessage]) -> str | None:
        prompt = (
            "Summarize this conversation segment as a handoff note for a later "
            "assistant. Preserve concrete details: file paths, tool calls and "
            "their results, decisions, error messages, what's done and what's "
            "next. Be specific, no fluff.\n\n"
            f"Conversation:\n{_serialize_for_summary(messages)}\n\n"
            "Write only the summary."
        )
        summary = self._chat(prompt, max_tokens=_SUMMARY_MAX_TOKENS)
        return f"{SUMMARY_PREFIX}\n{summary}" if summary else None

    def _prune_old_tool_results(self, messages: list[ModelMessage]) -> tuple[list[ModelMessage], int]:
        """Strip base64 / large HTML everywhere; replace old tool results with placeholder.

        "Old" = before the tail-protection boundary. Tail messages keep their
        full content (just minus base64). Returns (messages, pruned_count).
        """
        if not messages:
            return messages, 0
        prune_boundary = len(messages) - self.protect_last_n * 3
        out = list(messages)
        pruned = 0
        for i, msg in enumerate(out):
            if not isinstance(msg, ModelRequest):
                continue
            new_parts = []
            changed = False
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    content = str(part.content)
                    stripped = _strip_large_content(content)
                    if stripped != content:
                        new_parts.append(
                            ToolReturnPart(
                                tool_name=part.tool_name,
                                content=stripped,
                                tool_call_id=part.tool_call_id,
                                timestamp=part.timestamp,
                            )
                        )
                        pruned += 1
                        changed = True
                        continue
                    if i < prune_boundary and len(content) > 100 and content != _PRUNED_PLACEHOLDER:
                        new_parts.append(
                            ToolReturnPart(
                                tool_name=part.tool_name,
                                content=_PRUNED_PLACEHOLDER,
                                tool_call_id=part.tool_call_id,
                                timestamp=part.timestamp,
                            )
                        )
                        pruned += 1
                        changed = True
                        continue
                elif isinstance(part, (TextPart, UserPromptPart)):
                    content = str(part.content)
                    stripped = _strip_base64(content)
                    if stripped != content:
                        if isinstance(part, UserPromptPart):
                            new_parts.append(UserPromptPart(content=stripped, timestamp=part.timestamp))
                        else:
                            new_parts.append(TextPart(content=stripped))
                        changed = True
                        continue
                new_parts.append(part)
            if changed:
                out[i] = ModelRequest(parts=new_parts)
        return out, pruned
