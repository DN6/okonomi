"""Context window management for Okonomi.

Provides automatic context compression when conversations approach the
model's context limit, and prompt caching for Anthropic models via OpenRouter.

Adapted from Hermes Agent's context_compressor.py and prompt_caching.py
to work with pydantic-ai's ModelMessage types.
"""

import logging
import re
from typing import Optional

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from .config import OkonomiConfig
from .utils import create_chat_client

logger = logging.getLogger(__name__)

SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION] Earlier turns in this conversation were compacted "
    "to save context space. The summary below describes work that was "
    "already completed, and the current session state may still reflect "
    "that work (for example, files may already be changed). Use the summary "
    "and the current state to continue from where things left off, and "
    "avoid repeating work:"
)

_CHARS_PER_TOKEN = 4
_MIN_SUMMARY_TOKENS = 2000
_MAX_SUMMARY_TOKENS = 8000
_SUMMARY_RATIO = 0.20
_DEFAULT_TAIL_TOKEN_BUDGET = 20_000
_PRUNED_PLACEHOLDER = "[Old tool output cleared to save context space]"

_BASE64_PATTERN = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]{100,}")
_HTML_PREFIX_PATTERN = re.compile(r"^\s*(<(!DOCTYPE|html)\b)", re.IGNORECASE)


def _strip_base64(text: str) -> str:
    """Replace base64 data URIs with a short placeholder."""
    def _replace(m):
        size_kb = len(m.group(0)) * 3 // 4 // 1024
        return f"[embedded image, ~{size_kb}KB]"
    return _BASE64_PATTERN.sub(_replace, text)


def _strip_large_content(text: str) -> str:
    """Strip base64 and large HTML from a string."""
    text = _strip_base64(text)
    if len(text) > 500 and _HTML_PREFIX_PATTERN.match(text):
        return f"[HTML content, {len(text)} chars — saved to file]"
    return text


def estimate_tokens_rough(messages: list[ModelMessage]) -> int:
    """Rough token estimate for a pydantic-ai message list.

    Strips base64 data URIs before counting to avoid massively inflated estimates.
    """
    total_chars = 0
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if hasattr(part, "content"):
                    total_chars += len(_strip_base64(str(part.content)))
                if isinstance(part, ToolReturnPart):
                    total_chars += len(part.tool_name) + 10
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    total_chars += len(_strip_base64(part.content))
                elif isinstance(part, ToolCallPart):
                    total_chars += len(part.tool_name) + len(_strip_base64(str(part.args))) + 10
    return total_chars // _CHARS_PER_TOKEN


def _message_token_estimate(msg: ModelMessage) -> int:
    """Rough token estimate for a single message."""
    chars = 0
    if isinstance(msg, ModelRequest):
        for part in msg.parts:
            if hasattr(part, "content"):
                chars += len(str(part.content))
    elif isinstance(msg, ModelResponse):
        for part in msg.parts:
            if isinstance(part, TextPart):
                chars += len(part.content)
            elif isinstance(part, ToolCallPart):
                chars += len(part.tool_name) + len(str(part.args))
    return chars // _CHARS_PER_TOKEN + 10


class ContextManager:
    """Compresses conversation context when approaching the model's context limit.

    Algorithm:
      1. Prune old tool results (cheap, no LLM call)
      2. Protect head messages (first exchange)
      3. Protect tail messages by token budget (most recent ~20K tokens)
      4. Summarize middle turns with structured LLM prompt
      5. On subsequent compactions, iteratively update the previous summary
    """

    def __init__(
        self,
        config: OkonomiConfig,
        protect_first_n: int = 3,
        protect_last_n: int = 4,
    ):
        self.config = config
        self.context_length = config.context.context_length
        self.threshold_percent = config.context.compression_threshold
        self.threshold_tokens = int(self.context_length * self.threshold_percent)
        self.prune_threshold = getattr(config.context, "prune_threshold", 0.65)
        self.prune_threshold_tokens = int(self.context_length * self.prune_threshold)
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.compression_count = 0

        self._client = create_chat_client()
        self._previous_summary: Optional[str] = None

        self.last_prompt_tokens = 0

        logger.info(
            "Context manager: context_length=%d threshold=%d (%.0f%%)",
            self.context_length,
            self.threshold_tokens,
            self.threshold_percent * 100,
        )

    def update_token_usage(self, prompt_tokens: int) -> None:
        self.last_prompt_tokens = prompt_tokens

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return tokens >= self.threshold_tokens

    def should_compress_preflight(self, messages: list[ModelMessage]) -> bool:
        return estimate_tokens_rough(messages) >= self.threshold_tokens

    def prune_if_needed(self, messages: list[ModelMessage]) -> tuple[list[ModelMessage], bool]:
        """Lightweight prune pass at the lower threshold (no LLM summary).

        Strips base64, large HTML, and old tool results. Runs at 65% of context.
        """
        est = estimate_tokens_rough(messages)
        if est < self.prune_threshold_tokens:
            return messages, False

        messages, pruned = self._prune_old_tool_results(messages, self.protect_last_n * 3)
        if pruned:
            logger.info("Lightweight prune: stripped %d items at %d tokens (threshold %d)",
                        pruned, est, self.prune_threshold_tokens)
        return messages, pruned > 0

    # ------------------------------------------------------------------
    # Tool output pruning
    # ------------------------------------------------------------------

    def _prune_old_tool_results(
        self,
        messages: list[ModelMessage],
        protect_tail_count: int,
    ) -> tuple[list[ModelMessage], int]:
        """Replace old tool result contents with a short placeholder.

        Also strips base64 and large HTML from ALL messages (including tail).
        """
        if not messages:
            return messages, 0

        pruned = 0
        prune_boundary = len(messages) - protect_tail_count
        result = list(messages)

        for i in range(len(result)):
            msg = result[i]
            if not isinstance(msg, ModelRequest):
                continue
            new_parts = []
            changed = False
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    content = str(part.content)
                    stripped = _strip_large_content(content)
                    if stripped != content:
                        new_parts.append(ToolReturnPart(
                            tool_name=part.tool_name,
                            content=stripped,
                            tool_call_id=part.tool_call_id,
                            timestamp=part.timestamp,
                        ))
                        pruned += 1
                        changed = True
                        continue
                    if i < prune_boundary and len(content) > 100 and content != _PRUNED_PLACEHOLDER:
                        new_parts.append(ToolReturnPart(
                            tool_name=part.tool_name,
                            content=_PRUNED_PLACEHOLDER,
                            tool_call_id=part.tool_call_id,
                            timestamp=part.timestamp,
                        ))
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
                result[i] = ModelRequest(parts=new_parts)

        return result, pruned

    # ------------------------------------------------------------------
    # Serialization for summarizer
    # ------------------------------------------------------------------

    def _serialize_for_summary(self, messages: list[ModelMessage]) -> str:
        parts = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, (TextPart, UserPromptPart)):
                        content = str(part.content)
                        if len(content) > 3000:
                            content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
                        parts.append(f"[USER]: {content}")
                    elif isinstance(part, ToolReturnPart):
                        content = str(part.content)
                        if len(content) > 3000:
                            content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
                        parts.append(f"[TOOL RESULT {part.tool_name}]: {content}")
            elif isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        content = part.content
                        if len(content) > 3000:
                            content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
                        parts.append(f"[ASSISTANT]: {content}")
                    elif isinstance(part, ToolCallPart):
                        args = str(part.args)
                        if len(args) > 500:
                            args = args[:400] + "..."
                        parts.append(f"[ASSISTANT TOOL CALL]: {part.tool_name}({args})")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------

    def _compute_summary_budget(self, messages: list[ModelMessage]) -> int:
        content_tokens = estimate_tokens_rough(messages)
        budget = int(content_tokens * _SUMMARY_RATIO)
        return max(_MIN_SUMMARY_TOKENS, min(budget, _MAX_SUMMARY_TOKENS))

    def _chat_completion(self, prompt: str, max_tokens: int = 4096) -> str:
        model_id = self.config.agent.model_id
        response = self._client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _generate_summary(self, messages: list[ModelMessage]) -> Optional[str]:
        summary_budget = self._compute_summary_budget(messages)
        content = self._serialize_for_summary(messages)

        if self._previous_summary:
            prompt = f"""You are updating a context compaction summary. A previous compaction produced the summary below. New conversation turns have occurred since then and need to be incorporated.

PREVIOUS SUMMARY:
{self._previous_summary}

NEW TURNS TO INCORPORATE:
{content}

Update the summary using this exact structure. PRESERVE all existing information that is still relevant. ADD new progress.

## Goal
[What the user is trying to accomplish]

## Progress
### Done
[Completed work — include specific file paths, commands run, results obtained]
### In Progress
[Work currently underway]

## Key Decisions
[Important technical decisions and why they were made]

## Relevant Files
[Files read, modified, or created — with brief note on each]

## Next Steps
[What needs to happen next]

## Critical Context
[Specific values, error messages, or data that would be lost without preservation]

Target ~{summary_budget} tokens. Be specific — include file paths, outputs, and concrete values.

Write only the summary body."""
        else:
            prompt = f"""Create a structured handoff summary for a later assistant that will continue this conversation after earlier turns are compacted.

TURNS TO SUMMARIZE:
{content}

Use this exact structure:

## Goal
[What the user is trying to accomplish]

## Progress
### Done
[Completed work — include specific file paths, commands run, results obtained]
### In Progress
[Work currently underway]

## Key Decisions
[Important technical decisions and why they were made]

## Relevant Files
[Files read, modified, or created — with brief note on each]

## Next Steps
[What needs to happen next]

## Critical Context
[Specific values, error messages, or data that would be lost without preservation]

Target ~{summary_budget} tokens. Be specific — include file paths, outputs, and concrete values.

Write only the summary body."""

        try:
            summary = self._chat_completion(prompt, max_tokens=summary_budget * 2).strip()
            self._previous_summary = summary
            return f"{SUMMARY_PREFIX}\n{summary}"
        except Exception as e:
            logger.warning("Failed to generate context summary: %s", e)
            return None

    # ------------------------------------------------------------------
    # Tail protection by token budget
    # ------------------------------------------------------------------

    def _find_tail_cut(
        self,
        messages: list[ModelMessage],
        head_end: int,
        token_budget: int = _DEFAULT_TAIL_TOKEN_BUDGET,
    ) -> int:
        n = len(messages)
        accumulated = 0
        cut_idx = n

        for i in range(n - 1, head_end - 1, -1):
            msg_tokens = _message_token_estimate(messages[i])
            if accumulated + msg_tokens > token_budget and (n - i) >= self.protect_last_n:
                break
            accumulated += msg_tokens
            cut_idx = i

        fallback_cut = n - self.protect_last_n
        if cut_idx > fallback_cut:
            cut_idx = fallback_cut
        if cut_idx <= head_end:
            cut_idx = fallback_cut

        return max(cut_idx, head_end + 1)

    # ------------------------------------------------------------------
    # Main compression
    # ------------------------------------------------------------------

    def compress(
        self,
        messages: list[ModelMessage],
        current_tokens: int | None = None,
    ) -> tuple[list[ModelMessage], bool]:
        """Compress conversation messages by summarizing middle turns.

        Returns (compressed_messages, was_compressed).
        """
        n = len(messages)
        if n <= self.protect_first_n + self.protect_last_n + 1:
            logger.warning("Cannot compress: only %d messages", n)
            return messages, False

        display_tokens = current_tokens or self.last_prompt_tokens or estimate_tokens_rough(messages)

        messages, pruned = self._prune_old_tool_results(
            messages,
            protect_tail_count=self.protect_last_n * 3,
        )
        if pruned:
            logger.info("Pre-compression: pruned %d old tool result(s)", pruned)

        compress_start = self.protect_first_n
        compress_end = self._find_tail_cut(messages, compress_start)

        if compress_start >= compress_end:
            return messages, False

        turns_to_summarize = messages[compress_start:compress_end]
        logger.info(
            "Compressing: %d tokens >= %d threshold, summarizing turns %d-%d (%d turns)",
            display_tokens,
            self.threshold_tokens,
            compress_start + 1,
            compress_end,
            len(turns_to_summarize),
        )

        summary = self._generate_summary(turns_to_summarize)
        compressed = list(messages[:compress_start])

        if summary:
            compressed.append(ModelRequest(parts=[UserPromptPart(content=summary)]))
        else:
            logger.warning("No summary generated — middle turns dropped without summary")

        compressed.extend(messages[compress_end:])
        self.compression_count += 1

        new_estimate = estimate_tokens_rough(compressed)
        logger.info(
            "Compressed: %d -> %d messages (~%d tokens saved)",
            n,
            len(compressed),
            display_tokens - new_estimate,
        )

        if self.compression_count >= 3:
            print(
                "\033[33mSession context compressed "
                f"{self.compression_count} times. "
                "Consider /clear for a fresh start or /save to preserve this work.\033[0m"
            )

        return compressed, True

    def build_resume_context(
        self,
        session_id: str,
        task: str,
        steps_summary: str,
        generated_outputs: list | None = None,
        artifacts: list | None = None,
    ) -> list[ModelMessage]:
        """Build a lightweight message history for a resumed session.

        Instead of restoring the full conversation, injects a single
        summary message with outputs and artifacts inline.
        """
        parts = [f"[Resumed session {session_id[:8]}]"]
        parts.append(f"\nPrevious task: {task}")
        if steps_summary:
            parts.append(f"\n## Summary of previous work\n{steps_summary}")
        if generated_outputs:
            parts.append("\n## Generated outputs")
            for i, o in enumerate(generated_outputs, 1):
                line = f"{i}. {o.asset_type}: {o.asset_url}"
                if o.prompt:
                    line += f"\n   Prompt: {o.prompt}"
                if o.tool_name:
                    line += f"\n   Tool: {o.tool_name}"
                parts.append(line)
        if artifacts:
            non_output_artifacts = [a for a in artifacts if a.artifact_type not in ("image", "video")]
            if non_output_artifacts:
                parts.append("\n## Artifacts")
                for a in non_output_artifacts:
                    line = f"- [{a.artifact_type}] {a.path}"
                    if a.description:
                        line += f": {a.description}"
                    parts.append(line)

        return [ModelRequest(parts=[UserPromptPart(content="\n".join(parts))])]

    def generate_steps_summary(self, messages: list[ModelMessage]) -> str:
        """Generate LLM summary of session actions."""
        if not messages:
            return ""

        try:
            content = self._serialize_for_summary(messages)
            prompt = f"""Summarize the steps taken in this image/video generation session.
Focus on:
1. Tools used and their sequence
2. Key prompts and parameters
3. Any iterations or refinements
4. Final outcomes

Conversation:
{content}

Provide a concise summary (2-4 sentences):"""
            return self._chat_completion(prompt, max_tokens=256).strip()
        except Exception:
            return ""
