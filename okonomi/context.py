"""Context window management for Okonomi.

Provides automatic context compression when conversations approach the
model's context limit, and prompt caching for Anthropic models via OpenRouter.

Adapted from Hermes Agent's context_compressor.py and prompt_caching.py
to work with pydantic-ai's ModelMessage types.
"""

import logging
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


def estimate_tokens_rough(messages: list[ModelMessage]) -> int:
    """Rough token estimate for a pydantic-ai message list."""
    total_chars = 0
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if hasattr(part, "content"):
                    total_chars += len(str(part.content))
                if isinstance(part, ToolReturnPart):
                    total_chars += len(part.tool_name) + 10
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    total_chars += len(part.content)
                elif isinstance(part, ToolCallPart):
                    total_chars += len(part.tool_name) + len(str(part.args)) + 10
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

    # ------------------------------------------------------------------
    # Tool output pruning
    # ------------------------------------------------------------------

    def _prune_old_tool_results(
        self,
        messages: list[ModelMessage],
        protect_tail_count: int,
    ) -> tuple[list[ModelMessage], int]:
        """Replace old tool result contents with a short placeholder."""
        if not messages:
            return messages, 0

        pruned = 0
        prune_boundary = len(messages) - protect_tail_count
        result = list(messages)

        for i in range(max(prune_boundary, 0)):
            msg = result[i]
            if not isinstance(msg, ModelRequest):
                continue
            new_parts = []
            changed = False
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    content = str(part.content)
                    if len(content) > 200 and content != _PRUNED_PLACEHOLDER:
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

        return compressed, True

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
