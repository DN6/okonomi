"""Message memory management with truncation strategies."""

import os
from dataclasses import dataclass
from typing import Optional

from huggingface_hub import InferenceClient
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from .config import OkonomiConfig


@dataclass
class MemorySummary:
    """Summary of truncated message memory."""

    original_count: int
    retained_count: int
    truncated_count: int
    strategy: str
    llm_summary: Optional[str] = None


class MemoryManager:
    """Manages message memory with truncation to prevent context overflow."""

    def __init__(self, config: OkonomiConfig):
        self.memory_config = config.memory
        self.inference_config = config.inference_providers.huggingface
        self.client = InferenceClient(api_key=os.environ.get("HF_TOKEN"))

    def should_truncate(self, messages: list[ModelMessage]) -> bool:
        """Check if message memory should be truncated."""
        return len(messages) > self.memory_config.max_messages

    def truncate(
        self,
        messages: list[ModelMessage],
    ) -> tuple[list[ModelMessage], Optional[MemorySummary]]:
        """Truncate message memory based on configured strategy.

        If summarize=True, the truncated middle messages are summarized by LLM
        and the summary replaces them in the message history.

        Args:
            messages: List of messages to potentially truncate

        Returns:
            Tuple of (truncated messages, summary of truncation or None if no truncation)
        """
        if not self.should_truncate(messages):
            return messages, None

        strategy = self.memory_config.truncation_strategy

        # Apply truncation strategy
        if strategy == "keep_first_last":
            truncated, summary = self._keep_first_last_truncate(messages)
        else:
            truncated, summary = self._sliding_window_truncate(messages)

        # Optionally summarize the truncated messages
        if self.memory_config.summarize and summary:
            truncated, summary = self._apply_summarization(messages, truncated, summary)

        return truncated, summary

    def _sliding_window_truncate(
        self,
        messages: list[ModelMessage],
    ) -> tuple[list[ModelMessage], MemorySummary]:
        """Keep only the most recent messages (sliding window).

        Always preserves the first message (initial task context) and
        keeps the most recent messages up to max_messages.
        """
        max_messages = self.memory_config.max_messages
        original_count = len(messages)

        if original_count <= max_messages:
            return messages, None

        # Always keep the first message (usually the initial task)
        first_message = messages[0]

        # Keep the most recent messages (excluding the first)
        recent_count = max_messages - 1
        recent_messages = messages[-recent_count:]

        truncated = [first_message] + recent_messages

        summary = MemorySummary(
            original_count=original_count,
            retained_count=len(truncated),
            truncated_count=original_count - len(truncated),
            strategy="sliding_window",
        )

        return truncated, summary

    def _keep_first_last_truncate(
        self,
        messages: list[ModelMessage],
    ) -> tuple[list[ModelMessage], MemorySummary]:
        """Keep first few and last few messages, dropping middle.

        Useful for preserving initial context and recent state
        while dropping intermediate back-and-forth.
        """
        max_messages = self.memory_config.max_messages
        original_count = len(messages)

        if original_count <= max_messages:
            return messages, None

        # Split evenly between first and last
        keep_first = max_messages // 3
        keep_last = max_messages - keep_first

        first_messages = messages[:keep_first]
        last_messages = messages[-keep_last:]

        truncated = first_messages + last_messages

        summary = MemorySummary(
            original_count=original_count,
            retained_count=len(truncated),
            truncated_count=original_count - len(truncated),
            strategy="keep_first_last",
        )

        return truncated, summary

    def _message_to_text(self, message: ModelMessage) -> str:
        """Convert a ModelMessage to a readable text representation."""
        lines = []

        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, TextPart):
                    lines.append(f"User: {part.content}")
                elif isinstance(part, ToolReturnPart):
                    # Truncate long tool outputs
                    content = str(part.content)
                    if len(content) > 500:
                        content = content[:500] + "..."
                    lines.append(f"Tool Result ({part.tool_name}): {content}")

        elif isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, TextPart):
                    lines.append(f"Assistant: {part.content}")
                elif isinstance(part, ToolCallPart):
                    # Truncate long tool args
                    args = str(part.args)
                    if len(args) > 200:
                        args = args[:200] + "..."
                    lines.append(f"Tool Call ({part.tool_name}): {args}")

        return "\n".join(lines)

    def _summarize_messages(self, messages: list[ModelMessage]) -> str:
        """Use LLM to summarize a list of messages."""
        # Convert messages to readable text
        conversation_text = "\n\n".join(
            self._message_to_text(msg) for msg in messages
        )

        prompt = f"""Summarize this conversation excerpt concisely. Focus on:
1. Key user requests and goals
2. Important actions taken (images generated, edits made)
3. Results and outcomes (URLs, scores, feedback)
4. Any unresolved issues or next steps

Conversation:
{conversation_text}

Provide a brief summary (2-4 sentences) that captures the essential context:"""

        response = self.client.chat.completions.create(
            model=self.inference_config.prompt_model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256,
        )

        return response.choices[0].message.content.strip()

    def _apply_summarization(
        self,
        original_messages: list[ModelMessage],
        truncated_messages: list[ModelMessage],
        summary: MemorySummary,
    ) -> tuple[list[ModelMessage], MemorySummary]:
        """Summarize removed messages and inject as a system message.

        Takes the truncated result and:
        1. Identifies which messages were removed
        2. Summarizes them with LLM
        3. Injects summary as a ModelRequest with TextPart after the first message
        """
        # Find which messages were removed (the middle section)
        first_message = original_messages[0]
        retained_set = set(id(m) for m in truncated_messages)
        removed_messages = [m for m in original_messages if id(m) not in retained_set]

        if not removed_messages:
            return truncated_messages, summary

        try:
            llm_summary = self._summarize_messages(removed_messages)
            summary.llm_summary = llm_summary

            # Create a summary message to inject
            summary_text = f"[Previous conversation summary: {llm_summary}]"
            summary_message = ModelRequest(parts=[TextPart(content=summary_text)])

            # Insert summary after first message
            result = [first_message, summary_message] + truncated_messages[1:]

            # Update summary counts
            summary.retained_count = len(result)

            return result, summary

        except Exception:
            # If summarization fails, return original truncation
            return truncated_messages, summary

    def get_context_note(self, summary: MemorySummary) -> str:
        """Generate a note about truncated context for the agent.

        Args:
            summary: The truncation summary

        Returns:
            A note string to include in the conversation
        """
        note = (
            f"[Note: Conversation memory was truncated. "
            f"{summary.truncated_count} older messages were removed to stay within context limits. "
            f"Retained {summary.retained_count} messages using {summary.strategy} strategy."
        )

        if summary.llm_summary:
            note += f"\n\nSummary of truncated messages:\n{summary.llm_summary}"

        note += "]"
        return note
