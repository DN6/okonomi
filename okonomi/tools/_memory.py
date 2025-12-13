"""Memory tools using session-based storage with two-stage LLM retrieval."""

import json
import os

from huggingface_hub import InferenceClient

from okonomi.session import SessionManager
from okonomi.tools._metadata import ToolMetadataMixin


class MemoryTools(ToolMetadataMixin):
    __namespace__ = "memory"
    __description__ = "Tools for retrieving memory from past image generation sessions"

    def __init__(self, config):
        self.config = config
        self.session_manager = SessionManager()
        self.client = InferenceClient(api_key=os.environ.get("HF_TOKEN"))

    def _format_index_for_llm(self, index: list[dict], limit: int = 50) -> str:
        """Format the raw index for LLM consumption."""
        if not index:
            return "No previous sessions found."

        lines = ["# Session Index\n"]
        for i, entry in enumerate(index[:limit]):
            tools = ", ".join(entry.get("tools_used", [])) or "none"
            lines.append(
                f"{i+1}. [{entry['session_id'][:8]}] {entry.get('task_input', 'No task')} "
                f"(outputs: {entry.get('num_outputs', 0)}, tools: {tools}, status: {entry.get('status', 'unknown')})"
            )

        return "\n".join(lines)

    def _get_session_details(self, session_ids: list[str]) -> str:
        """Fetch and format full details for selected sessions."""
        lines = []

        for sid in session_ids:
            session = self.session_manager.load_session(sid)
            if not session:
                continue

            lines.append(f"## Session: {sid[:8]}")
            lines.append(f"Task: {session.metadata.task_input}")
            lines.append(f"Status: {session.metadata.status}")
            if session.metadata.tools_used:
                lines.append(f"Tools: {', '.join(session.metadata.tools_used)}")
            if session.metadata.models_used:
                lines.append(f"Models: {', '.join(session.metadata.models_used)}")
            if session.input_assets:
                lines.append(f"Input assets: {len(session.input_assets)}")
            if session.steps_summary:
                lines.append(f"Summary: {session.steps_summary}")
            lines.append("")

            for j, output in enumerate(session.generated_outputs):
                lines.append(f"### Output {j+1} ({output.asset_type})")
                if output.prompt:
                    lines.append(f"Prompt: {output.prompt}")
                lines.append(f"URL: {output.asset_url}")
                if output.model_id:
                    lines.append(f"Model: {output.model_id}")
                if output.loras:
                    lines.append(f"LoRAs: {output.loras}")
                lines.append("")

        return "\n".join(lines)

    async def retrieve_from_memory(self, query: str, top_k: int = 3) -> dict:
        """Retrieve relevant information from past sessions using two-stage LLM retrieval.

        Stage 1: Scan session index (metadata) to find top matching sessions
        Stage 2: Fetch full details for top matches and summarize

        Args:
            query: What you're looking for (e.g., "cyberpunk images",
                   "portraits", "prompts that worked well")
            top_k: Number of top sessions to retrieve details for (default: 3)

        Returns:
            Dict with relevant results from past sessions.
        """
        try:
            # Stage 1: Read index directly and format for LLM
            index = self.session_manager._read_index()

            if not index:
                return {"results": [], "message": "No previous sessions to search."}

            index_summary = self._format_index_for_llm(index)

            # Ask LLM to select most relevant sessions
            selection_prompt = f"""Given this query: "{query}"

And this session index:
{index_summary}

Return ONLY a JSON array of the {top_k} most relevant session IDs (the 8-character codes in brackets).
Example: ["abc12345", "def67890", "ghi11111"]

If fewer than {top_k} sessions are relevant, return fewer. If none are relevant, return [].
Return ONLY the JSON array, no explanation."""

            selection_response = self.client.chat.completions.create(
                model=self.config.inference_providers.huggingface.prompt_model_id,
                messages=[{"role": "user", "content": selection_prompt}],
                temperature=0.1,
                max_tokens=256,
            )

            # Parse selected session IDs
            try:
                selected_ids_short = json.loads(selection_response.choices[0].message.content.strip())
            except json.JSONDecodeError:
                # Fallback: take first top_k sessions
                selected_ids_short = [entry["session_id"][:8] for entry in index[:top_k]]

            if not selected_ids_short:
                return {"results": "No relevant sessions found for this query.", "sessions_searched": len(index)}

            # Map short IDs back to full IDs
            selected_full_ids = []
            for short_id in selected_ids_short:
                for entry in index:
                    if entry["session_id"].startswith(short_id):
                        selected_full_ids.append(entry["session_id"])
                        break

            # Stage 2: Fetch details and summarize
            session_details = self._get_session_details(selected_full_ids)
            summary_prompt = f"""Query: "{query}"

Relevant session details:
{session_details}

Based on this query and the session details, provide a helpful summary that:
1. Highlights the most relevant prompts and techniques
2. Notes successful outputs
3. Extracts patterns or learnings applicable to the query

Be concise and actionable."""

            summary_response = self.client.chat.completions.create(
                model=self.config.inference_providers.huggingface.prompt_model_id,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=1024,
            )

            return {
                "results": summary_response.choices[0].message.content,
                "sessions_searched": len(index),
                "sessions_selected": selected_full_ids,
            }

        except Exception as e:
            return {"error": str(e)}
