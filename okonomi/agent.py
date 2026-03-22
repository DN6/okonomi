"""Pydantic AI-based agent for Okonomi with code execution.

This module provides the OkonomiAgent class that uses Pydantic AI for LLM interactions
with a local Python code executor that has access to all tool instances.
"""

import asyncio
import inspect
import io
import logging
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.openrouter import OpenRouterModel

from .config import OkonomiConfig
from .context import ContextManager
from .display import DIM, RST, RED, print_ansi_box, print_code_box
from .session import GeneratedOutput, Session, SessionManager, SessionMetadata
from .tools._image_generation import ImageGenerationTools
from .tools._prompt import PromptTools
from .tools._research import ResearchTools
from .tools._sessions import SessionTools
from .tools.skills import SkillTools
from .tools._video import VideoTools
from .tools.terminal import TerminalTools


logger = logging.getLogger(__name__)


class CachedOpenRouterModel(OpenRouterModel):
    """OpenRouterModel with Anthropic prompt caching.

    Uses OpenRouter's top-level cache_control to automatically cache
    the conversation prefix for Claude models, reducing input token
    costs by ~75% on multi-turn conversations.
    """

    def __init__(self, model_name: str, enable_caching: bool = True, **kwargs: Any):
        super().__init__(model_name, **kwargs)
        self._enable_caching = enable_caching and "claude" in model_name.lower()
        if self._enable_caching:
            logger.info("Prompt caching enabled for %s", model_name)

    async def _completions_create(self, *args: Any, **kwargs: Any) -> Any:
        if self._enable_caching:
            model_settings = args[2] if len(args) > 2 else kwargs.get("model_settings", {})
            extra_body = dict(model_settings.get("extra_body", {}) or {})
            extra_body["cache_control"] = {"type": "ephemeral"}
            if isinstance(model_settings, dict):
                model_settings = {**model_settings, "extra_body": extra_body}
            else:
                model_settings["extra_body"] = extra_body
            if len(args) > 2:
                args = (*args[:2], model_settings, *args[3:])
            else:
                kwargs["model_settings"] = model_settings
        return await super()._completions_create(*args, **kwargs)


@dataclass
class TokenUsage:
    """Tracks token usage across agent runs."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0

    def add(self, input_tokens: int, output_tokens: int, requests: int = 1) -> None:
        """Add usage from a single run."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.requests += requests

    def __str__(self) -> str:
        return f"tokens: {self.total_tokens:,} (in: {self.input_tokens:,}, out: {self.output_tokens:,}) | requests: {self.requests}"


@dataclass
class AgentDependencies:
    """Dependencies injected into the agent and tools at runtime."""

    config: OkonomiConfig
    session_id: str
    session_manager: SessionManager
    tools: dict[str, Any] = field(default_factory=dict)
    generated_outputs: list[GeneratedOutput] = field(default_factory=list)
    input_assets: list[str] = field(default_factory=list)
    tools_used: set[str] = field(default_factory=set)
    models_used: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)
    verbose: bool = False
    # Persistent state for code execution across multiple calls
    execution_globals: dict[str, Any] = field(default_factory=dict)

    def add_output(self, output: GeneratedOutput) -> None:
        """Track a generated output."""
        self.generated_outputs.append(output)
        if output.tool_name:
            self.tools_used.add(output.tool_name)
        if output.model_id:
            self.models_used.add(output.model_id)

    def add_input_asset(self, url: str) -> None:
        """Track an input asset (image/video URL)."""
        if url and url not in self.input_assets:
            self.input_assets.append(url)

    def add_error(self, error: str) -> None:
        """Track an error that occurred during execution."""
        self.errors.append(error)

    def get_tool(self, namespace: str) -> Any:
        """Get a tool instance by namespace."""
        return self.tools.get(namespace)


class ToolWrapper:
    """Wrapper that adds tracking to tool method calls."""

    def __init__(self, tool_instance: Any, deps: AgentDependencies, namespace: str):
        self._tool_instance = tool_instance
        self._deps = deps
        self._namespace = namespace
        self._build_doc()

    def _build_doc(self) -> None:
        """Build documentation string from tool metadata."""
        if hasattr(self._tool_instance, "__get_metadata__"):
            meta = self._tool_instance.__get_metadata__()
            lines = [f"{self._namespace}: {meta.get('description', '')}", ""]
            lines.append("Methods:")
            for method_name, method_info in meta.get("methods", {}).items():
                sig = method_info.get("signature", method_name)
                desc = method_info.get("description", "")
                lines.append(f"  {sig}")
                lines.append(f"      {desc}")
            lines.append("")
            lines.append("Properties:")
            lines.append("  session_dir: str")
            lines.append("      Path to the current session directory for saving files.")
            self.__doc__ = "\n".join(lines)
        else:
            self.__doc__ = self._tool_instance.__doc__ or ""

    def __dir__(self) -> list[str]:
        """List available methods and properties."""
        tool_attrs = [
            name
            for name in dir(self._tool_instance)
            if not name.startswith("_") and callable(getattr(self._tool_instance, name, None))
        ]
        return tool_attrs + ["session_dir"]

    def __repr__(self) -> str:
        return f"<{self._namespace} tool - use help({self._namespace}) for details>"

    @property
    def session_dir(self) -> str:
        """Get the current session directory path, creating it if needed."""
        session_dir = self._deps.session_manager.get_session_dir(self._deps.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        return str(session_dir)

    def _generate_output_path(self, method_name: str, extension: str = ".mp4") -> str:
        """Generate an output path in the session directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{method_name}_{timestamp}{extension}"
        return f"{self.session_dir}/{filename}"

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._tool_instance, name)

        if callable(attr) and not name.startswith("_"):

            def wrapper(*args, **kwargs):
                tool_name = f"{self._namespace}.{name}"

                if "output_path" in kwargs and kwargs["output_path"] is None:
                    kwargs["output_path"] = self._generate_output_path(name)

                result = attr(*args, **kwargs)

                self._deps.tools_used.add(tool_name)

                for key in ["image", "image_url", "video", "video_url", "first_frame", "last_frame"]:
                    if key in kwargs and kwargs[key]:
                        self._deps.add_input_asset(kwargs[key])

                if isinstance(result, dict):
                    self._track_outputs(result, name, kwargs)

                return result

            wrapper.__name__ = name
            wrapper.__doc__ = attr.__doc__
            try:
                wrapper.__signature__ = inspect.signature(attr)
            except (ValueError, TypeError):
                pass
            return wrapper
        return attr

    def _track_outputs(self, result: dict, method_name: str, kwargs: dict) -> None:
        """Extract and track generated outputs from tool results."""
        now = datetime.now().isoformat()

        # Handle images (always plural in results)
        if "images" in result:
            for img_url in result["images"]:
                output = GeneratedOutput(
                    asset_url=img_url,
                    asset_type="image",
                    prompt=kwargs.get("prompt", ""),
                    tool_name=f"{self._namespace}.{method_name}",
                    model_id=result.get("model_id", ""),
                    timestamp=now,
                )
                self._deps.add_output(output)

        # Handle video (always singular in results)
        if "video" in result:
            output = GeneratedOutput(
                asset_url=result["video"],
                asset_type="video",
                prompt=kwargs.get("prompt", ""),
                tool_name=f"{self._namespace}.{method_name}",
                model_id=result.get("model_id", ""),
                timestamp=now,
            )
            self._deps.add_output(output)


class OkonomiAgent:
    """Pydantic AI-based agent with local code execution.

    This agent provides:
    - Code execution with tool instances injected into the environment
    - Session management for continuity
    - Message history for multi-turn conversations
    """

    def __init__(
        self,
        config: OkonomiConfig,
        session_id: Optional[str] = None,
        verbose: bool = False,
    ):
        self.config = config
        self.session_manager = SessionManager()
        self.session_id = session_id or self.session_manager.create_session_id()
        self._setup_tools()
        self.message_history: list[ModelMessage] = []
        self.task: str = ""
        self.verbose = verbose

        # Persistent REPL state across turns
        self.execution_globals: dict[str, Any] = {}

        # Interrupt state
        self._was_interrupted: bool = False
        self._partial_output: str = ""

        # Session tracking
        self.generated_outputs: list[GeneratedOutput] = []
        self.input_assets: list[str] = []
        self.tools_used: set[str] = set()
        self.models_used: set[str] = set()
        self.errors: list[str] = []

        self.token_usage = TokenUsage()
        self.context = ContextManager(config)
        self.spinner_text: str = ""

        # Skill creation nudge: after N tool-calling turns without skill
        # management, append a reminder to consider saving as a skill.
        self._turns_since_skill = 0
        self._skill_nudge_interval = 10

        self.agent = self._create_agent()

    def _setup_tools(self) -> None:
        """Create all tool instances.

        Tools are split into two categories:
        - **Programmatic tools**: Available inside ``execute_code`` via ToolWrapper.
          The model writes Python that calls these in loops, filters results, etc.
        - **Standalone tools**: Registered directly on the agent. Each call is a
          separate model round-trip.
        """
        # Programmatic tools — called from inside execute_code
        self.programmatic_tools: dict[str, Any] = {
            "image_generation": ImageGenerationTools(self.config),
            "prompt": PromptTools(self.config),
            "research": ResearchTools(self.config),
            "sessions": SessionTools(self.config),
            "video": VideoTools(self.config),
        }

        # Standalone tools — direct agent tools (one call per round-trip)
        self.terminal = TerminalTools(self.config)
        self.skills = SkillTools(
            self.config,
            available_tools=set(self.programmatic_tools.keys()) | {"terminal", "skills"},
        )

    def _create_agent(self) -> Agent[AgentDependencies, str]:
        """Create and configure the Pydantic AI agent."""
        model = self._create_model()

        agent = Agent(
            model,
            deps_type=AgentDependencies,
            system_prompt=self._get_system_prompt(),
        )

        # Standalone tools
        self._register_tool(agent, self.skills)
        self._register_tool(agent, self.terminal)

        # Programmatic tools (available inside execute_code)
        self._register_code_execution_tool(agent)

        return agent

    def _create_model(self) -> CachedOpenRouterModel:
        return CachedOpenRouterModel(self.config.agent.model_id)

    def _build_skills_index(self) -> str:
        """Build a skills index from discovered skills for the system prompt."""
        result = self.skills.list_skills()
        skills = result.get("skills", [])
        if not skills:
            return ""

        by_category: dict[str, list[dict]] = {}
        for skill in skills:
            cat = skill.get("category", "") or "general"
            by_category.setdefault(cat, []).append(skill)

        lines = []
        for category in sorted(by_category.keys()):
            lines.append(f"  {category}:")
            for skill in sorted(by_category[category], key=lambda s: s["name"]):
                if skill["description"]:
                    lines.append(f"    - {skill['name']}: {skill['description']}")
                else:
                    lines.append(f"    - {skill['name']}")
        return "\n".join(lines)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        tool_docs = self._build_tool_documentation()
        skills_index = self._build_skills_index()

        return f"""You are an expert creative assistant for generating images and videos using code.

## Available Tools

{tool_docs}

## Skills (mandatory)

Before replying, scan the skill list below. If one clearly matches the task, you MUST
load it by calling `load_skill("skill-name")` directly. Then follow the loaded instructions step by step.

After difficult or iterative tasks, offer to save the approach as a skill using
`create_skill(name, content)`.

<available_skills>
{skills_index}
</available_skills>

If none match, proceed normally without loading a skill.

## Creative Workflow

Before generating any media, determine if the request is **direct** or **conceptual**:

### Direct Requests (proceed to generation)
User provides specific visual details: subject, composition, style, colors, lighting.
Examples:
- "A red fox sitting in snow, soft morning light, photorealistic"
- "Edit this image to change the background to a beach"

### Conceptual Requests (develop concept first)
User provides themes, moods, abstract ideas, or vague descriptions.
Examples:
- "Something cyberpunk"
- "Create art about loneliness"
- "Make a cool character"

**For conceptual requests, follow this workflow:**

1. **Research & Ideation** - Load the `deep-concept-research` skill for structured creative exploration. Use `research.web_search()` for inspiration and references.
2. **Determine Style** - Decide on the visual style (photorealistic, illustration, anime, painterly, etc.) based on the concept.
3. **Create Prompt** - Use `prompt.create_prompt()` with context from your research.
4. **Generate** - Proceed to image generation with the developed concept.

## Tool-Specific Requirements

### edit_image & reference_image
These tools require **structured prompts** that explicitly state:
- **What to preserve**: style, colors, composition, subject identity
- **What to change**: specific elements to modify
- **Reference attribution**: "The style from image1", "The person from image2"

Example structured prompts:
- "Preserve the color palette and lighting from image1. Apply it to a new scene of a mountain landscape at dawn."
- "Keep the person's face and clothing from image1. Change the background to a futuristic cityscape."
- "Using the painterly style from image1, recreate the composition with autumn colors."

## Rules

1. Use `execute_code` to run Python code for image/video generation tools
2. Use `print()` to see results from code execution
3. Variables persist between executions
4. All methods are synchronous - do NOT use `await`
5. For tools with `[detailed docs]`, call `get_tool_documentation('tool.method')` first.
6. **Saving files**: All tools have a `session_dir` property (e.g., `video.session_dir`) for the current session folder.
   Methods with `output_path` parameters auto-save to session directory when passed `output_path=None`.
7. **Standalone tools** (call directly, NOT through `execute_code`):
   - Terminal: `run`, `run_background`, `search`
   - Skills: `load_skill`, `create_skill`, `edit_skill`, `list_skills`
8. **Programmatic tools** (use inside `execute_code` only):
   - `image_generation`, `prompt`, `research`, `sessions`, `video`
9. **Session history**: Past sessions are stored in `~/.okonomi/sessions/` as JSON and JSONL files.
   Use `search(pattern="<keyword>", path="~/.okonomi/sessions/")` to grep through past session
   data, concept research, prompts, and outputs. Always prefer `search` for finding sessions —
   it's faster and doesn't require code execution. Only use programmatic session tools
   (`sessions.list_sessions()`, `sessions.get_session_details()`, etc.) when you need
   structured access to a specific session you've already identified.

Be creative and help users generate stunning images and videos."""

    def _build_tool_documentation(self) -> str:
        """Build documentation for all available tools."""
        docs = []

        for namespace, tool_instance in self.programmatic_tools.items():
            if hasattr(tool_instance, "__get_metadata__"):
                metadata = tool_instance.__get_metadata__()
                docs.append(f"### {namespace}")
                docs.append(f"{metadata.get('description', '')}")
                docs.append("")

                for method_name, method_info in metadata.get("methods", {}).items():
                    signature = method_info.get("signature", f"{method_name}(...)")
                    description = method_info.get("description", "")
                    # Mark methods with detailed documentation (docstring > 500 chars)
                    docstring = method_info.get("docstring", "")
                    has_detailed_docs = len(docstring) > 500
                    doc_marker = " [detailed docs]" if has_detailed_docs else ""
                    docs.append(f"- `{namespace}.{signature}` - {description}{doc_marker}")

                docs.append("")

        return "\n".join(docs)

    def _register_code_execution_tool(self, agent: Agent) -> None:
        """Register the code execution tool with the agent."""

        @agent.tool
        async def execute_code(ctx: RunContext[AgentDependencies], code: str) -> str:
            """Execute Python code with access to all tool objects.

            The code runs in an environment with these tool objects available:
            - image_generation: Text-to-image, image-to-image, editing
            - prompt: Prompt creation and optimization
            - research: Web search
            - sessions: List, load, and inspect past sessions and their outputs
            - video: Video generation

            Args:
                code: Python code to execute. Use print() for output.

            Returns:
                The stdout output from the code execution, or error message.
            """
            # Initialize execution globals with tool wrappers if not already done
            if not ctx.deps.execution_globals:
                ctx.deps.execution_globals = {
                    "__builtins__": __builtins__,
                    "asyncio": asyncio,
                }
                # Add wrapped tools
                for namespace, tool_instance in ctx.deps.tools.items():
                    ctx.deps.execution_globals[namespace] = ToolWrapper(tool_instance, ctx.deps, namespace)

                # Add helper functions for tool discovery
                def get_tools_readme() -> str:
                    """Get a summary of all available tools and their methods."""
                    readme = "# Available Tools\n\n"
                    for ns, tool in ctx.deps.tools.items():
                        if hasattr(tool, "__get_metadata__"):
                            meta = tool.__get_metadata__()
                            readme += f"## {ns}\n{meta.get('description', '')}\n\n"
                            for method_name, method_info in meta.get("methods", {}).items():
                                sig = method_info.get("signature", f"{method_name}(...)")
                                desc = method_info.get("description", "")
                                readme += f"- `{ns}.{sig}` - {desc}\n"
                            readme += "\n"
                    return readme

                def get_tool_documentation(method_path: str) -> str:
                    """Get detailed documentation for a specific tool method.

                    IMPORTANT: Use this function before calling tools with complex parameters
                    or specific instructions. Many tools (especially reference_image, text_to_image,
                    edit_image) have detailed usage guides in their documentation.

                    Args:
                        method_path: The path to the method, e.g., 'image_generation.reference_image'

                    Returns:
                        Full documentation including signature, parameters, and usage examples.

                    Example:
                        docs = get_tool_documentation('image_generation.reference_image')
                        print(docs)
                    """
                    parts = method_path.split(".")
                    if len(parts) != 2:
                        return f"Invalid method path: {method_path}. Use format 'tool_name.method_name'"

                    namespace, method_name = parts
                    tool = ctx.deps.tools.get(namespace)
                    if not tool:
                        available = ", ".join(ctx.deps.tools.keys())
                        return f"Tool '{namespace}' not found. Available tools: {available}"

                    method = getattr(tool, method_name, None)
                    if not method:
                        # List available methods
                        methods = [m for m in dir(tool) if not m.startswith("_") and callable(getattr(tool, m))]
                        return f"Method '{method_name}' not found in '{namespace}'. Available: {', '.join(methods)}"

                    import inspect

                    doc = inspect.getdoc(method) or "No documentation available"
                    sig = inspect.signature(method)

                    # Build parameter details
                    params_doc = []
                    for param_name, param in sig.parameters.items():
                        if param_name == "self":
                            continue
                        param_type = str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
                        default = ""
                        if param.default != inspect.Parameter.empty:
                            default = f" = {param.default!r}"
                        params_doc.append(f"  - {param_name}: {param_type}{default}")

                    params_section = "\n".join(params_doc) if params_doc else "  (no parameters)"

                    return f"""# {namespace}.{method_name}

Signature: `{method_name}{sig}`

## Parameters:
{params_section}

## Documentation:
{doc}"""

                # Alias for backward compatibility
                def get_tool_method_documentation(method_path: str) -> str:
                    """Alias for get_tool_documentation."""
                    return get_tool_documentation(method_path)

                def get_session_dir() -> str:
                    """Get the current session's directory path.

                    Use this to save files (like stitched videos) to the session folder.

                    Returns:
                        Absolute path to the current session directory.

                    Example:
                        session_dir = get_session_dir()
                        output_path = f"{session_dir}/final_video.mp4"
                        result = video.stitch_videos(video_urls, output_path)
                    """
                    session_dir = ctx.deps.session_manager.get_session_dir(ctx.deps.session_id)
                    session_dir.mkdir(parents=True, exist_ok=True)
                    return str(session_dir)

                ctx.deps.execution_globals["get_tools_readme"] = get_tools_readme
                ctx.deps.execution_globals["get_tool_documentation"] = get_tool_documentation
                ctx.deps.execution_globals["get_tool_method_documentation"] = get_tool_method_documentation
                ctx.deps.execution_globals["get_session_dir"] = get_session_dir

            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            self.spinner_text = "Running"

            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, ctx.deps.execution_globals)

                stdout_output = stdout_capture.getvalue()
                stderr_output = stderr_capture.getvalue()

                result = ""
                if stdout_output:
                    result += stdout_output
                if stderr_output:
                    result += f"\nStderr:\n{stderr_output}"

                output_text = result.strip() if result.strip() else "Code executed successfully"
                is_skill_load = "skills.load_skill" in code
                if not is_skill_load:
                    print_ansi_box(output_text, label="output")

                return result.strip() if result.strip() else "Code executed successfully (no output)"

            except Exception:
                error_msg = f"Error executing code:\n{traceback.format_exc()}"
                print_ansi_box(error_msg, label="error", color="31")
                return error_msg
            finally:
                self.spinner_text = ""

    def _register_tool(
        self,
        agent: Agent,
        tool_instance: Any,
        rename: dict[str, str] | None = None,
    ) -> None:
        """Register all public methods of a tool instance as direct agent tools.

        Uses each method's existing signature and docstring for schema generation.

        Args:
            agent: The Pydantic AI agent to register tools on.
            tool_instance: An object whose public methods become agent tools.
            rename: Optional mapping of method names to alternate tool names.
        """
        rename = rename or {}

        for name, method in inspect.getmembers(tool_instance, predicate=inspect.ismethod):
            if name.startswith("_"):
                continue

            tool_name = rename.get(name, name)
            sig = inspect.signature(method)

            def make_wrapper(m, s, tn):
                def wrapper(*args, **kwargs):
                    return m(*args, **kwargs)

                wrapper.__name__ = tn
                wrapper.__qualname__ = tn
                wrapper.__doc__ = m.__doc__
                wrapper.__signature__ = s
                wrapper.__annotations__ = {
                    k: v.annotation for k, v in s.parameters.items() if v.annotation is not inspect.Parameter.empty
                }
                if s.return_annotation is not inspect.Signature.empty:
                    wrapper.__annotations__["return"] = s.return_annotation
                return wrapper

            agent.tool_plain(make_wrapper(method, sig, tool_name), name=tool_name)

    def _get_dependencies(self) -> AgentDependencies:
        """Create dependencies for an agent run."""
        return AgentDependencies(
            config=self.config,
            session_id=self.session_id,
            session_manager=self.session_manager,
            tools=self.programmatic_tools,
            generated_outputs=self.generated_outputs,
            input_assets=self.input_assets,
            tools_used=self.tools_used,
            models_used=self.models_used,
            errors=self.errors,
            verbose=self.verbose,
            execution_globals=self.execution_globals,
        )

    def _sanitize_history(self, messages: list[ModelMessage]) -> list[ModelMessage]:
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

    def _display_tool_call(self, part: ToolCallPart) -> None:
        """Display a tool call with appropriate formatting."""
        if part.tool_name == "execute_code":
            args = part.args
            if isinstance(args, dict):
                code = args.get("code", "")
            elif isinstance(args, str):
                import json

                try:
                    parsed = json.loads(args)
                    code = parsed.get("code", "") if isinstance(parsed, dict) else ""
                except json.JSONDecodeError:
                    code = args
            elif hasattr(args, "code"):
                code = args.code
            else:
                code = str(args)

            if code:
                print_code_box(code)
        elif part.tool_name == "load_skill":
            import json

            args = part.args
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            name = args.get("name", args) if isinstance(args, dict) else args
            print(f"{DIM}Loading skill: {name}{RST}")
        else:
            print(f"{DIM}{part.tool_name}({part.args}){RST}")

    async def run_async(self, user_message: str, reset: bool = False) -> str:
        """Run the agent asynchronously with a user message.

        Supports cancellation via asyncio.CancelledError — partial message history
        and generated outputs are preserved when interrupted.

        Args:
            user_message: The user's message/task
            reset: Whether to reset conversation history

        Returns:
            The agent's response
        """
        if reset:
            self.message_history = []
            self.execution_globals = {}
            self.generated_outputs = []
            self.input_assets = []
            self.tools_used = set()
            self.models_used = set()
            self.errors = []
            self.token_usage = TokenUsage()

        self.task = user_message
        self._was_interrupted = False
        self._partial_output = ""
        deps = self._get_dependencies()
        final_output = ""

        if self.message_history:
            self.message_history = self._sanitize_history(self.message_history)

            if self.context.should_compress_preflight(self.message_history):
                compressed, was_compressed = self.context.compress(self.message_history)
                if was_compressed:
                    print(
                        f"{DIM}Context compressed: {len(self.message_history)} → " f"{len(compressed)} messages{RST}"
                    )
                    self.message_history = compressed

        # Append skill creation nudge when threshold is reached
        if self._skill_nudge_interval > 0 and self._turns_since_skill >= self._skill_nudge_interval:
            user_message += (
                "\n\n[System note: You've completed several tool-heavy turns. "
                "If you developed a reusable approach, consider saving it as a "
                "skill with create_skill().]"
            )
            self._turns_since_skill = 0

        try:
            async with self.agent.iter(
                user_message,
                deps=deps,
                message_history=self.message_history if self.message_history else None,
            ) as agent_run:
                try:
                    async for node in agent_run:
                        if Agent.is_model_request_node(node):
                            self.spinner_text = "Thinking"
                            async with node.stream(agent_run.ctx) as stream:
                                accumulated_text = ""
                                line_buf = ""

                                async for text in stream.stream_text(delta=True, debounce_by=None):
                                    if self.spinner_text:
                                        self.spinner_text = ""
                                    accumulated_text += text
                                    line_buf += text
                                    while "\n" in line_buf:
                                        line, line_buf = line_buf.split("\n", 1)
                                        print(f"{DIM}{line}{RST}")

                                self.spinner_text = ""

                                if line_buf.strip():
                                    print(f"{DIM}{line_buf}{RST}")

                                if accumulated_text.strip():
                                    final_output = accumulated_text

                                response = stream.response
                                if response and response.parts:
                                    for part in response.parts:
                                        if isinstance(part, ToolCallPart):
                                            self._display_tool_call(part)

                except asyncio.CancelledError:
                    self._was_interrupted = True
                    self._partial_output = final_output
                    self.message_history = self._sanitize_history(agent_run.all_messages())
                    self.generated_outputs = deps.generated_outputs
                    usage = agent_run.usage()
                    self.token_usage.add(
                        input_tokens=usage.request_tokens or 0,
                        output_tokens=usage.response_tokens or 0,
                        requests=usage.requests or 1,
                    )
                    return self._partial_output

                result = agent_run.result

                # Track token usage
                usage = agent_run.usage()
                self.token_usage.add(
                    input_tokens=usage.request_tokens or 0,
                    output_tokens=usage.response_tokens or 0,
                    requests=usage.requests or 1,
                )

            # Update message history for conversation continuity
            self.message_history = result.all_messages()
            self.generated_outputs = deps.generated_outputs

            # Skill creation nudge: track tool-calling turns and remind
            # the model to consider saving approaches as skills.
            messages = result.all_messages()
            used_skill_manage = any(
                isinstance(part, ToolCallPart) and part.tool_name in ("create_skill", "edit_skill")
                for msg in messages
                if isinstance(msg, ModelResponse)
                for part in msg.parts
            )
            if used_skill_manage:
                self._turns_since_skill = 0
            elif deps.tools_used:
                self._turns_since_skill += 1

            # Return the final output (use result.output if we didn't capture it during streaming)
            return final_output if final_output else result.output

        except asyncio.CancelledError:
            return self._partial_output
        except Exception as e:
            print(f"{RED}Error: {e}{RST}")
            return f"Error: {str(e)}"

    def run(self, user_message: str, reset: bool = False) -> str:
        """Run the agent synchronously with a user message.

        Args:
            user_message: The user's message/task
            reset: Whether to reset conversation history

        Returns:
            The agent's response
        """
        return asyncio.run(self.run_async(user_message, reset=reset))

    def _generate_steps_summary(self) -> str:
        """Generate LLM summary of session actions."""
        return self.context.generate_steps_summary(self.message_history)

    def _download_asset(self, url: str, session_dir: Path, asset_type: str) -> str:
        """Download an asset from URL to the session directory.

        Args:
            url: URL to download from
            session_dir: Directory to save the asset
            asset_type: "image" or "video"

        Returns:
            Local file path, or empty string if download failed
        """
        try:
            # Extract filename from URL path (e.g., "1Bq5pfalxttyXEHIrNcNX.jpg")
            parsed = urlparse(url)
            url_filename = Path(parsed.path).name

            # Use the URL filename if it has an extension, otherwise add default
            if Path(url_filename).suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
                ".gif",
                ".mp4",
                ".webm",
                ".mov",
            ]:
                filename = url_filename
            else:
                ext = ".jpg" if asset_type == "image" else ".mp4"
                filename = f"{url_filename}{ext}"

            local_path = session_dir / filename

            # Download the file
            with httpx.Client(timeout=60.0) as client:
                response = client.get(url, follow_redirects=True)
                response.raise_for_status()

                with open(local_path, "wb") as f:
                    f.write(response.content)

            return str(local_path)
        except Exception:
            return ""

    def _download_session_assets(self, session_dir: Path) -> None:
        """Download all generated outputs to the session directory.

        Args:
            session_dir: Directory to save assets
        """
        for output in self.generated_outputs:
            if output.asset_url and not output.local_path:
                local_path = self._download_asset(output.asset_url, session_dir, output.asset_type)
                output.local_path = local_path

    def save_session(self, status: str = "completed") -> None:
        """Save the current session state.

        Args:
            status: Status of the session (completed, error, abandoned)
        """
        # Get session directory and ensure it exists
        session_dir = self.session_manager.get_session_dir(self.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        # Download all assets to the session directory
        self._download_session_assets(session_dir)

        now = datetime.now().isoformat()
        metadata = SessionMetadata(
            session_id=self.session_id,
            task_input=self.task,
            status=status,
            created_at=now,
            updated_at=now,
            num_outputs=len(self.generated_outputs),
            tools_used=list(self.tools_used),
            models_used=list(self.models_used),
            has_input_assets=len(self.input_assets) > 0,
            token_usage={
                "input_tokens": self.token_usage.input_tokens,
                "output_tokens": self.token_usage.output_tokens,
                "total_tokens": self.token_usage.total_tokens,
                "requests": self.token_usage.requests,
            },
        )

        session = Session(
            metadata=metadata,
            generated_outputs=self.generated_outputs,
            input_assets=self.input_assets,
            steps_summary=self._generate_steps_summary(),
            errors=self.errors,
            message_history=self.message_history if self.message_history else None,
        )

        self.session_manager.save_session(session)

    def load_session(self, session_id: str) -> bool:
        """Load agent state from a saved session.

        Args:
            session_id: ID of the session to load

        Returns:
            True if session was loaded successfully
        """
        session = self.session_manager.load_session(session_id)

        if not session:
            return False

        self.session_id = session_id
        self.task = session.metadata.task_input
        self.generated_outputs = session.generated_outputs
        self.input_assets = session.input_assets
        self.tools_used = set(session.metadata.tools_used)
        self.models_used = set(session.metadata.models_used)
        self.errors = session.errors

        if session.message_history:
            self.message_history = session.message_history

        if session.metadata.token_usage:
            tu = session.metadata.token_usage
            self.token_usage = TokenUsage(
                input_tokens=tu.get("input_tokens", 0),
                output_tokens=tu.get("output_tokens", 0),
                total_tokens=tu.get("total_tokens", 0),
                requests=tu.get("requests", 0),
            )

        return True
