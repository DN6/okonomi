"""Pydantic AI-based agent for Okonomi with code execution.

This module provides the OkonomiAgent class that uses Pydantic AI for LLM interactions
with a local Python code executor that has access to all tool instances.
"""

import asyncio
import io
import os
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from langfuse import Langfuse, observe
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

from .config import OkonomiConfig
from .memory import MemoryManager
from .session import GeneratedOutput, Session, SessionManager, SessionMetadata
from .tools._image import ImageTools
from .tools._image_generation import ImageGenerationTools
from .tools._memory import MemoryTools
from .tools._prompt import PromptTools
from .tools._research import ResearchTools
from .tools._sessions import SessionTools
from .tools._strategies import StrategiesTools
from .tools._video import VideoTools


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
    loras_used: list[tuple] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    console: Optional[Console] = None
    # Persistent state for code execution across multiple calls
    execution_globals: dict[str, Any] = field(default_factory=dict)

    def add_output(self, output: GeneratedOutput) -> None:
        """Track a generated output."""
        self.generated_outputs.append(output)
        if output.tool_name:
            self.tools_used.add(output.tool_name)
        if output.model_id:
            self.models_used.add(output.model_id)
        if output.loras:
            for lora in output.loras:
                if lora not in self.loras_used:
                    self.loras_used.append(lora)

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


class AsyncToolWrapper:
    """Wrapper that converts async tool methods to sync for code execution."""

    def __init__(self, tool_instance: Any, deps: AgentDependencies, namespace: str):
        self._tool_instance = tool_instance
        self._deps = deps
        self._namespace = namespace

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._tool_instance, name)

        if asyncio.iscoroutinefunction(attr):

            def sync_wrapper(*args, **kwargs):
                tool_name = f"{self._namespace}.{name}"

                # Use Langfuse observation for tool tracing
                @observe(name=tool_name, as_type="tool")
                def traced_call():
                    try:
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, attr(*args, **kwargs))
                            return future.result()
                    except RuntimeError:
                        return asyncio.run(attr(*args, **kwargs))

                try:
                    result = traced_call()

                    # Track tool usage
                    self._deps.tools_used.add(tool_name)

                    # Track input assets from kwargs
                    for key in ["image", "image_url", "video", "video_url", "first_frame", "last_frame"]:
                        if key in kwargs and kwargs[key]:
                            self._deps.add_input_asset(kwargs[key])

                    # Track outputs
                    if isinstance(result, dict):
                        self._track_outputs(result, name, kwargs)

                    return result

                except Exception:
                    raise

            return sync_wrapper
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
                    loras=kwargs.get("lora_id", []) or [],
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
    - Optional Arize Phoenix tracing for observability
    """

    _tracing_configured: bool = False

    def __init__(
        self,
        config: OkonomiConfig,
        session_id: Optional[str] = None,
        console: Optional[Console] = None,
        verbose: bool = False,
    ):
        """Initialize the Okonomi agent.

        Args:
            config: Okonomi configuration object
            session_id: Optional session ID to resume from
            console: Rich console for output
            verbose: Show executed code and output when True
        """
        self.config = config
        self.session_manager = SessionManager()
        self.session_id = session_id or self.session_manager.create_session_id()
        self.console = console or Console()
        self.tool_instances = self._create_tools()
        self.message_history: list[ModelMessage] = []
        self.task: str = ""
        self.verbose = verbose

        # Session tracking
        self.generated_outputs: list[GeneratedOutput] = []
        self.input_assets: list[str] = []
        self.tools_used: set[str] = set()
        self.models_used: set[str] = set()
        self.loras_used: list[tuple] = []
        self.errors: list[str] = []

        # Token usage tracking
        self.token_usage = TokenUsage()

        # Memory management for message history
        self.memory_manager = MemoryManager(config)

        # Configure tracing if enabled
        self._setup_tracing()

        # Create the Pydantic AI agent
        self.agent = self._create_agent()

    def _setup_tracing(self) -> None:
        """Configure Langfuse tracing if enabled in config."""
        if not self.config.tracing.enabled:
            return

        if OkonomiAgent._tracing_configured:
            return

        # Initialize Langfuse client and verify authentication
        try:
            langfuse = Langfuse()
            if not langfuse.auth_check():
                self.console.print("[dim]Langfuse: Authentication failed, tracing disabled[/dim]")
                return
        except Exception as e:
            self.console.print(f"[dim]Langfuse: {e}, tracing disabled[/dim]")
            return

        # Use Pydantic AI's built-in instrumentation
        Agent.instrument_all()
        OkonomiAgent._tracing_configured = True
        self.console.print("[dim]Langfuse: Tracing enabled[/dim]")

    def _create_tools(self) -> dict[str, Any]:
        """Create all tool instances."""
        return {
            "image": ImageTools(self.config),
            "image_generation": ImageGenerationTools(self.config),
            "prompt": PromptTools(self.config),
            "memory": MemoryTools(self.config),
            "research": ResearchTools(self.config),
            "sessions": SessionTools(self.config),
            "strategies": StrategiesTools(self.config),
            "video": VideoTools(self.config),
        }

    def _create_agent(self) -> Agent[AgentDependencies, str]:
        """Create and configure the Pydantic AI agent."""
        model = self._create_model()

        agent = Agent(
            model,
            deps_type=AgentDependencies,
            system_prompt=self._get_system_prompt(),
            instrument=self._tracing_configured,
        )

        # Register the code execution tool
        self._register_code_execution_tool(agent)

        return agent

    def _create_model(self) -> AnthropicModel | OpenAIModel:
        """Create the appropriate model based on configuration."""
        model_id = self.config.agent.model_id
        provider = self.config.agent.model_provider

        # Use native Anthropic model for Anthropic provider
        if provider == "anthropic":
            return AnthropicModel(model_id)

        # OpenAI-compatible providers
        provider_configs = {
            "together": {
                "base_url": "https://api.together.xyz/v1",
                "api_key_env": "TOGETHER_API_KEY",
            },
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
            "groq": {
                "base_url": "https://api.groq.com/openai/v1",
                "api_key_env": "GROQ_API_KEY",
            },
            "fireworks": {
                "base_url": "https://api.fireworks.ai/inference/v1",
                "api_key_env": "FIREWORKS_API_KEY",
            },
            "huggingface": {
                "base_url": "https://router.huggingface.co/v1",
                "api_key_env": "HF_TOKEN",
            },
        }

        config = provider_configs.get(provider, provider_configs["openai"])

        return OpenAIModel(
            model_id,
            provider=OpenAIProvider(
                base_url=config["base_url"],
                api_key=os.environ.get(config["api_key_env"]),
            ),
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        tool_docs = self._build_tool_documentation()

        return f"""You are an expert creative assistant for generating images and videos using code.

## Available Tools

{tool_docs}

## Quick Examples

**Generate an image:**
```python
result = prompt.create_prompt(context="A cyberpunk cityscape at night")
print(result)
image = image_generation.text_to_image(prompt=result["prompt"])
print(image)
```

**Edit an image:**
```python
edited = image_generation.edit_image(
    prompt="Change the sky to sunset",
    image="https://example.com/image.jpg"
)
print(edited)
```

**Use LoRAs:**
```python
loras = research.find_loras(query="cinematic")
print(loras)
# Then use in text_to_image with lora_id parameter
```

## Rules

1. Use `execute_code` to run Python code
2. Use `print()` to see results
3. Variables persist between executions
4. All methods are synchronous - do NOT use `await`
5. **IMPORTANT**: For tools marked with `[detailed docs]`, ALWAYS call `get_tool_documentation('tool.method')` first to read the full documentation before using them. These tools have specific instructions for best results.

## Documentation Helper

```python
# Read detailed documentation before using complex tools
docs = get_tool_documentation('image_generation.reference_image')
print(docs)
```

Be creative and help users generate stunning images and videos."""

    def _build_tool_documentation(self) -> str:
        """Build documentation for all available tools."""
        docs = []

        for namespace, tool_instance in self.tool_instances.items():
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
            - image: Image loading utilities
            - image_generation: Text-to-image, image-to-image, editing
            - prompt: Prompt creation and optimization
            - memory: Memory management
            - research: Web search and LoRA discovery
            - sessions: Session management
            - strategies: Creative strategies
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
                    ctx.deps.execution_globals[namespace] = AsyncToolWrapper(tool_instance, ctx.deps, namespace)

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

                ctx.deps.execution_globals["get_tools_readme"] = get_tools_readme
                ctx.deps.execution_globals["get_tool_documentation"] = get_tool_documentation
                ctx.deps.execution_globals["get_tool_method_documentation"] = get_tool_method_documentation

            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            # Get console for output display
            console = ctx.deps.console or Console()

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

                # Always display output panel
                output_text = result.strip() if result.strip() else "Code executed successfully"
                console.print(Panel(output_text, title="[dim]output[/dim]", border_style="dim"))

                return result.strip() if result.strip() else "Code executed successfully (no output)"

            except Exception:
                error_msg = f"Error executing code:\n{traceback.format_exc()}"
                console.print(Panel(error_msg, title="[dim]error[/dim]", border_style="red dim"))
                return error_msg

    def _get_dependencies(self) -> AgentDependencies:
        """Create dependencies for an agent run."""
        return AgentDependencies(
            config=self.config,
            session_id=self.session_id,
            session_manager=self.session_manager,
            tools=self.tool_instances,
            generated_outputs=self.generated_outputs,
            input_assets=self.input_assets,
            tools_used=self.tools_used,
            models_used=self.models_used,
            loras_used=self.loras_used,
            errors=self.errors,
            console=self.console,
        )

    def _display_tool_call(self, part: ToolCallPart) -> None:
        """Display a tool call with appropriate formatting."""
        if part.tool_name == "execute_code":
            # Handle different types of args
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
                self.console.print()
                syntax = Syntax(code, "python", theme="gruvbox-dark", line_numbers=False, word_wrap=True)
                self.console.print(Panel(syntax, title="[dim]code[/dim]", border_style="dim"))
        else:
            self.console.print(f"\n[dim]{part.tool_name}[/dim]")
            self.console.print(f"[dim]{part.args}[/dim]")

    @observe(name="okonomi-agent", capture_input=True, capture_output=True)
    async def run_async(self, user_message: str, reset: bool = False) -> str:
        """Run the agent asynchronously with a user message.

        Args:
            user_message: The user's message/task
            reset: Whether to reset conversation history

        Returns:
            The agent's response
        """
        if reset:
            self.message_history = []
            self.generated_outputs = []
            self.input_assets = []
            self.tools_used = set()
            self.models_used = set()
            self.loras_used = []
            self.errors = []
            self.token_usage = TokenUsage()

        self.task = user_message
        deps = self._get_dependencies()
        final_output = ""

        # Truncate message history if needed to prevent context overflow
        if self.message_history:
            truncated_history, summary = self.memory_manager.truncate(self.message_history)
            if summary:
                self.console.print(
                    f"[dim]Memory: Truncated {summary.truncated_count} messages "
                    f"(keeping {summary.retained_count})[/dim]"
                )
                self.message_history = truncated_history

        try:
            async with self.agent.iter(
                user_message,
                deps=deps,
                message_history=self.message_history if self.message_history else None,
            ) as agent_run:
                async for node in agent_run:
                    # Handle model request nodes - stream the text output
                    if Agent.is_model_request_node(node):
                        async with node.stream(agent_run.ctx) as stream:
                            accumulated_text = ""

                            # Use Live display for streaming text
                            with Live(
                                Spinner("dots", style="#d787af"),
                                console=self.console,
                                refresh_per_second=15,
                                transient=True,
                            ) as live:
                                async for text in stream.stream_text(delta=True, debounce_by=None):
                                    accumulated_text += text
                                    live.update(Text(accumulated_text, style="dim"))

                            # Print accumulated text if any
                            if accumulated_text.strip():
                                self.console.print(Text(accumulated_text, style="dim"))
                                final_output = accumulated_text

                            # Display tool calls from the response
                            response = stream.response
                            if response and response.parts:
                                for part in response.parts:
                                    if isinstance(part, ToolCallPart):
                                        self._display_tool_call(part)

                result = agent_run.result

                # Track token usage
                usage = agent_run.usage()
                self.token_usage.add(
                    input_tokens=usage.request_tokens or 0,
                    output_tokens=usage.response_tokens or 0,
                    requests=usage.requests or 1,
                )

            # Update message history for conversation continuity
            self.message_history = result.new_messages()
            self.generated_outputs = deps.generated_outputs

            # Display token usage
            self.console.print(f"[dim]{self.token_usage}[/dim]")

            # Return the final output (use result.output if we didn't capture it during streaming)
            return final_output if final_output else result.output

        except Exception as e:
            self.console.print(f"[dim red]Error: {e}[/dim red]")
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

    def _message_to_text(self, message: ModelMessage) -> str:
        """Convert a ModelMessage to readable text."""
        lines = []
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, TextPart):
                    lines.append(f"User: {part.content}")
                elif isinstance(part, ToolReturnPart):
                    content = str(part.content)[:500]
                    lines.append(f"Tool Result ({part.tool_name}): {content}")
        elif isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, TextPart):
                    lines.append(f"Assistant: {part.content}")
                elif isinstance(part, ToolCallPart):
                    args = str(part.args)[:200]
                    lines.append(f"Tool Call ({part.tool_name}): {args}")
        return "\n".join(lines)

    def _generate_steps_summary(self) -> str:
        """Generate LLM summary of session actions."""
        if not self.message_history:
            return ""

        try:
            conversation_text = "\n\n".join(self._message_to_text(msg) for msg in self.message_history)

            prompt = f"""Summarize the steps taken in this image/video generation session.
Focus on:
1. Tools used and their sequence
2. Key prompts and parameters
3. Any iterations or refinements
4. Final outcomes

Conversation:
{conversation_text}

Provide a concise summary (2-4 sentences):"""

            response = self.memory_manager.client.chat.completions.create(
                model=self.config.inference_providers.huggingface.prompt_model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return ""

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
        )

        session = Session(
            metadata=metadata,
            generated_outputs=self.generated_outputs,
            input_assets=self.input_assets,
            steps_summary=self._generate_steps_summary(),
            loras_used=self.loras_used,
            errors=self.errors,
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
        self.loras_used = session.loras_used
        self.errors = session.errors

        return True
