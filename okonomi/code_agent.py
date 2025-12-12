import json
import textwrap
import time
from collections.abc import Generator
from typing import List

from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from smolagents import (
    ChatMessage,
    CodeAgent,
    LogLevel,
    MessageRole,
    PlanningStep,
    Timing,
    TokenUsage,
    populate_template,
)

from .session import SessionManager
from .utils import write_to_permanent_memory


class OkonomiCodeAgent(CodeAgent):
    """
    Extended CodeAgent with clean method organization for customizing agent behavior.

    This agent inherits from smolagents.CodeAgent and provides:
    - Clean method structure for overriding behavior
    - Session management integration
    - Permanent memory writing
    - Custom planning logic
    - Tool execution tracking
    - MCP server tool injection via python executor
    """

    stop_sequences = ["Observation:", "<end_action>"]
    eval_keys = ["prompt_concept_score", "visual_critique_score"]

    def __init__(
        self,
        eval_keys: List[str] = None,
        session_id: str = None,
        server_tools: dict = None,
        **kwargs,
    ) -> None:
        """Initialize the OkonomiCodeAgent.

        Args:
            eval_keys: List of keys to track for evaluation metrics
            session_id: Optional session ID for loading/saving sessions
            server_tools: Dict of tool instances from MCP server (namespace -> tool_instance)
                         e.g., {"image_generation": ImageGenerationTools(...), "prompt": PromptTools(...)}
            **kwargs: Additional arguments passed to CodeAgent
        """
        self.eval_keys = eval_keys or self.eval_keys
        self.generated_outputs = []
        self.session_manager = SessionManager()
        self.session_id = session_id or self.session_manager.create_session_id()
        self.run_start_time = None
        self.server_tools = server_tools or {}

        super().__init__(**kwargs)

    # ================================================================================
    # Memory and Message Management
    # ================================================================================

    def write_memory_to_messages(
        self,
        summary_mode: bool = False,
    ) -> list[ChatMessage]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM.

        Args:
            summary_mode: Whether to use summary mode for messages

        Returns:
            List of ChatMessage objects
        """
        # Keep system prompt so that mandatory instructions and tool definitions
        # are always in context
        messages = self.memory.system_prompt.to_messages(summary_mode=False)
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    # ================================================================================
    # Planning Methods
    # ================================================================================

    def _build_initial_plan_messages(self, task):
        """Build messages for initial planning step.

        Args:
            task: The task to plan for

        Returns:
            List of ChatMessage objects for initial planning
        """
        return [
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["initial_plan"],
                            variables={
                                "task": task,
                                "tools": self.tools,
                                "managed_agents": self.managed_agents,
                                "planning_interval": self.planning_interval,
                            },
                        ),
                    }
                ],
            )
        ]

    def _build_update_plan_messages(self, task, step):
        """Build messages for plan update step.

        Args:
            task: The task to plan for
            step: Current step number

        Returns:
            List of ChatMessage objects for plan update
        """
        memory_messages = self.write_memory_to_messages(summary_mode=True)

        plan_update_pre = ChatMessage(
            role=MessageRole.SYSTEM,
            content=[
                {
                    "type": "text",
                    "text": populate_template(
                        self.prompt_templates["planning"]["update_plan_pre_messages"], variables={"task": task}
                    ),
                }
            ],
        )

        plan_update_post = ChatMessage(
            role=MessageRole.USER,
            content=[
                {
                    "type": "text",
                    "text": populate_template(
                        self.prompt_templates["planning"]["update_plan_post_messages"],
                        variables={
                            "task": task,
                            "tools": self.tools,
                            "planning_interval": self.planning_interval,
                            "managed_agents": self.managed_agents,
                            "remaining_steps": (self.max_steps - step),
                        },
                    ),
                }
            ],
        )

        return [plan_update_pre] + memory_messages + [plan_update_post]

    def _extract_token_usage(self, message):
        """Extract token usage from a message.

        Args:
            message: The message to extract token usage from

        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        if message.token_usage:
            return message.token_usage.input_tokens, message.token_usage.output_tokens
        return None, None

    def _generate_planning_step(self, task, is_first_step: bool, step: int) -> Generator[PlanningStep]:
        """Generate a planning step for the agent.

        Args:
            task: The task to plan for
            is_first_step: Whether this is the first planning step
            step: Current step number

        Yields:
            PlanningStep containing the plan and metadata
        """
        start_time = time.time()

        if is_first_step:
            input_messages = self._build_initial_plan_messages(task)
            plan_template = "Here are the facts I know and the plan of action that I will follow to solve the task:\n```\n{plan_content}\n```"
        else:
            input_messages = self._build_update_plan_messages(task, step)
            plan_template = "I still need to solve the task I was given:\n```\n{task}\n```\n\nHere are the facts I know and my new/updated plan of action to solve the task:\n```\n{plan_content}\n```"

        plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
        plan_message_content = plan_message.content

        input_tokens, output_tokens = self._extract_token_usage(plan_message)
        plan = textwrap.dedent(
            plan_template.format(task=self.task if not is_first_step else "", plan_content=plan_message_content)
        )

        log_headline = "Initial plan" if is_first_step else "Updated plan"
        self.logger.log(Rule(f"{log_headline}", style="orange"), Text(plan), level=LogLevel.INFO)

        yield PlanningStep(
            model_input_messages=input_messages,
            plan=plan,
            model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content=plan_message_content),
            token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            timing=Timing(start_time=start_time, end_time=time.time()),
        )

    # ================================================================================
    # Code Execution Methods
    # ================================================================================

    def _format_code_output(self, output):
        """Format code execution output.

        Args:
            output: The output from code execution

        Returns:
            Formatted output string
        """
        # Override this method to customize code output formatting
        return str(output).strip()

    def _track_code_execution(self, code: str, output: str, error: str = None):
        """Track code execution for analytics/debugging.

        Args:
            code: The code that was executed
            output: The output from execution
            error: Any error that occurred
        """
        # Override this method to add custom tracking
        pass

    # ================================================================================
    # Session and Memory Management
    # ================================================================================

    def _save_session(self, status: str, merged_outputs: list) -> None:
        """Save the current session state.

        Args:
            status: Status of the session (e.g., "completed", "error", "max_steps")
            merged_outputs: List of merged outputs to save
        """
        from datetime import datetime
        from .session import Session, SessionMetadata

        # Get model ID if available
        model_id = ""
        if hasattr(self, "model") and hasattr(self.model, "model_id"):
            model_id = self.model.model_id

        # Create session metadata
        metadata = SessionMetadata(
            session_id=self.session_id,
            task=self.task if hasattr(self, "task") else "",
            timestamp=datetime.now().isoformat(),
            status=status,
            num_outputs=len(merged_outputs),
            model_id=model_id,
            total_steps=len(self.memory.steps),
        )

        # Create session object
        session = Session(
            metadata=metadata,
            generated_outputs=merged_outputs,
            agent_memory_state=None,  # Could serialize memory if needed for resumption
        )

        # Save session
        self.session_manager.save_session(session)

    def load_from_session(self, session_id: str) -> bool:
        """Load agent state from a saved session.

        Args:
            session_id: ID of the session to load

        Returns:
            bool: True if session was loaded successfully, False otherwise
        """
        session = self.session_manager.load_session(session_id)

        if not session:
            return False

        # Restore session state
        self.session_id = session_id
        self.task = session.metadata.task
        self.generated_outputs = session.generated_outputs

        # Note: We don't restore memory state by default to allow continuing with new tasks
        # If full resumption is needed, agent_memory_state could be used to restore memory

        self.logger.log(
            f"Loaded session {session_id}",
            level=LogLevel.INFO,
        )
        return True

    # ================================================================================
    # Run Methods
    # ================================================================================

    def _calculate_token_usage(self) -> TokenUsage | None:
        """Calculate total token usage across all steps.

        Returns:
            TokenUsage object with totals, or None if no usage tracked
        """
        total_input_tokens = 0
        total_output_tokens = 0

        for step in self.memory.steps:
            if step.token_usage:
                if step.token_usage.input_tokens:
                    total_input_tokens += step.token_usage.input_tokens
                if step.token_usage.output_tokens:
                    total_output_tokens += step.token_usage.output_tokens

        if total_input_tokens > 0 or total_output_tokens > 0:
            return TokenUsage(input_tokens=total_input_tokens, output_tokens=total_output_tokens)
        return None

    def _determine_run_state(self) -> str:
        """Determine the final state of the run.

        Returns:
            String representing the run state ("completed", "max_steps", "error")
        """
        if self.memory.steps and hasattr(self.memory.steps[-1], "error") and self.memory.steps[-1].error:
            return "error"
        elif len(self.memory.steps) >= self.max_steps:
            return "max_steps"
        return "completed"

    def _inject_server_tools(self):
        """Inject server tool instances directly into the python executor.

        This makes the tool instances available as global variables in the code execution
        environment, allowing the agent to call them naturally like:

            result = image_generation.text_to_image(prompt="...", concept_id="...")

        Since the Python executor can't run async code, we wrap async methods with
        sync wrappers that run them in an event loop.
        """
        if not hasattr(self, "python_executor"):
            return

        import asyncio

        # Create sync wrapper classes for each tool
        wrapped_tools = {}

        for namespace, tool_instance in self.server_tools.items():
            # Create a wrapper class that converts async methods to sync
            class ToolWrapper:
                def __init__(self, tool_instance):
                    self._tool_instance = tool_instance

                def __getattr__(self, name):
                    attr = getattr(self._tool_instance, name)

                    # If it's an async method, wrap it to run synchronously
                    if asyncio.iscoroutinefunction(attr):

                        def sync_wrapper(*args, **kwargs):
                            try:
                                loop = asyncio.get_running_loop()
                                # We're in an async context, can't use run_until_complete
                                import concurrent.futures

                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(asyncio.run, attr(*args, **kwargs))
                                    return future.result()
                            except RuntimeError:
                                # No running loop, create one
                                return asyncio.run(attr(*args, **kwargs))

                        return sync_wrapper

                    return attr

            wrapped_tools[namespace] = ToolWrapper(tool_instance)

        # Send wrapped tool instances as variables to the executor
        self.python_executor.send_variables(wrapped_tools)

    def run(self, task: str, stream: bool = False, reset: bool = True, **kwargs):
        """
        Run the agent on a task with session management and server tool injection.

        Args:
            task: The task to execute
            stream: Whether to stream the execution
            reset: Whether to reset memory before running
            **kwargs: Additional arguments

        Returns:
            RunResult with execution details
        """
        self.run_start_time = time.time()

        # Inject server tools into python executor before running
        if self.server_tools and hasattr(self, "python_executor"):
            self._inject_server_tools()

        try:
            # Call parent run method
            result = super().run(task=task, stream=stream, reset=reset, **kwargs)

            # Post-processing
            run_state = self._determine_run_state()
            self._save_session(status=run_state, merged_outputs=[])

            return result

        except Exception as e:
            # Handle errors and save session
            self._save_session(status="error", merged_outputs=[])
            raise e
