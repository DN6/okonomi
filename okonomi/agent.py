import json
import textwrap
import time
from collections import defaultdict
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from json import JSONDecodeError
from typing import List

import PIL
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from smolagents import (
    ActionOutput,
    ActionStep,
    AgentAudio,
    AgentGenerationError,
    AgentImage,
    AgentMaxStepsError,
    AgentParsingError,
    AgentToolExecutionError,
    ChatMessage,
    ChatMessageStreamDelta,
    FinalAnswerStep,
    LogLevel,
    MessageRole,
    PlanningStep,
    RunResult,
    SystemPromptStep,
    TaskStep,
    Timing,
    TokenUsage,
    ToolCall,
    ToolCallingAgent,
    ToolOutput,
    parse_json_if_needed,
    populate_template,
)

from .utils import write_to_permanent_memory


def _format_json_item(obj, level=0, separator=": ", indent="  "):
    """
    Helper function to format a JSON object or list with indentation.

    Args:
        obj: Object to format (dict, list, or primitive)
        level: Current indentation level
        separator: String between key and value
        indent: String for nested indentation

    Returns:
        str: Formatted text representation
    """
    lines = []
    prefix = indent * level

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_format_json_item(value, level + 1, separator, indent))
            else:
                lines.append(f"{prefix}{key}{separator}{value}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                lines.append(f"{prefix}[{i}]:")
                lines.append(_format_json_item(item, level + 1, separator, indent))
            else:
                lines.append(f"{prefix}- {item}")
    else:
        lines.append(f"{prefix}{obj}")

    return "\n".join(lines)


def json_to_simple_text(data, separator=": ", indent="  "):
    """
    Simple JSON to text converter with basic formatting.

    Args:
        data: JSON object or list of JSON objects
        separator: String between key and value
        indent: String for nested indentation

    Returns:
        str: Simple text representation
    """
    try:
        data = json.loads(data)
    except JSONDecodeError:
        return data

    if isinstance(data, list):
        blocks = []
        for i, item in enumerate(data):
            blocks.append(f"--- Item {i + 1} ---")
            blocks.append(_format_json_item(item, 0, separator, indent))
        return "\n\n".join(blocks)
    else:
        return _format_json_item(data, 0, separator, indent)


class CreateAgent(ToolCallingAgent):
    stop_sequences = ["Observation:", "Calling tools:"]
    eval_keys = ["prompt_concept_score", "visual_critique_score"]

    def __init__(
        self, structured_output: bool = False, tool_callbacks: dict = None, eval_keys: List[str] = None, **kwargs
    ) -> None:
        self.tool_callbacks = tool_callbacks or {}
        self.structured_output = structured_output
        self.eval_keys = eval_keys or self.eval_keys
        self.generated_outputs = []
        super().__init__(**kwargs)

    def write_memory_to_messages(
        self,
        summary_mode: bool = False,
    ) -> list[ChatMessage]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """

        # Keep system prompt so that mandatory instructions and tool definitions
        # are always in context
        messages = self.memory.system_prompt.to_messages(summary_mode=False)
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    def _build_initial_plan_messages(self, task):
        """Build messages for initial planning step."""
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
        """Build messages for plan update step."""
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
        """Extract token usage from a message."""
        if message.token_usage:
            return message.token_usage.input_tokens, message.token_usage.output_tokens
        return None, None

    def _generate_planning_step(self, task, is_first_step: bool, step: int) -> Generator[PlanningStep]:
        """Generate a planning step for the agent.

        Args:
            task: The task to plan for
            is_first_step: Whether this is the first planning step
            step: Current step number

        Returns:
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
        self.logger.log(Rule(f"[bold]{log_headline}", style="orange"), Text(plan), level=LogLevel.INFO)

        yield PlanningStep(
            model_input_messages=input_messages,
            plan=plan,
            model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content=plan_message_content),
            token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            timing=Timing(start_time=start_time, end_time=time.time()),
        )

    def _format_observation(self, tool_call_result):
        """Format the observation based on the tool call result type.

        Args:
            tool_call_result: The result from the tool call.

        Returns:
            str: Formatted observation string.
        """
        result_type = type(tool_call_result)

        if result_type == AgentImage:
            observation_name = "image.png"
            self.state[observation_name] = tool_call_result
            return f"Stored '{observation_name}' in memory."
        elif result_type == AgentAudio:
            observation_name = "audio.mp3"
            self.state[observation_name] = tool_call_result
            return f"Stored '{observation_name}' in memory."
        else:
            observation = str(tool_call_result).strip()
            if not self.structured_output:
                observation = json_to_simple_text(observation)
            return observation

    def _process_single_tool_call(self, tool_call: ToolCall) -> ToolOutput:
        """Process a single tool call and return its output.

        Args:
            tool_call: The tool call to process.

        Returns:
            ToolOutput: The result of the tool call.
        """
        tool_name = tool_call.name
        tool_callback_fn = self.tool_callbacks.get(tool_name, None)
        tool_arguments = tool_call.arguments or {}

        self.logger.log(
            Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
            level=LogLevel.INFO,
        )

        try:
            tool_call_result = self.execute_tool_call(tool_name, tool_arguments)
        except Exception as e:
            raise e

        if tool_callback_fn:
            tool_call_result = json.loads(tool_call_result)

            callback_kwargs = {}
            callback_kwargs.update(tool_arguments)
            callback_kwargs.update(tool_call_result)
            callback_results = tool_callback_fn(**callback_kwargs)

            tool_call_result.update(**json.loads(callback_results))

            if any(key in tool_call_result for key in self.eval_keys):
                concept_id = tool_call_result["concept_id"]
                generated_output = {}
                generated_output.update({"concept_id": concept_id, "tool_name": tool_name}, **tool_arguments)
                generated_output.update(**tool_call_result)
                self.generated_outputs.append(generated_output)

            tool_call_result = json.dumps(tool_call_result)

        observation = self._format_observation(tool_call_result)
        self.logger.log(
            f"Observations: \n\n{observation}",
            level=LogLevel.INFO,
        )
        return ToolOutput(
            id=tool_call.id,
            output=tool_call_result,
            is_final_answer=(tool_name == "final_answer"),
            observation=observation,
            tool_call=tool_call,
        )

    def _execute_tool_calls(self, tool_calls: dict[str, ToolCall]) -> dict[str, ToolOutput]:
        """Execute tool calls either sequentially or in parallel.

        Args:
            tool_calls: Dictionary of tool calls to execute.

        Returns:
            Dictionary of tool outputs.
        """
        outputs = {}

        if len(tool_calls) == 1:
            tool_call = list(tool_calls.values())[0]
            tool_output = self._process_single_tool_call(tool_call)
            outputs[tool_output.id] = tool_output
        else:
            with ThreadPoolExecutor(self.max_tool_threads) as executor:
                futures = [
                    executor.submit(self._process_single_tool_call, tool_call) for tool_call in tool_calls.values()
                ]
                for future in as_completed(futures):
                    tool_output = future.result()
                    outputs[tool_output.id] = tool_output

        return outputs

    def _aggregate_observations(self, outputs: dict[str, ToolOutput]) -> str:
        """Aggregate observations from tool outputs.

        Args:
            outputs: Dictionary of tool outputs.

        Returns:
            Aggregated observations string.
        """
        if not outputs:
            return ""

        observations = []
        for k in sorted(outputs.keys()):
            observations.append(outputs[k].observation)

        return "\n".join(observations)

    def process_tool_calls(
        self, chat_message: ChatMessage, memory_step: ActionStep
    ) -> Generator[ToolCall | ToolOutput]:
        """Process tool calls from the model output and update agent memory.

        Args:
            chat_message (`ChatMessage`): Chat message containing tool calls from the model.
            memory_step (`ActionStep)`: Memory ActionStep to update with results.

        Yields:
            `ToolCall | ToolOutput`: The tool call or tool output.
        """
        tool_calls: dict[str, ToolCall] = {}
        assert chat_message.tool_calls is not None

        for chat_tool_call in chat_message.tool_calls:
            tool_call = ToolCall(
                name=chat_tool_call.function.name, arguments=chat_tool_call.function.arguments, id=chat_tool_call.id
            )
            yield tool_call
            tool_calls[tool_call.id] = tool_call

        outputs = self._execute_tool_calls(tool_calls)
        for output in outputs.values():
            yield output
        memory_step.tool_calls = [tool_calls[k] for k in sorted(tool_calls.keys())]
        memory_step.observations = self._aggregate_observations(outputs)

    def _generate_model_response(self, input_messages, memory_step):
        """Generate response from the model, either streaming or non-streaming.

        Args:
            input_messages: Input messages for the model.
            memory_step: Memory step to update with model output.

        Returns:
            tuple: (ChatMessage, list of stream events)
        """
        try:
            chat_message = self.model.generate(
                input_messages,
                stop_sequences=self.stop_sequences,
                tools_to_call_from=self.tools_and_managed_agents,
            )
            log_content = str(
                chat_message.raw if chat_message.content is None and chat_message.raw else chat_message.content or ""
            )
            self.logger.log_markdown(
                content=log_content,
                title="Output message of the LLM:",
                level=LogLevel.DEBUG,
            )

            memory_step.model_output_message = chat_message
            memory_step.model_output = chat_message.content
            memory_step.token_usage = chat_message.token_usage

            return chat_message

        except Exception as e:
            raise AgentGenerationError(f"Error while generating output:\n{e}", self.logger) from e

    def _parse_tool_calls(self, chat_message):
        """Ensure tool calls are parsed from the chat message.

        Args:
            chat_message: The chat message to check and parse.

        Returns:
            ChatMessage: The chat message with parsed tool calls.
        """
        if not chat_message.tool_calls:
            try:
                chat_message = self.model.parse_tool_calls(chat_message)
            except Exception as e:
                raise AgentParsingError(f"Error while parsing tool call from model output: {e}", self.logger)

        for tool_call in chat_message.tool_calls:
            tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)

        return chat_message

    def _postprocess_response(self, chat_message, memory_step):
        """Process tool calls and collect the final answer if present.

        Args:
            chat_message: Chat message containing tool calls.
            memory_step: Memory step to update.

        Returns:
            dict: Contains 'outputs' list, 'answer', and 'got_answer' boolean.
        """
        outputs = []
        final_answer = None
        got_final_answer = False

        for output in self.process_tool_calls(chat_message, memory_step):
            outputs.append(output)

            if isinstance(output, ToolOutput) and output.is_final_answer:
                if got_final_answer:
                    raise AgentToolExecutionError(
                        "You returned multiple final answers. Please return only one single final answer!",
                        self.logger,
                    )

                final_answer = output.output
                got_final_answer = True

                if isinstance(final_answer, str) and final_answer in self.state:
                    final_answer = self.state[final_answer]

        return {"outputs": outputs, "answer": final_answer, "got_answer": got_final_answer}

    def _write_to_permanent_memory(self):
        concept_data = defaultdict(lambda: {"prompt_data": None, "image_generations": []})

        for entry in self.generated_outputs:
            concept_id = entry["concept_id"]
            if "prompt_concept_score" in entry:
                concept_data[concept_id]["prompt_data"] = {
                    "user_input": entry["user_input"],
                    "creative_strategy": entry["creative_strategy"],
                    "prompt_concept_score": entry["prompt_concept_score"],
                }
                continue

            concept_data[concept_id]["image_generations"].append(entry)

        merged_outputs = []
        for data in concept_data.values():
            prompt_data = data["prompt_data"] or {}
            image_generations = data["image_generations"]
            for image_generation in image_generations:
                image_generation.update(**prompt_data)
                merged_outputs.append(image_generation)

        write_to_permanent_memory(merged_outputs)

    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        memory_messages = self.write_memory_to_messages()
        input_messages = memory_messages.copy()
        memory_step.model_input_messages = input_messages

        chat_message = self._generate_model_response(input_messages, memory_step)

        chat_message = self._parse_tool_calls(chat_message)

        result = self._postprocess_response(chat_message, memory_step)
        for output in result["outputs"]:
            yield output

        yield ActionOutput(
            output=result["answer"],
            is_final_answer=result["got_answer"],
        )

    def _calculate_token_usage(self) -> TokenUsage | None:
        """Calculate total token usage from memory steps.

        Returns:
            TokenUsage object or None if token usage cannot be calculated.
        """
        total_input_tokens = 0
        total_output_tokens = 0

        for step in self.memory.steps:
            if isinstance(step, (ActionStep, PlanningStep)):
                if step.token_usage is None:
                    return None
                total_input_tokens += step.token_usage.input_tokens
                total_output_tokens += step.token_usage.output_tokens

        return TokenUsage(input_tokens=total_input_tokens, output_tokens=total_output_tokens)

    def _determine_run_state(self) -> str:
        """Determine the final state of the run.

        Returns:
            str: "max_steps_error" or "success"
        """
        if self.memory.steps and isinstance(getattr(self.memory.steps[-1], "error", None), AgentMaxStepsError):
            return "max_steps_error"
        return "success"

    def run(
        self,
        task: str,
        reset: bool = True,
        images: list["PIL.Image.Image"] | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
    ):
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in streaming mode.
                If `True`, returns a generator that yields each step as it is executed. You must iterate over this generator to process the individual steps (e.g., using a for loop or `next()`).
                If `False`, executes all steps internally and returns only the final answer after completion.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.
            additional_args (`dict`, *optional*): Any other variables that you want to pass to the agent run, for instance images or dataframes. Give them clear names!
            max_steps (`int`, *optional*): Maximum number of steps the agent can take to solve the task. if not provided, will use the agent's default value.

        Example:
        ```py
        from smolagents import CodeAgent
        agent = CodeAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        max_steps = max_steps or self.max_steps
        self.task = task
        self.interrupt_switch = False

        if additional_args:
            self.state.update(additional_args)
            self.task += f"""
You have been provided with these additional arguments, that you can access directly using the keys as variables:
{str(additional_args)}."""

        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
            self.monitor.reset()

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )
        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        run_start_time = time.time()
        steps = list(self._run_stream(task=self.task, max_steps=max_steps, images=images))
        assert isinstance(steps[-1], FinalAnswerStep)
        output = steps[-1].output

        self.logger.log("Saving session and generated outputs to permanent memory", level=LogLevel.INFO)
        try:
            self._write_to_permanent_memory()
        except Exception as e:
            self.logger.log(f"Error saving to memory: {e}")

        if self.return_full_result:
            return RunResult(
                output=output,
                token_usage=self._calculate_token_usage(),
                messages=self.memory.get_full_steps(),
                timing=Timing(start_time=run_start_time, end_time=time.time()),
                state=self._determine_run_state(),
            )

        return output
