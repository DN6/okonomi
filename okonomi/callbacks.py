import asyncio
import json
import math
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import IntPrompt
from rich.text import Text
from smolagents import PlanningStep


# Import configuration
from .config import config

# Constants
DEFAULT_TIMEOUT = 300  # 5 minutes
MAX_IMAGE_COLUMNS = 2
DEFAULT_SCORE = 5
EDITED_PROMPT_SCORE = 10

# Model configuration
PROMPT_EVAL_MODEL_ID = config["inference_providers"]["huggingface"]["prompt_eval_model_id"]
concept_eval_client = InferenceClient(api_key=os.environ["HF_TOKEN"])
theme = gr.themes.Base(
    primary_hue="amber",
    radius_size="xxl",
)


def display_plan(plan_content):
    """Display the plan in a formatted way using rich"""
    console = Console()
    console.print("\n")
    console.print(
        Panel(
            Markdown(plan_content),
            title="[bold cyan]🤖 AGENT PLAN CREATED[/bold cyan]",
            border_style="bright_blue",
            expand=True,
        )
    )


def get_user_choice():
    """Get user's choice for plan approval using rich"""
    console = Console()
    console.print("\n[bold yellow]Choose an option:[/bold yellow]")
    console.print("[green]1.[/green] Approve plan")
    console.print("[yellow]2.[/yellow] Modify plan")
    console.print("[red]3.[/red] Cancel")

    while True:
        choice = IntPrompt.ask("\n[bold]Your choice[/bold]", choices=["1", "2", "3"])
        return choice


def get_modified_plan(original_plan):
    """Allow user to modify the plan using rich"""
    console = Console()

    console.print("\n")
    console.print(Panel("[bold yellow]MODIFY PLAN[/bold yellow]", border_style="yellow", expand=True))

    console.print("\n[bold]Current plan:[/bold]")
    console.print(Panel(Markdown(original_plan), border_style="dim", expand=True))

    console.print("\n[dim]Enter your modified plan (press Enter twice to finish):[/dim]")

    lines = []
    empty_line_count = 0

    while empty_line_count < 2:
        if len(lines) == 0:
            line = console.input("[bright_yellow]>>> [/bright_yellow]")
        else:
            line = console.input("[bright_yellow]... [/bright_yellow]")

        if line.strip() == "":
            empty_line_count += 1
        else:
            empty_line_count = 0
        lines.append(line)

    # Remove the last two empty lines
    modified_plan = "\n".join(lines[:-2])
    return modified_plan if modified_plan.strip() else original_plan


def interrupt_after_plan(memory_step, agent, autoplan=False):
    """
    Step callback that interrupts the agent after a planning step is created.
    Shows the plan using rich formatting and opens Gradio UI only if user wants to modify.
    """
    if not isinstance(memory_step, PlanningStep):
        return

    if autoplan:
        return

    console = Console()
    console.print("\n")
    console.print(Panel(Text("🛑 Agent Planning Step", style="bold cyan"), border_style="bright_blue", expand=True))

    # Display the created plan using rich formatting
    # display_plan(memory_step.plan)

    # Get user choice using rich formatting
    choice = get_user_choice()

    if choice == 1:  # Approve plan
        console.print("\n[bold green]✅ Plan approved! Continuing execution...[/bold green]\n")
        return

    elif choice == 2:  # Modify plan - Open Gradio UI
        console.print("\n[dim]Opening Gradio interface to modify plan...[/dim]\n")

        results_queue = _create_evaluation_queue()

        def run_gradio_interface():
            """Run the Gradio interface for plan modification"""
            loop = _create_async_event_loop()

            def create_interface():
                with gr.Blocks(title="Modify Agent Plan", theme=theme) as demo:
                    gr.Markdown("# ✏️ Modify Agent Plan")
                    gr.Markdown("Edit the plan below and click 'Apply Changes' when done.")

                    # Display the plan in an editable textbox
                    with gr.Row():
                        plan_field = gr.Textbox(
                            value=memory_step.plan,
                            label="Agent's Plan",
                            lines=20,
                            interactive=True,
                            info="✏️ Edit the plan as needed",
                            elem_classes=["monospace"],
                        )

                    # Store original plan for comparison
                    original_plan = gr.Textbox(value=memory_step.plan, visible=False)

                    # Status indicator
                    with gr.Row():
                        status = gr.Markdown("")

                    # Action buttons
                    with gr.Row():
                        apply_btn = gr.Button("✅ Apply Changes", variant="primary", size="lg")
                        cancel_btn = gr.Button("❌ Cancel Modifications", variant="stop", size="lg")

                    # Hidden output for results
                    output = gr.JSON(visible=False)

                    def check_modifications(edited_plan, original):
                        """Check if the plan has been modified"""
                        if edited_plan.strip() != original.strip():
                            return "✏️ **Plan has been modified**"
                        return "No changes detected yet"

                    def apply_modified_plan(plan_text, original):
                        """Handle modified plan submission"""
                        if plan_text.strip() == original.strip():
                            gr.Warning("No modifications detected. Please edit the plan or click 'Cancel'.")
                            return gr.update()

                        results = {"action": "modify", "plan": plan_text}
                        results_queue.put(results)
                        gr.Info("Changes applied! Continuing with updated plan...")
                        return results

                    def cancel_modifications():
                        """Handle cancellation - use original plan"""
                        results = {"action": "cancel_modify"}
                        results_queue.put(results)
                        gr.Info("Modifications cancelled. Using original plan.")
                        return results

                    # Set up event handlers
                    plan_field.change(fn=check_modifications, inputs=[plan_field, original_plan], outputs=status)

                    apply_btn.click(fn=apply_modified_plan, inputs=[plan_field, original_plan], outputs=output).then(
                        lambda: gr.update(visible=True), outputs=output
                    )

                    cancel_btn.click(fn=cancel_modifications, outputs=output).then(
                        lambda: gr.update(visible=True), outputs=output
                    )

                return demo

            try:
                demo = create_interface()
                demo.launch(inbrowser=True, prevent_thread_lock=True, quiet=True)

                # Keep the demo alive until a decision is made
                while results_queue.empty():
                    time.sleep(0.1)

                # Close the demo after results are received
                demo.close()

            except Exception as e:
                results_queue.put({"error": f"Failed to launch Gradio interface: {str(e)}"})
            finally:
                loop.close()

        # Start the Gradio interface in a separate thread
        _launch_gradio_in_thread(run_gradio_interface, results_queue)

        # Wait for user decision
        try:
            decision = results_queue.get(timeout=DEFAULT_TIMEOUT)

            if "error" in decision:
                console.print(f"[bold red]❌ Error with Gradio interface: {decision['error']}[/bold red]")
                console.print("[yellow]Falling back to console interface...[/yellow]\n")
                # Fall back to console interface for modification
                modified_plan = get_modified_plan(memory_step.plan)
                memory_step.plan = modified_plan
                console.print("\n[bold green]Plan updated![/bold green]")
                display_plan(modified_plan)
                console.print("[bold green]✅ Continuing with modified plan...[/bold green]\n")
                return

            # Process the Gradio UI decision
            action = decision.get("action")

            if action == "modify":
                modified_plan = decision.get("plan")
                memory_step.plan = modified_plan
                console.print("\n[bold green]✅ Plan updated![/bold green]")
                display_plan(modified_plan)
                console.print("[bold green]Continuing with modified plan...[/bold green]\n")
                return

            elif action == "cancel_modify":
                console.print("\n[yellow]Modifications cancelled. Using original plan.[/yellow]")
                console.print("[bold green]✅ Continuing with original plan...[/bold green]\n")
                return

        except queue.Empty:
            console.print(
                f"[yellow]⏱️ Timeout - no modifications received within {DEFAULT_TIMEOUT / 60:.0f} minutes[/yellow]"
            )
            console.print("[dim]Continuing with original plan...[/dim]\n")
            return
        except Exception as e:
            console.print(f"[bold red]❌ Error processing modifications: {str(e)}[/bold red]")
            console.print("[dim]Continuing with original plan...[/dim]\n")
            return

    elif choice == 3:  # Cancel
        console.print("\n[bold red]❌ Execution cancelled by user.[/bold red]\n")
        agent.interrupt()
        return


async def autoeval_prompt(input_prompt: str, candidate_prompt: str, creative_strategy: str) -> Dict[str, Any]:
    """
    Evaluate how well a candidate prompt generated using a creative strategy
    matches the intent and requirements of the original input prompt.

    Args:
        input_prompt: The original creative brief or problem statement
        candidate_prompt: The generated prompt using the creative strategy
        creative_strategy: The name of the creative strategy that was applied

    Returns:
        Dict containing evaluation score, reasoning, and recommendations
    """

    system_prompt = f"""You are an expert creative strategy evaluator. Your task is to assess how well a
    candidate prompt generated using a specific creative strategy aligns with and enhances the original input prompt.

        EVALUATION CRITERIA:

        1. INTENT PRESERVATION (25%):
        - Does the candidate maintain the core intent and goals of the input prompt?
        - Are the essential requirements and constraints preserved?
        - Is the fundamental creative challenge still addressed?

        2. STRATEGIC APPLICATION (25%):
        - Is the creative strategy ({creative_strategy}) applied correctly and effectively?
        - Does the strategy genuinely enhance the original prompt rather than distract from it?
        - Are the key principles of {creative_strategy} evident in the candidate?

        3. CREATIVE ENHANCEMENT (25%):
        - Does the candidate prompt open new creative possibilities?
        - Are fresh perspectives, insights, or approaches introduced?
        - Is there evidence of breakthrough thinking or novel connections?

        4. PRACTICAL UTILITY (25%):
        - Is the candidate prompt actionable and usable for creative work?
        - Does it provide clear direction while maintaining creative freedom?
        - Would this prompt inspire and guide effective creative exploration?

        SCORING SCALE (1-10):
        - 1-2: Poor - Misses intent, incorrect strategy application, no creative value
        - 3-4: Weak - Some intent preserved but strategy poorly applied or minimal enhancement
        - 5-6: Adequate - Basic intent maintained, strategy applied but limited creative impact
        - 7-8: Good - Intent preserved, strategy well-applied, clear creative enhancement
        - 9-10: Excellent - Intent elevated, masterful strategy use, significant creative breakthrough

        Be objective, constructive, and focus on both what works and what could be improved.
    """

    user_message = f"""Please evaluate this creative strategy application:

        ORIGINAL INPUT PROMPT:
        {input_prompt}

        CREATIVE STRATEGY USED:
        {creative_strategy}

        CANDIDATE PROMPT (generated using the strategy):
        {candidate_prompt}

        Provide your evaluation following the specified criteria and JSON format.
    """

    class ConceptEval(BaseModel):
        prompt_concept_score: int = Field(description="overall score for the provided candidate_prompt")
        recommendations: str = Field(
            description="1-2 suggestions on how to improve the candidate_prompt so that the core intent from the input_prompt is maintained"
        )

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "ConceptEval",
            "schema": ConceptEval.model_json_schema(),
        },
    }

    try:
        completion = concept_eval_client.chat.completions.create(
            model=PROMPT_EVAL_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=config["inference_providers"]["huggingface"]["prompt_eval_model_temperature"],
            max_tokens=config["inference_providers"]["huggingface"]["prompt_eval_max_tokens"],
            response_format=response_format,
        )
        response_content = completion.choices[0].message.content

        # Try to parse as JSON, fallback to structured text if needed
        try:
            evaluation = json.loads(response_content)
        except json.JSONDecodeError:
            evaluation = {
                "raw_response": response_content,
                "error": "Failed to parse JSON response",
            }

        return evaluation

    except Exception as e:
        return {
            "prompt_concept_score": None,
            "error": f"Evaluation failed: {str(e)}",
            "input_prompt": input_prompt,
            "candidate_prompt": candidate_prompt,
            "creative_strategy": creative_strategy,
        }


async def autoeval_image(candidate_prompt: str, image_url: str, loras: Optional[List] = None) -> Dict[str, Any]:
    """
    Auto-evaluate a generated image using AI vision model.

    Args:
        candidate_prompt: The prompt used for image generation
        image_url: URL or path to the generated image
        loras: List of LoRAs applied to the image

    Returns:
        Dict containing evaluation scores and recommendations
    """
    from pydantic import BaseModel, Field, computed_field

    system_prompt = """
    You are an expert visual art evaluator specializing in AI-generated imagery.
    Your task is to assess how well a generated image fulfills the original user intent and
    evaluate the technical and artistic quality of the result, including the effectiveness
    of any applied LoRA (Low-Rank Adaptation) models.

    SCORING SCALE (1-10):
    - 1-2: Poor - Major issues, doesn't meet basic requirements
    - 3-4: Below Average - Some elements work but significant problems
    - 5-6: Average - Acceptable quality with room for improvement
    - 7-8: Good - Strong execution with minor issues
    - 9-10: Excellent - Outstanding quality, exceeds expectations

    Be specific about what you observe in the image and provide constructive feedback.
    """

    # Format LoRA information for the prompt
    lora_info = ""
    if loras:
        lora_info = "\n\nLORAs APPLIED:\n"
        for i, lora in enumerate(loras, 1):
            lora_info += f"{i}. LoRA ID: {lora.get('lora_id', 'Unknown')}\n"
            lora_info += f"   Expected visual effect: {lora.get('description', 'No description provided')}\n"

    user_message = f"""Please evaluate this AI-generated image:

    CANDIDATE PROMPT USED:
    {candidate_prompt}{lora_info}

    Analyze the image according to the specified criteria and provide your evaluation in the requested JSON format.
    """

    class ImageEval(BaseModel):
        composition_score: int = Field(description="Composition quality score (1-10)")
        color_score: int = Field(description="Color mechanics score (1-10)")
        technical_execution_score: int = Field(description="Technical execution score (1-10)")
        intent_fulfillment_score: int = Field(description="Intent fulfillment score (1-10)")
        lora_effectiveness_score: int = Field(description="LoRA effectiveness score (1-10, or 10 if no LoRAs)")
        recommendations: str = Field(description="Comprehensive recommendations for improvement")

        @computed_field
        @property
        def visual_critique_score(self) -> int:
            scores = [
                self.composition_score,
                self.color_score,
                self.technical_execution_score,
                self.intent_fulfillment_score,
                self.lora_effectiveness_score,
            ]
            return sum(scores) // len(scores)

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "ImageEvaluation",
            "schema": ImageEval.model_json_schema(mode="serialization"),
        },
    }

    try:
        # Use the image evaluation model
        IMAGE_EVAL_MODEL_ID = config["inference_providers"]["huggingface"].get(
            "image_eval_model_id", "Qwen/Qwen2.5-VL-32B-Instruct"
        )

        completion = concept_eval_client.chat.completions.create(
            model=IMAGE_EVAL_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": user_message},
                    ],
                },
            ],
            temperature=0.2,
            max_tokens=1500,
            response_format=response_format,
        )
        response_content = completion.choices[0].message.content

        evaluation = json.loads(response_content)
        return evaluation

    except Exception as e:
        return {
            "error": f"Auto-evaluation failed: {str(e)}",
            "candidate_prompt": candidate_prompt,
            "image_url": image_url,
        }


def _create_evaluation_queue():
    """Create a queue for capturing Gradio evaluation results."""
    return queue.Queue()


def _create_async_event_loop():
    """Create and set a new asyncio event loop for the current thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def _launch_gradio_in_thread(interface_func, results_queue):
    """Launch a Gradio interface in a separate thread.

    Args:
        interface_func: Function that creates and runs the Gradio interface
        results_queue: Queue to capture results
    """
    gradio_thread = threading.Thread(target=interface_func)
    gradio_thread.daemon = True
    gradio_thread.start()
    return gradio_thread


def _launch_gradio_interface(demo, results_queue, timeout=DEFAULT_TIMEOUT):
    """Launch a Gradio interface and wait for results.

    Args:
        demo: The Gradio Blocks interface
        results_queue: Queue to capture results
        timeout: Maximum time to wait for results in seconds

    Returns:
        Dict containing evaluation results or error
    """
    try:
        demo.launch(inbrowser=True, prevent_thread_lock=True)
        evaluation_results = results_queue.get(timeout=timeout)
        demo.close()
        return evaluation_results
    except queue.Empty:
        return {"error": f"Evaluation timeout - no results received within {timeout / 60:.0f} minutes"}
    except Exception as e:
        return {"error": f"Error retrieving evaluation results: {str(e)}"}


def eval_image(
    images: str, prompt: str = "", lora_ids: Optional[List[Tuple[str, float]]] = None, autoeval: bool = False, **kwargs
) -> Dict:
    """
    Evaluate generated images with both manual user evaluation and optional AI auto-evaluation.
    Similar to eval_prompt but for images.

    Args:
        images: List of image URLs or file paths to evaluate
        prompt: The prompt used for image generation (for auto-eval)
        lora_ids: List of LoRAs applied (for auto-eval)
        autoeval: If True, skip Gradio interface and run auto-evaluation directly

    Returns:
        Dict containing evaluation results for all images
    """
    if not images or len(images) == 0:
        return json.dumps({"error": "No images provided"})

    # If autoeval is True, skip Gradio and run auto-evaluation directly
    if autoeval:
        # Create an async event loop for auto-evaluation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            evaluation_results = {}
            for idx, image_url in enumerate(images):
                # Run auto-evaluation for each image
                eval_result = loop.run_until_complete(
                    autoeval_image(candidate_prompt=prompt, image_url=image_url, loras=lora_ids or [])
                )

                evaluation_results["image_url"] = image_url
                evaluation_results["visual_critique_score"] = eval_result.get("visual_critique_score", 0)
                evaluation_results["recommendations"] = eval_result.get("recommendations", "")

            # Return as JSON string
            return json.dumps(evaluation_results)

        except Exception as e:
            return json.dumps({"error": f"Auto-evaluation failed: {str(e)}"})
        finally:
            loop.close()

    # Otherwise, continue with the Gradio interface
    results_queue = _create_evaluation_queue()

    def submit_evaluation(evaluation_data):
        """Handle the evaluation submission"""
        results_queue.put(evaluation_data)
        gr.Info("Evaluation submitted successfully!")
        return evaluation_data

    def run_gradio_interface():
        """Run the Gradio interface in a separate thread"""
        loop = _create_async_event_loop()

        async def auto_evaluate_image(image_url, current_scores, current_feedback):
            """Handle auto-evaluation for a specific image"""
            if not prompt:
                gr.Warning("Auto-eval requires candidate_prompt parameter")
                return current_scores, current_feedback, "❌ Missing candidate prompt for auto-evaluation"

            try:
                gr.Info("Running AI image evaluation...")

                # Call the auto-evaluation function
                eval_result = await autoeval_image(candidate_prompt=prompt, image_url=image_url, loras=lora_ids or [])

                if "error" in eval_result:
                    gr.Warning(f"Auto-evaluation error: {eval_result['error']}")
                    return current_scores, current_feedback, f"❌ Auto-evaluation failed: {eval_result['error']}"

                # Extract scores and recommendations
                visual_score = eval_result.get("visual_critique_score", 5)
                recommendations = eval_result.get("recommendations", "No recommendations provided.")

                formatted_feedback = (
                    f"🤖 AI Evaluation:\n\nVisual Score: {visual_score}/10\n\nRecommendations:\n{recommendations}"
                )

                gr.Info(f"AI evaluation complete! Score: {visual_score}/10")

                return visual_score, formatted_feedback, f"✅ AI evaluation complete - Score: {visual_score}/10"

            except Exception as e:
                gr.Warning(f"Auto-evaluation failed: {str(e)}")
                return current_scores, current_feedback, f"❌ Auto-evaluation error: {str(e)}"

        def create_interface():
            with gr.Blocks(title="Image Evaluation", theme=theme) as demo:
                gr.Markdown(
                    f"# Image Evaluation\nEvaluate {len(images)} generated image(s). You can score manually or use AI auto-evaluation."
                )

                # Show context if available
                if prompt:
                    with gr.Accordion("Generation Context", open=False):
                        gr.Markdown(f"**Prompt Used:** {prompt}")
                        if lora_ids:
                            lora_text = "\n".join(
                                [
                                    f"- {lora.get('lora_id', 'Unknown')}: {lora.get('description', '')}"
                                    for lora in lora_ids
                                ]
                            )
                            gr.Markdown(f"**LoRAs Applied:**\n{lora_text}")

                # Store evaluation data
                score_components = []
                feedback_components = []

                # Create image evaluation sections
                cols = min(MAX_IMAGE_COLUMNS, len(images))
                rows = math.ceil(len(images) / cols)

                idx = 0
                for _ in range(rows):
                    with gr.Row():
                        for _ in range(cols):
                            if idx < len(images):
                                with gr.Column():
                                    # Display image
                                    gr.ImageEditor(
                                        value=images[idx],
                                        label=f"Image {idx + 1}",
                                        type="filepath",
                                        interactive=False,
                                    )

                                    # Score slider
                                    score = gr.Slider(
                                        minimum=1,
                                        maximum=10,
                                        step=1,
                                        value=DEFAULT_SCORE,
                                        label=f"Score for Image {idx + 1}",
                                        info="Rate the overall quality",
                                    )

                                    # Feedback textbox
                                    feedback = gr.Textbox(
                                        label=f"Feedback for Image {idx + 1}",
                                        placeholder="What works well? What could be improved?",
                                        lines=3,
                                    )

                                    # Auto-eval button for this image
                                    if prompt:
                                        auto_eval_btn = gr.Button(
                                            f"🤖 Auto-Evaluate Image {idx + 1}",
                                            variant="secondary",
                                        )
                                        status = gr.Markdown("", visible=True)

                                        # Wrapper for async auto_evaluate
                                        def make_auto_eval_wrapper(img_idx):
                                            def wrapper(score_val, feedback_val):
                                                return loop.run_until_complete(
                                                    auto_evaluate_image(images[img_idx], score_val, feedback_val)
                                                )

                                            return wrapper

                                        auto_eval_btn.click(
                                            fn=make_auto_eval_wrapper(idx),
                                            inputs=[score, feedback],
                                            outputs=[score, feedback, status],
                                        )

                                    score_components.append(score)
                                    feedback_components.append(feedback)
                                    idx += 1

                # Submit button
                submit_btn = gr.Button("✅ Submit All Evaluations", variant="primary")
                output = gr.JSON(label="Evaluation Results", visible=False)

                # Create submission handler
                def handle_submission(*args):
                    results = {}
                    for i in range(len(images)):
                        results[f"image_{i + 1}"] = {
                            "image_path": images[i],
                            "score": int(args[i * 2]),
                            "feedback": args[i * 2 + 1],
                        }
                    return submit_evaluation(results)

                # Prepare inputs for submit button
                all_inputs = []
                for i in range(len(images)):
                    all_inputs.extend([score_components[i], feedback_components[i]])

                submit_btn.click(
                    fn=handle_submission,
                    inputs=all_inputs,
                    outputs=output,
                ).then(lambda: gr.update(visible=True), outputs=output)

            return demo

        try:
            demo = create_interface()
            demo.launch(inbrowser=True, prevent_thread_lock=True)
            while results_queue.empty():
                time.sleep(0.1)
            demo.close()

        except Exception as e:
            results_queue.put({"error": f"Failed to launch Gradio interface: {str(e)}"})
        finally:
            loop.close()

    _launch_gradio_in_thread(run_gradio_interface, results_queue)

    try:
        evaluation_results = results_queue.get(timeout=DEFAULT_TIMEOUT)

        # Return as JSON string
        return json.dumps(evaluation_results)

    except queue.Empty:
        return json.dumps(
            {"error": f"Evaluation timeout - no results received within {DEFAULT_TIMEOUT / 60:.0f} minutes"}
        )

    except Exception as e:
        return json.dumps({"error": f"Error retrieving evaluation results: {str(e)}"})


def eval_prompt(
    candidate_prompt: str, user_input: str = "", creative_strategy: str = "", autoeval: bool = False, **kwargs
) -> str:
    """
    Evaluate how well a candidate prompt generated using a creative strategy
    matches the intent and requirements of the original input prompt.

    Args:
        candidate_prompt: The candidate prompt to evaluate
        user_input: The original input prompt (optional, for auto-eval)
        creative_strategy: The creative strategy used (optional, for auto-eval)
        autoeval: If True, skip Gradio interface and run auto-evaluation directly

    Returns:
        str: JSON string containing the prompt, was_edited, score, and feedback
    """
    if not candidate_prompt or candidate_prompt.strip() == "":
        return json.dumps({"error": "No prompt provided"})

    # If autoeval is True, skip Gradio and run auto-evaluation directly
    if autoeval:
        if not user_input or not creative_strategy:
            return json.dumps({"error": "Auto-eval requires user_input and creative_strategy parameters"})

        # Create an async event loop for auto-evaluation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run auto-evaluation
            eval_result = loop.run_until_complete(
                autoeval_prompt(
                    input_prompt=user_input,
                    candidate_prompt=candidate_prompt.strip(),
                    creative_strategy=creative_strategy,
                )
            )

            if "error" in eval_result:
                return json.dumps({"error": eval_result["error"]})

            # Format the result to match interactive output format
            result = {
                "candidate_prompt": candidate_prompt.strip(),
                "was_edited": False,  # Not edited in auto-eval
                "prompt_concept_score": eval_result.get("prompt_concept_score", 0),
                "feedback": eval_result.get("recommendations", "No recommendations provided."),
            }

            return json.dumps(result)

        except Exception as e:
            return json.dumps({"error": f"Auto-evaluation failed: {str(e)}"})
        finally:
            loop.close()

    # Otherwise, continue with the Gradio interface
    results_queue = _create_evaluation_queue()

    # Store the original prompt for comparison
    original_prompt = candidate_prompt

    def submit_evaluation(edited_prompt, score, feedback_text):
        """Handle the evaluation submission"""
        final_prompt = edited_prompt.strip()
        was_edited = final_prompt != original_prompt.strip()

        results = {
            "candidate_prompt": final_prompt,
            "was_edited": was_edited,
            "prompt_concept_score": int(score),
            "feedback": feedback_text,
        }

        # Put results in queue for the main thread to retrieve
        results_queue.put(results)

        # Close the interface after submission
        gr.Info("Evaluation submitted successfully!")

        return results

    def run_gradio_interface():
        """Run the Gradio interface in a separate thread"""
        loop = _create_async_event_loop()

        async def auto_evaluate(edited_prompt, current_score, current_feedback):
            """Handle auto-evaluation using eval_concept"""
            if not user_input or not creative_strategy:
                gr.Warning("Auto-eval requires input_prompt and creative_strategy parameters")
                return (
                    current_score,
                    "Auto-evaluation requires the original input prompt and creative strategy to be provided.",
                    "❌ Missing required parameters for auto-evaluation",
                )

            try:
                # Show loading status
                gr.Info("Running AI evaluation...")

                # Call the eval_concept function
                eval_result = await autoeval_prompt(
                    input_prompt=user_input,
                    candidate_prompt=edited_prompt.strip(),
                    creative_strategy=creative_strategy,
                )

                if "error" in eval_result:
                    gr.Warning(f"Auto-evaluation error: {eval_result['error']}")
                    return (
                        current_score,
                        current_feedback,
                        f"❌ Auto-evaluation failed: {eval_result['error']}",
                    )

                # Extract score and recommendations
                score = eval_result.get("prompt_concept_score", 5)
                recommendations = eval_result.get("recommendations", "No recommendations provided.")

                gr.Info(f"AI evaluation complete! Score: {score}/10")

                return (
                    score,
                    recommendations,
                    f"✅ AI evaluation complete - Score: {score}/10",
                )

            except Exception as e:
                gr.Warning(f"Auto-evaluation failed: {str(e)}")
                return (
                    current_score,
                    current_feedback,
                    f"❌ Auto-evaluation error: {str(e)}",
                )

        def create_interface():
            with gr.Blocks(title="Prompt Evaluation", theme=theme) as demo:
                gr.Markdown(
                    "# Prompt Evaluation\nYou can edit the prompt to improve it, score it manually, or use AI auto-evaluation."
                )

                # Show context if available
                if user_input and creative_strategy:
                    with gr.Accordion("Context", open=False):
                        gr.Markdown(f"**Original Input:** {user_input}")
                        gr.Markdown(f"**Creative Strategy:** {creative_strategy}")

                # Editable prompt field
                with gr.Row():
                    prompt_field = gr.Textbox(
                        value=candidate_prompt,
                        label="Candidate Prompt (Editable)",
                        lines=4,
                        interactive=True,
                        info="✏️ Feel free to edit and improve the prompt",
                    )

                # Hidden field to store original prompt
                original_prompt_store = gr.Textbox(value=original_prompt, visible=False)

                # Score slider
                score = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=5,
                    label="Score",
                    info="Score will automatically become 10 if you edit the prompt manually",
                )

                # Feedback textbox
                feedback = gr.Textbox(
                    label="Feedback",
                    placeholder="What works well? What could be improved?",
                    lines=5,
                    info="Feedback will be cleared if you edit the prompt manually",
                )

                # Status message
                edit_status = gr.Markdown("", visible=True)

                # Buttons row
                with gr.Row():
                    auto_eval_btn = gr.Button(
                        "🤖 Auto-Evaluate with AI",
                        variant="secondary",
                        visible=bool(user_input and creative_strategy),
                    )
                    submit_btn = gr.Button("✅ Submit Evaluation", variant="primary")

                output = gr.Textbox(label="Evaluation Results", visible=False)

                # Set up change handler for prompt editing
                def check_edit_and_update(edited, original):
                    if edited.strip() != original.strip():
                        return (
                            EDITED_PROMPT_SCORE,  # score
                            "",  # feedback
                            "✅ Prompt edited - Score set to 10",  # status
                        )
                    else:
                        return (
                            gr.update(),  # score - no change
                            gr.update(),  # feedback - no change
                            "",  # status - clear
                        )

                prompt_field.change(
                    fn=check_edit_and_update,
                    inputs=[prompt_field, original_prompt_store],
                    outputs=[score, feedback, edit_status],
                )

                # Wrapper for async auto_evaluate
                def auto_eval_wrapper(edited_prompt, current_score, current_feedback):
                    return loop.run_until_complete(auto_evaluate(edited_prompt, current_score, current_feedback))

                # Set up auto-eval button
                auto_eval_btn.click(
                    fn=auto_eval_wrapper,
                    inputs=[prompt_field, score, feedback],
                    outputs=[score, feedback, edit_status],
                )

                # Set up click handler for submission
                submit_btn.click(
                    fn=submit_evaluation,
                    inputs=[prompt_field, score, feedback],
                    outputs=output,
                ).then(lambda: gr.update(visible=True), outputs=output)

            return demo

        try:
            demo = create_interface()
            demo.launch(inbrowser=True, prevent_thread_lock=True)

            # Keep the demo alive until results are submitted
            while results_queue.empty():
                time.sleep(0.1)

            # Close the demo after results are received
            demo.close()

        except Exception as e:
            results_queue.put({"error": f"Failed to launch Gradio interface: {str(e)}"})
        finally:
            loop.close()

    # Start the Gradio interface in a separate thread
    _launch_gradio_in_thread(run_gradio_interface, results_queue)

    # Wait for results from the queue with a timeout
    try:
        evaluation_results = results_queue.get(timeout=DEFAULT_TIMEOUT)
        # Return as JSON string
        return json.dumps(evaluation_results)

    except queue.Empty:
        return json.dumps(
            {"error": f"Evaluation timeout - no results received within {DEFAULT_TIMEOUT / 60:.0f} minutes"}
        )
    except Exception as e:
        return json.dumps({"error": f"Error retrieving evaluation results: {str(e)}"})
