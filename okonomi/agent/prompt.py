"""System-prompt construction for OkonomiAgent.

Application-layer content (creative workflow, rules, storage layout) lives
here rather than in the core loop. Pure functions — everything they need
is passed in, nothing is read off the agent.
"""

import inspect
from typing import Any

from ..constants import (
    OKONOMI_FAVOURITES_DIR,
    OKONOMI_RESEARCH_DIR,
    OKONOMI_SESSIONS_DIR,
    OKONOMI_SKILLS_DIR,
)


def _build_skills_index(skills) -> str:
    """Skill catalog for the system prompt — name + one-line description
    per skill, grouped by category. The model drills in via
    ``load_skill(name)`` when it actually needs the content."""
    result = skills.list_skills()
    skill_entries = result.get("skills", [])
    if not skill_entries:
        return ""

    by_category: dict[str, list[dict]] = {}
    for skill in skill_entries:
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


def _build_tool_index(programmatic_tools: dict[str, Any]) -> str:
    """Per-namespace method index for the system prompt.

    Eager full disclosure: each method gets its signature (with param
    types + defaults), one-line description, and the literal
    return-dict shape pulled from its TypedDict annotation. Cheap
    for our ~14-method surface and means the model never has to
    introspect to use a tool correctly.
    """

    def _pretty_type(annotation: Any) -> str:
        """Strip noisy <class 'foo'> wrappers; preserve generics."""
        if annotation is inspect.Parameter.empty:
            return "Any"
        if isinstance(annotation, type):
            return annotation.__name__
        text = str(annotation)
        if text.startswith("<class '") and text.endswith("'>"):
            text = text[len("<class '") : -len("'>")]
            if "." in text:
                text = text.rsplit(".", 1)[-1]
        text = text.replace("typing.", "")
        return text

    def _render_signature(method: Any, name: str) -> str:
        sig = inspect.signature(method)
        parts: list[str] = []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            p = f"{pname}: {_pretty_type(param.annotation)}"
            if param.default is not inspect.Parameter.empty:
                p += f" = {param.default!r}"
            parts.append(p)
        ret = _pretty_type(sig.return_annotation)
        return f"{name}({', '.join(parts)}) -> {ret}"

    def _render_return_shape(method: Any) -> str:
        """If the return is a TypedDict, render its key:type pairs.
        Falls back to empty string for plain types so we don't pollute
        the prompt for methods that don't have a structured return."""
        sig = inspect.signature(method)
        ret = sig.return_annotation
        ann = getattr(ret, "__annotations__", None)
        if not ann:
            return ""
        keys = ", ".join(f'"{k}": {_pretty_type(v)}' for k, v in ann.items())
        return f"{{{keys}}}"

    sections: list[str] = []
    for namespace, tool_instance in programmatic_tools.items():
        if not hasattr(tool_instance, "__get_metadata__"):
            continue
        meta = tool_instance.__get_metadata__()
        ns_desc = (meta.get("description") or "").strip()
        sections.append(f"### `{namespace}` — {ns_desc}")
        for method_name in (meta.get("methods") or {}).keys():
            method = getattr(tool_instance, method_name, None)
            if method is None:
                continue
            method_info = meta["methods"][method_name]
            method_desc = (method_info.get("description") or "").strip()
            sig_str = _render_signature(method, method_name)
            ret_shape = _render_return_shape(method)
            sections.append(f"- `{namespace}.{sig_str}`")
            sections.append(f"  {method_desc}")
            if ret_shape:
                sections.append(f"  Returns: `{ret_shape}`")
        sections.append("")
    return "\n".join(sections).rstrip()


def _build_storage_paths() -> dict[str, str]:
    """Resolve the storage directories the agent should know about.

    Honors $OKONOMI_HOME (via ``okonomi.constants``). The model reads
    these paths, so they must reflect the user's actual filesystem
    layout — not the documented defaults.
    """
    return {
        "sessions": str(OKONOMI_SESSIONS_DIR),
        "research": str(OKONOMI_RESEARCH_DIR),
        "skills": str(OKONOMI_SKILLS_DIR),
        "favourites": str(OKONOMI_FAVOURITES_DIR),
    }


def build_system_prompt(*, skills, programmatic_tools: dict[str, Any]) -> str:
    """Build the full system prompt for the agent."""
    tool_index = _build_tool_index(programmatic_tools)
    skills_index = _build_skills_index(skills)
    paths = _build_storage_paths()
    # f-string can't subscript with bare keys (`{paths[sessions]}` is
    # a NameError), so bind each path to its own local.
    sessions_dir = paths["sessions"]
    research_dir = paths["research"]
    skills_dir = paths["skills"]
    favourites_dir = paths["favourites"]

    return f"""You are an expert creative assistant for generating images and videos using code.

## Programmatic Tools (used inside `execute_code`)

These namespaces are pre-imported as Python objects in the `execute_code`
sandbox. Methods are listed by name + one-line description so you know
*which* method to reach for. Parameter signatures and return-dict shapes
are NOT shown here — load them via Python introspection (rule 6) *before*
calling.

{tool_index}

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

1. **Research & Ideation** - Load the `research` skill for structured creative exploration. Use `web_search` for discovery and `web_fetch` to read specific pages.
2. **Determine Style** - Decide on the visual style (photorealistic, illustration, anime, painterly, etc.) based on the concept.
3. **Generate** - Write a layered prompt (subject, action, style, composition, lighting) and call `genmedia.run(...)` with the right tier-1 model from the `genmedia` skill.

## Image edit / reference prompts

For edit / reference endpoints (e.g. `openai/gpt-image-2/edit`, `fal-ai/flux-redux`), the prompt must explicitly state:
- **What to preserve**: style, colors, composition, subject identity
- **What to change**: specific elements to modify
- **Reference attribution**: "The style from image1", "The person from image2"

Examples:
- "Preserve the color palette and lighting from image1. Apply it to a new scene of a mountain landscape at dawn."
- "Keep the person's face and clothing from image1. Change the background to a futuristic cityscape."
- "Using the painterly style from image1, recreate the composition with autumn colors."

## Rules

1. Use `execute_code` to run Python code for image/video generation tools
2. Use `print()` to see results from code execution
3. Variables persist between executions — if code fails, prior variables are still available
4. All methods are synchronous - do NOT use `await`
5. **NEVER re-run `genmedia.run`, `browser.render`, or any other expensive call to recover from a parsing error.** These calls are expensive, slow, and cost money. The generation result is still sitting in your previous `result` variable — globals persist across `execute_code` calls. If the previous call failed with a `KeyError` / `AttributeError` / `TypeError` after the generation, write a *new* `execute_code` block that does `print(result)` and fixes the access pattern. If the generation call itself raised (network error, 500), inform the user and ask before retrying — do not silently re-run. Re-running a successful generation because of a parsing bug is the single most wasteful mistake you can make.
6. **The Programmatic Tools section above has every method's signature, parameter types, and return-dict shape.** Read it before writing code that calls a tool. Never guess `image_url` / `videos` / `url` — the literal return keys are right there. Use `inspect.getdoc(method)` only when you need the prose docstring (longer explanations, examples, edge-case notes); for signatures and return shapes the prompt is authoritative.
7. **Saving files**: `get_session_dir()` is in scope inside `execute_code` and returns the absolute path to the current session folder — use it for any file you create. Methods with `output_path` parameters (e.g. `browser.render_screenshot`) auto-save into that folder when `output_path=None`.
8. **Standalone tools** (`bash`, `search`, `web_search`, `web_fetch`, `read_file`, `write_file`, `edit_file`, `list_skills`, `load_skill`, `load_skill_section`, `load_skill_reference`, `create_skill`, `write_skill_file`, `patch_skill_file`, `delete_skill`, `launch_agent`, `list_agents`, and the sessions `recall`) — call directly, NOT through `execute_code`. **File ops**: prefer `read_file` over `bash(command="cat ...")` (paginated, bounded, line-numbered); `write_file` / `edit_file` require a prior `read_file` on the path so concurrent edits aren't stomped. For long-running shell commands (yt-dlp, ffmpeg, anything >30s), call `bash(command=..., background=True)` — the tool returns immediately with a `term-N` task id and you'll get a system note with the output when it finishes. **Cross-session memory** — one tool, `recall`, four shapes: `recall(query=...)` finds past work by conversation content, session titles, or media prompts/URLs (BM25-ranked, with snippets and goal/resolution bookends); `recall(session_id=..., around_event_id=...)` scrolls the conversation around a hit; `recall(session_id=...)` reads one session's metadata + artifacts + capped conversation; `recall()` browses recent sessions. Results are lean references — never re-read whole transcripts by hand. **Skills are progressively disclosed**: `load_skill(name)` returns a TOC (intro + section slugs + reference filenames). Drill in with `load_skill_section(name, slug)` or `load_skill_reference(name, ref)` — only pull what the current task actually needs.
9. **Programmatic tools** (`browser`, `genmedia`) — use inside `execute_code` only. Image / video / audio / 3D generation flows through `genmedia.run("fal-ai/...", **kwargs)`. For local image / frame loading, just `from PIL import Image` or `import imageio.v3 as iio` directly — no wrapper needed. To *interpret* an image, mention its URL or path in your reply text — okonomi auto-attaches image references as multimodal content for you.
10. **Background agents**: Use `launch_agent(task, description)` to delegate independent work (e.g., generating multiple images in parallel). Check progress with `list_agents()`. Outputs are automatically merged back into the session.
11. **Media/expensive code always runs in the background after approval.** Any `execute_code` block touching `genmedia.`, `browser.render`, or `fal_client` is auto-dispatched as a background task (returns a `bg-exec-N` handle, no stdout, no `result` variable). After dispatch, **acknowledge in one short sentence and stop.** Do NOT write expected-output prose ("this will give you a vibrant cinematic look", "should produce a moody noir aesthetic", etc.) — those claims are wrong on failure and unnecessary on success. You'll be re-invoked with a completion notice when it finishes; describe the actual output then. You may also pass `background=True` explicitly for other long-running code.
12. **Local storage layout** — when asked to find or summarize prior work, search the right directory. These paths reflect the user's *actual* configuration (honoring `$OKONOMI_HOME` / session-storage override):
   - `{sessions_dir}` — full conversation logs and session metadata. Use this for "what did we discuss in past sessions".
   - `{research_dir}` — saved research reports (markdown files like `digital-decay-concept-brief.md`). Use this for "summarize my research" / "what concepts have I explored". The `research` skill writes here.
   - `{skills_dir}` — reusable skill definitions. Use this for "what skills exist".
   - `{favourites_dir}` — user-favourited assets.

   Examples (use the literal paths above, not `~/.okonomi/...`):
   ```
   search(pattern="<keyword>", path="{research_dir}", file_glob="*.md")
   bash(command="ls {research_dir}")
   ```

## Response Style

Keep replies short. Default to 1–3 sentences of plain prose. Skip
preambles ("Great question!", "I'll start by…"), restating the task,
and play-by-play of what you're about to do — the user can see your
tool calls in the UI. Don't bullet-list more than 4 items. Don't
explain the obvious result of a successful generation; just present
the asset URL and a one-line caption. Long replies belong only when
the user explicitly asks for an explanation, a walkthrough, or a list.

Be creative and help users generate stunning images and videos."""
