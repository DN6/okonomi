---
name: deep-concept-research
description: Deep creative research and conceptual exploration before image creation. Use for any visual medium — AI image gen (Flux, SD), photography, illustration, concept art. Trigger when the user wants to brainstorm visual ideas, build mood boards, explore styles or movements, research references, develop concept briefs, craft image prompts, plan photoshoots, or flesh out a vague visual idea. Also trigger for art history deep dives, palette/composition exploration, or when someone says "I want something that looks like X." If the user is thinking about an image before making it, use this skill.
---

# Deep Concept Research

Creative conceptual exploration and context establishment upstream of image creation. Develops rich, well-researched visual concepts before any pixels are made — whether the destination is an AI image generator, a camera, a canvas, or a brief for a human artist.
Great images start with great thinking. Move from a vague feeling or idea to a concrete, actionable visual concept.

## Process

### 1. Listen and Clarify

Understand what the user is reaching for. Often the first description is a placeholder for something deeper. Ask targeted questions — don't interrogate.
Surface as needed: **Intent** (what's it for), **Destination** (Flux, photography, etc.), **Emotional target** (what should the viewer feel), **Existing references** (likes and anti-references), **Constraints** (aspect ratio, brand guidelines, medium limits). Infer what you can, ask about the rest. Bias yourself towards asking questions, rather than making assumptions. 

### 2. Check Prior Research

Before starting fresh, search for existing concept research that might be relevant. Previous research files represent significant work — reuse and build on them rather than duplicating effort.

**Search saved research files** — Use `search` to look through `$OKONOMI_HOME/research/` for past concept briefs, prompt packages, and exploration documents:
1. `search(pattern="<keyword>", path="$OKONOMI_HOME/research/", file_glob="*.md")` with keywords from the current concept — style names, artist references, themes, moods. Try multiple queries: broad first (e.g., "cyberpunk"), then specific (e.g., "neon rain reflection").
2. Use `run(command="ls $OKONOMI_HOME/research/")` to list what research files exist if you need an overview.
3. Read promising files to assess relevance. A past brief on "solarpunk architecture" is highly relevant to a new "solarpunk fashion" request.

**Search past sessions** — Past sessions are stored in `~/.okonomi/sessions/` as JSON and JSONL files containing prompts, tool calls, outputs, and summaries. Use the standalone `search` tool to mine them:
1. `search(pattern="<keyword>", path="~/.okonomi/sessions/")` to grep across all session data for relevant concepts, prompts, artist names, style references, or techniques used previously.
2. `search(pattern="<keyword>", path="~/.okonomi/sessions/", file_glob="*.json")` to narrow results to session metadata files.
3. When you find a relevant session, use `run(command="cat ~/.okonomi/sessions/<session_id>/session.json")` to read its full details.

**Iterate on search** — Don't stop at one query. If the first search returns partial matches, refine: try synonyms, related styles, artist names mentioned in results. The goal is to exhaust what's already been explored before duplicating work.

If relevant prior work exists, surface it to the user: "You explored something similar when you worked on X — want to build from that?" Prior research can serve as a starting point, a set of established references to extend, or a record of directions already tried.

### 3. Apply Creative Strategies

Before diving into research, consider applying a creative strategy to push the concept beyond the obvious. See `creative_strategies.md` in this skill's directory for the full set of techniques — pick the one that fits the concept's needs:

- **Conceptual Blending** — Merge two unrelated domains (e.g., Baroque painting + brutalist architecture)
- **Replacement Template** — Symbolic substitution to reveal hidden qualities
- **Forced Connections** — Bridge distant domains for unexpected angles
- **Assumption Reversal** — Flip fundamental premises to find new possibilities
- **Distance Associations** — Walk the semantic network from obvious to surprising
- **Perspective Shifting** — View the concept through different lenses, scales, or timeframes

Use one or combine several. The goal is to arrive at a concept direction that's specific and surprising before research begins.

### 4. Research and Explore

Use available tools to build real context — don't rely solely on internalized knowledge. Let the concept's needs drive tool usage, not a fixed checklist. Err toward *more* research rather than less.

**Tools and when to reach for them:**

- **Web search**: Artists, movements, techniques, cultural context, contemporary trends, model-specific prompting guides. Use short, specific queries — "Saul Leiter color composition techniques" not "good photography styles."
- **Image search**: Visual examples of styles, palettes, compositions, lighting, reference imagery. Use liberally — this is a visual skill, showing beats telling.
- **Web fetch**: Full articles, artist retrospectives, technique breakdowns, prompt engineering guides. When a search snippet isn't enough, fetch the page.
- **User files (MCP tools)**: Mood boards, brand guidelines, previous briefs, inspiration folders from connected tools like Google Drive. The user's own material is often the most valuable context — check proactively when they reference existing work.

**Scale tool usage to complexity.** A quick prompt request might need one image search and one web search. A full visual world needs multi-threaded research across all tools.

**Research threads (pursue as relevant):**

**Visual language analysis** — Identify relevant traditions, movements, and trends. Break down what makes a style *work*: composition patterns, color relationships, texture, light behavior. Search for technique breakdowns; fetch detailed articles when the style has specific technical underpinnings.

**Reference mapping** — Find specific artists, photographers, films, or works in the same visual space. Search for and *show* visual examples. Describe what each reference contributes — not just "like this" but what specifically it teaches about handling light, space, texture, or mood. When the user names something unfamiliar, search rather than guessing.

**Cross-pollination** — Look for unexpected connections across disciplines: architecture, film, fashion, industrial design, nature. A concept rooted in Baroque painting might benefit from brutalist typography. A cyberpunk scene might learn from ukiyo-e composition.

**User context integration** — If the user has existing materials, pull them in via connected tools. Existing materials represent decisions already made and taste already established — anchor the research there.

### 5. Synthesize and Structure

Choose and combine output formats based on what the user needs:

**Concept Brief** — Full visual direction for complex or multi-image concepts:
Core Concept (emotional/conceptual intent) → Visual Direction (mood, color palette, lighting, texture, composition, movement) → Style References (with what each contributes) → Cultural/Historical Context → Technical Notes (medium-specific: prompt keywords, camera settings, illustration techniques).

**Prompt Package** — For AI image generation, not just a single prompt:
Core Prompt (ready-to-use, structured for target model) → Modifiers (swappable variations) → Negative Prompt (if applicable) → Parameter Suggestions (aspect ratio, style weight, model-specific settings) → Iteration Notes (how to evolve it).

**Exploration Document** — Freeform creative exploration:
Starting Point (the seed) → Threads (multiple directions, each with visual language and references) → Tensions and Contrasts (oppositions that drive visual interest) → Open Questions (prompts for further thinking).

**Photography Direction** — For planned shoots:
Concept → Shot List (framing, angle, subject) → Lighting Direction (quality, source, temperature) → Styling/Art Direction (wardrobe, props, location) → Post-Processing Direction → Mood References.

### 6. Save the Research

Always save the final output as a markdown file (`.md`) so the user can reference or reuse it later. Use a descriptive filename based on the concept — e.g., `solarpunk-cityscape-concept-brief.md`, `noir-portrait-prompt-package.md`.

Write the file to `$OKONOMI_HOME/research/` and present it to the user. The saved file should be self-contained: someone reading it cold (including the user in a future session) should understand the concept, references, and direction without needing the conversation for context.

- **Be specific over generic.** "Warm golden hour light raking across textured concrete at a low angle" beats "nice warm lighting."
- **Name real things.** Real artists, films, movements, techniques. When in doubt, search to confirm — a confidently wrong reference is worse than none.
- **Show, don't just describe.** Use image search to surface visual examples alongside written descriptions.
- **Go to the source.** Fetch full articles when search snippets aren't enough — the depth is in the details.
- **Research before you assert.** Verify with a search rather than relying on memory, especially for non-Western traditions, contemporary artists, and niche movements.
- **Explain the *why*.** Not just "use complementary colors" — explain what that does to the viewer's eye and why it serves this concept.
- **Respect the user's taste.** Research and suggest, don't override.
- **Think in systems.** A good brief describes a visual *system* that could generate many images, not just one.
- **Match the medium.** Search for current best practices — prompt syntax and model capabilities change frequently.
