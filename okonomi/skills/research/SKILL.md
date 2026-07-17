---
name: research
description: "Deep creative research and conceptual exploration upstream of media creation. Develops rich, well-researched visual concepts before any pixels are made — for AI image gen, photography, illustration, concept art, or video. Covers: brainstorming visual ideas, mood boards, style/movement exploration, reference mapping, concept briefs, prompt packages, palette/composition analysis, and art history deep dives. Trigger when users want to: brainstorm visuals, explore a style, build a mood board, develop a concept brief, craft image prompts, plan a photoshoot, flesh out a vague idea, or when they say 'I want something that looks like X.'"
metadata:
  tags: [concept, research, ideation, visual-language, prompt-crafting, mood-board, art-direction]
  related_skills: [genmedia]
---

# Research

Great media start with great thinking. Move from a vague feeling to a concrete, actionable visual concept — *then* generate.

The six-step process and the pitfalls below are each their own ``## H2``. Pull the section you need with `load_skill_section("research", "<slug>")` rather than the whole file. Deep lookups (composition vocabulary, color harmonies, art movements, creative-strategy techniques, output templates) live in `references/` — load on demand via `load_skill_reference("research", "<name>")`.

## When to use

- The user's brief is vague ("make something cool", "I want a vibe like X") and needs sharpening before generation.
- Brainstorming visual ideas, mood boards, style/movement exploration.
- Building a concept brief or a prompt package for a multi-image project.
- Reference mapping — finding specific artists / films / palettes that match a target.
- Art history or cross-cultural deep dives feeding into a project.
- Whenever the user names a style they want but you can't articulate what makes it work.

**Not for**: actual generation (do this *first*, then hand the brief to `genmedia`), pure technical questions (look up the relevant skill directly), one-shot prompt requests where the user is already specific.

## How to use

Six-step process; each is a drillable section:

1. **Listen and clarify** — surface intent, destination, emotional target, references, constraints. See `listen-and-clarify`.
2. **Check prior research** — `recall(query=...)` (one call covers conversation content, session titles, and media prompts) first; then `bash` over `$OKONOMI_HOME/research/`. See `check-prior-research`.
3. **Apply creative strategies** — push beyond the obvious (conceptual blending, forced connections, assumption reversal). See `apply-creative-strategies`.
4. **Research and explore** — `web_search` / `web_fetch` for artists, movements, references. Pursue visual language, reference mapping, cross-pollination. **For 2-4 genuinely independent strands (e.g. "films + comics + photography" for a noir brief), fan out via `launch_agent` instead of serial tool calls** — cuts a 5-min serial session to ~2 min of parallel work. See `research-and-explore` for the pattern.
5. **Synthesize and structure** — pick a format (concept brief, prompt package, exploration document, photography direction). See `synthesize-and-structure`.
6. **Save the research** — write self-contained markdown to `$OKONOMI_HOME/research/<slug>-concept-brief.md`. See `save-the-research`.

Vocabulary lookups (composition, color, lighting, art movements, creative-strategy techniques, output templates) live in `references/` — load on demand via `load_skill_reference("research", "<name>")`.

### Listen and clarify

Understand what the user is reaching for. Surface as needed:

| Dimension | Question |
|---|---|
| Intent | What's it for? |
| Destination | Which medium? (Flux, photography, video, illustration) |
| Emotional target | What should the viewer feel? |
| References | What do they like? What do they *not* want? |
| Constraints | Aspect ratio, brand guidelines, medium limits |

Infer what you can; ask about the rest. Bias toward asking rather than assuming.

### Check prior research

Before starting fresh, search for existing work. Use the cross-session memory ladder (cheap → fall-back):

1. `recall(query=...)` — BM25-ranked FTS5 over every saved conversation, session title, and artifact. Hits carry snippets and goal/resolution bookends. The right starting point for "have we explored X before?".
2. Saved research files on disk:
   ```python
   bash(command='ls $OKONOMI_HOME/research/')
   search(pattern='<keyword>', path='$OKONOMI_HOME/research/', file_glob='*.md')
   ```
   Read promising files to assess relevance.
3. `recall(session_id=...)` — drill into one promising hit for full metadata + artifacts + the conversation head/tail.

Try synonyms, related styles, artist names. If relevant prior work exists, surface it: *"You explored something similar when you worked on X — want to build from that?"*

### Apply creative strategies

Push the concept beyond the obvious before research begins. Quick reference:

| Strategy | Use when |
|---|---|
| Conceptual blending | Two domains can merge (Baroque + brutalism) |
| Replacement template | Symbolic substitution reveals hidden qualities |
| Forced connections | Need unexpected angles from distant domains |
| Assumption reversal | Flip fundamental premises to find new possibilities |
| Distance associations | Walk from obvious to surprising via semantic network |
| Perspective shifting | View through different lenses, scales, or timeframes |

Use one or combine several. The goal: a direction that's specific *and* surprising. Full technique writeups in `creative-strategies` reference.

### Research and explore

Let the concept's needs drive tool usage, not a fixed checklist. Err toward more research.

**Tools**
- `web_search` — artists, movements, techniques, cultural context, contemporary trends.
- `web_fetch` — full articles when search snippets aren't enough.
- `bash(command="open <url>")` or share image URLs in your reply — visual examples come back via okonomi's auto-attached multimodal pipeline so you can read them yourself.
- User-supplied files (mood boards, brand guidelines, prior briefs) — represent decisions already made and taste already established.

**Threads to pursue (as relevant)**
- **Visual language** — what makes a style work? Composition patterns, color relationships, texture, light behavior. Vocabulary in `visual-language` reference.
- **Reference mapping** — find specific artists, photographers, films, works. Describe what each reference *contributes*, not just "like this."
- **Cross-pollination** — unexpected connections across disciplines. A cyberpunk scene might learn from ukiyo-e composition.
- **User context integration** — existing materials anchor the search.

**Parallel fan-out for independent strands** — when the concept naturally
splits into 2-4 independent angles (e.g. *film cinematography* +
*illustration lineage* + *contemporary photography* for a "noir"
brief), dispatch one sub-agent per strand instead of researching them
serially. Cuts a 5-minute serial session to ~2 minutes of parallel
work, and each strand gets full attention from its own agent context
rather than the parent juggling angles.

Pattern — sub-agents write briefs to disk, parent reads + synthesizes:

```python
import os
session_dir = get_session_dir()
strands = [
    ("films", "Research 1940s film noir cinematography — lighting, framing, B&W tonality. Cite 3-5 specific films with shot examples."),
    ("comics", "Research noir illustration / comic art lineage (Sin City, Eisner, etc.). Line-work conventions, panel composition."),
    ("photo", "Research contemporary noir photographers. Modern reinterpretations of the noir aesthetic."),
]
for slug, prompt in strands:
    out_path = f"{session_dir}/research_strand_{slug}.md"
    full_prompt = (
        f"{prompt}\n\n"
        f"Use web_search / web_fetch. Produce a tight 200-word brief. "
        f"Save it as markdown to: {out_path}\n"
        f"End by printing the file path so the parent knows where to find it."
    )
    launch_agent(task=full_prompt, description=f"noir/{slug}")
```

After launching, poll `list_agents()` until all show `done`. Then read
the strand files, synthesize them into the final concept brief, and
proceed to generation as usual. Sub-agents share the session dir,
share the artifact registry, and auto-merge their tracked outputs back
when they finish.

Don't fan out when the strands are interdependent (each depends on
the previous strand's conclusion) — serial reasoning beats parallel
context-loss for that case. Reach for fan-out only when the angles
are genuinely independent.

### Synthesize and structure

Pick the output format that fits. Full templates in `output-formats` reference:

| Format | Use when |
|---|---|
| Concept brief | Complex or multi-image projects needing cohesive direction |
| Prompt package | AI image generation — a system for iteration, not just one prompt |
| Exploration document | Open-ended concepts with multiple possible directions |
| Photography direction | Planned shoots with shot lists and lighting direction |

### Save the research

Always save output as markdown to `$OKONOMI_HOME/research/` with a descriptive filename (e.g., `solarpunk-cityscape-concept-brief.md`). The file must be **self-contained** — someone reading it cold should understand the concept, references, and direction without needing the conversation.

```python
session_dir = get_session_dir()  # not used here, but available
import os
research_dir = os.path.expandvars("$OKONOMI_HOME/research")
os.makedirs(research_dir, exist_ok=True)
path = f"{research_dir}/<slug>-concept-brief.md"
with open(path, "w") as f:
    f.write(brief_markdown)
print(path)
```

## Pitfalls

- **Assuming instead of asking.** The first description is often a placeholder for something deeper — running with it skips the real intent. Surface it with targeted questions.
- **Asserting from memory.** Relying on recall instead of a search produces confident errors, especially for non-Western traditions, contemporary artists, and niche movements. Verify first.
- **Vague references.** A confidently wrong artist / film / movement is worse than none. Name real things.
- **Describing without showing.** Words alone leave the concept ambiguous — use image search to surface visual examples alongside the writeup.
- **Thinking in one-offs.** A brief that describes a single image, not a visual *system* that could generate many, doesn't scale to the project.
- **Generic over specific.** "Nice warm lighting" tells the model nothing; "warm golden hour light raking across textured concrete at a low angle" does.
- **Stopping at snippets.** Search previews often aren't enough — fetch the full article; the depth is in the details.
- **Stating the what without the why.** "Use complementary colors" without explaining what it does to the viewer's eye gives the user nothing to reason with.
- **Overriding the user's taste.** Research and suggest — don't impose your own direction.
- **Mis-scaling effort.** A quick prompt needs one search; a visual world needs multi-threaded research across all tools. Match the depth to the ask.

## References

Topic-deep material, available via `load_skill_reference("research", "<name>")`:

| Reference | Contents |
|---|---|
| `creative-strategies` | Conceptual blending, replacement templates, forced connections, assumption reversal, distance associations, perspective shifting — full writeups with examples. |
| `visual-language` | Composition (rule of thirds, golden ratio, leading lines, negative space), color harmonies (mono, analogous, complementary, split-comp, triadic), lighting (hard/soft, Rembrandt, butterfly, split, rim), texture vocabulary, art-movement glossary (Impressionism → Vaporwave → Solarpunk). |
| `output-formats` | Templates for concept brief, prompt package, exploration document, photography direction. |
