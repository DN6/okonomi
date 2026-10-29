---
name: research
description: "Deep creative research and conceptual exploration upstream of media creation. Develops rich, well-researched visual concepts before any pixels are made — for AI image gen, photography, illustration, concept art, or video. Covers: brainstorming visual ideas, mood boards, style/movement exploration, reference mapping, concept briefs, prompt packages, palette/composition analysis, and art history deep dives. Trigger when users want to: brainstorm visuals, explore a style, build a mood board, develop a concept brief, craft image prompts, plan a photoshoot, flesh out a vague idea, or when they say 'I want something that looks like X.'"
metadata:
  tags: [concept, research, ideation, visual-language, prompt-crafting, mood-board, art-direction]
---

# Research

Great media start with great thinking. Move from a vague feeling to a concrete, actionable visual concept — *then* generate.

The six-step process and the pitfalls below are each their own ``## H2``. Pull the section you need with `load_skill_section("research", "<slug>")` rather than the whole file. Deep lookups (composition vocabulary, color harmonies, art movements, output templates) live in `references/` — load on demand via `load_skill_reference("research", "<name>")`.

## When to use

- The user's brief is vague ("make something cool", "I want a vibe like X") and needs sharpening before generation.
- Brainstorming visual ideas, mood boards, style/movement exploration.
- Building a concept brief or a prompt package for a multi-image project.
- Reference mapping — finding specific artists / films / palettes that match a target.
- Art history or cross-cultural deep dives feeding into a project.
- Whenever the user names a style they want but you can't articulate what makes it work.

**Not for**: actual generation (do this *first*, then hand the brief off for generation), pure technical questions (look up the relevant skill directly), one-shot prompt requests where the user is already specific.

## How to use

Six-step process; each is a drillable section:

1. **Listen and clarify** — surface intent, destination, emotional target, references, constraints. See `listen-and-clarify`.
2. **Check prior research** — `recall(query=...)` (one call covers conversation content, session titles, media prompts, and mind wiki pages) first; then walk the `[[wikilinks]]` of any mind page it surfaces. See `check-prior-research`.
3. **Apply creative strategies** — only when the brief is open-ended or the first direction is obvious; skip for specific briefs. See `apply-creative-strategies`.
4. **Research and explore** — `web_search` / `web_fetch` for artists, movements, references. Pursue visual language, reference mapping, cross-pollination. **For 2-4 genuinely independent strands (e.g. "films + comics + photography" for a noir brief), fan out via `launch_agent` instead of serial tool calls** — cuts a 5-min serial session to ~2 min of parallel work. See `research-and-explore` for the pattern.
5. **Synthesize and structure** — pick a format (concept brief, prompt package, exploration document, photography direction). See `synthesize-and-structure`.
6. **Save the research** — file findings into the mind wiki with `remember(...)` as interlinked pages. See `save-the-research`.

Vocabulary lookups (composition, color, lighting, art movements, output templates) live in `references/` — load on demand via `load_skill_reference("research", "<name>")`.

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

1. `recall(query=...)` — BM25-ranked FTS5 over every saved conversation, session title, artifact, and mind wiki page. Hits carry snippets; conversation hits add goal/resolution bookends, mind hits a `path`. The right starting point for "have we explored X before?".
2. The mind wiki — for a mind hit, `read_file` its `path`, then **walk its `[[wikilinks]]`** — neighboring pages hold related concepts that share no keywords with your query. Backlinks work too: `search(pattern='\[\[<slug>\]\]', path='$OKONOMI_HOME/mind/')` finds every page that references a concept.
3. `recall(session_id=...)` — drill into one promising hit for full metadata + artifacts + the conversation head/tail.

Try synonyms, related styles, artist names. If relevant prior work exists, surface it: *"You explored something similar when you worked on X — want to build from that?"*

### Apply creative strategies

Skip this step when the brief is already specific. Otherwise push past the obvious direction before research begins, with whatever technique fits the brief — there is no fixed set.

Strategies that worked for this user live in the mind wiki as pages tagged `strategy`; `recall(query="strategy")` lists them, and each page's `## Uses` section records where it was applied and whether the user kept the result. Weigh those, but don't limit yourself to them.

The goal: a direction that's specific *and* surprising.

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

Pattern — sub-agents write briefs to disk, parent reads + synthesizes.
Build the strand list conceptually — for a "noir" brief: `films` (1940s
film noir cinematography), `comics` (noir illustration / comic art
lineage), `photo` (contemporary noir photographers) — then make one
direct `launch_agent` tool call per strand:

```
launch_agent(
    task="Research 1940s film noir cinematography — lighting, framing, "
         "B&W tonality. Cite 3-5 specific films with shot examples.\n\n"
         "Use web_search / web_fetch. Produce a tight 200-word brief. "
         "Save it as markdown to: $OKONOMI_SESSION_DIR/research_strand_films.md\n"
         "End by printing the file path so the parent knows where to find it.",
    description="noir/films",
)
launch_agent(
    task="Research noir illustration / comic art lineage (Sin City, Eisner, "
         "etc.). Line-work conventions, panel composition.\n\n"
         "Use web_search / web_fetch. Produce a tight 200-word brief. "
         "Save it as markdown to: $OKONOMI_SESSION_DIR/research_strand_comics.md\n"
         "End by printing the file path so the parent knows where to find it.",
    description="noir/comics",
)
launch_agent(
    task="Research contemporary noir photographers. Modern reinterpretations "
         "of the noir aesthetic.\n\n"
         "Use web_search / web_fetch. Produce a tight 200-word brief. "
         "Save it as markdown to: $OKONOMI_SESSION_DIR/research_strand_photo.md\n"
         "End by printing the file path so the parent knows where to find it.",
    description="noir/photo",
)
```

After launching, end your reply. Each sub-agent's closing message
arrives in a system note when it finishes; do not poll `list_agents()`
for it. Once every strand has reported, read the strand files,
synthesize them into the final concept brief, and proceed to
generation as usual. Sub-agents share the session dir,
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

File results into the mind wiki with the `remember` tool — the only way pages are written (file tools refuse the mind directory). The wiki is a flat set of interlinked pages, not monolithic reports. Two kinds of page come out of a research run:

1. **Concept pages** — one per distinct concept the research surfaced: an aesthetic, an artist/film/work reference, a recurring motif, a technique. The slug is the kebab-case concept name (`sodium-vapour-night`).
2. **The synthesis page** — the concept brief / prompt package itself, as its own page that links out to the concept pages it draws on. It stays **self-contained** (someone reading it cold should understand the concept, references, and direction) while its `[[links]]` connect it into the wiki.

**Update before create.** `recall(query='<concept>')` for each concept first. If a mind page comes back, `read_file` it, merge the new findings into its body, and write it back with its slug and `update=True` — never write a duplicate page for the same concept.

```python
remember(
    title="Sodium-Vapour Night",
    slug="sodium-vapour-night",
    tags=["aesthetic", "night", "colour"],
    related=["grain", "theyyam-portraits"],
    body="Prose: what this is, what makes it work, where it came from. Reference related "
         "concepts inline as [[grain]] — the links are the graph; link generously to pages "
         "that exist or should exist.",
)
```

`remember` writes the frontmatter (`title`, `tags`, `created`, `updated`, `related`) and the title heading; pass only the body.

**Record the strategy.** If a creative strategy produced the direction the user chose (or rejected), add an entry to that strategy page's `## Uses` section — project, what was combined or flipped, and the outcome — via the same update-before-create path. A strategy with no page yet gets a new one tagged `strategy`. Link the synthesis page to it.

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
| `visual-language` | Composition (rule of thirds, golden ratio, leading lines, negative space), color harmonies (mono, analogous, complementary, split-comp, triadic), lighting (hard/soft, Rembrandt, butterfly, split, rim), texture vocabulary, art-movement glossary (Impressionism → Vaporwave → Solarpunk). |
| `output-formats` | Templates for concept brief, prompt package, exploration document, photography direction. |
