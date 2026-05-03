---
name: concept-research
description: "Deep creative research and conceptual exploration upstream of media creation. Develops rich, well-researched visual concepts before any pixels are made — for AI image gen, photography, illustration, concept art, or video. Covers: brainstorming visual ideas, mood boards, style/movement exploration, reference mapping, concept briefs, prompt packages, palette/composition analysis, and art history deep dives. Trigger when users want to: brainstorm visuals, explore a style, build a mood board, develop a concept brief, craft image prompts, plan a photoshoot, flesh out a vague idea, or when they say 'I want something that looks like X.'"
metadata:
  tags: [concept, research, ideation, visual-language, prompt-crafting, mood-board, art-direction]
  related_skills: [video-generation]
---

# Concept Research

Great images start with great thinking. Move from a vague feeling to a concrete, actionable visual concept.

## Creative Standard

**Ask before assuming.** The first description is often a placeholder for something deeper. Surface the real intent with targeted questions.

**Research before asserting.** Verify with a search rather than relying on memory — especially for non-Western traditions, contemporary artists, and niche movements.

**Name real things.** Real artists, films, movements, techniques. A confidently wrong reference is worse than none.

**Show, don't just describe.** Use image search to surface visual examples alongside written descriptions.

**Think in systems.** A good brief describes a visual *system* that could generate many images, not just one.

## Process

### 1. Listen and Clarify

Understand what the user is reaching for. Surface as needed:

| Dimension | Question |
|-----------|----------|
| **Intent** | What's it for? |
| **Destination** | Which medium? (Flux, photography, video, illustration) |
| **Emotional target** | What should the viewer feel? |
| **References** | What do they like? What do they *not* want? |
| **Constraints** | Aspect ratio, brand guidelines, medium limits |

Infer what you can, ask about the rest. Bias toward asking rather than assuming.

### 2. Check Prior Research

Before starting fresh, search for existing work:

**Saved research files:**
1. `search(pattern="<keyword>", path="$OKONOMI_HOME/research/", file_glob="*.md")` — style names, artist references, themes, moods
2. `run(command="ls $OKONOMI_HOME/research/")` — overview of what exists
3. Read promising files to assess relevance

**Past sessions:**
1. `search(pattern="<keyword>", path="~/.okonomi/sessions/")` — grep across all session data
2. `search(pattern="<keyword>", path="~/.okonomi/sessions/", file_glob="*.json")` — narrow to metadata

Iterate on search — try synonyms, related styles, artist names. Exhaust what's been explored before duplicating. If relevant prior work exists, surface it: "You explored something similar when you worked on X — want to build from that?"

### 3. Apply Creative Strategies

Push the concept beyond the obvious before research begins. See `references/creative-strategies.md` for the full techniques:

| Strategy | Use When |
|----------|----------|
| **Conceptual Blending** | Two domains can merge (Baroque + brutalism) |
| **Replacement Template** | Symbolic substitution reveals hidden qualities |
| **Forced Connections** | Need unexpected angles from distant domains |
| **Assumption Reversal** | Flip fundamental premises to find new possibilities |
| **Distance Associations** | Walk from obvious to surprising via semantic network |
| **Perspective Shifting** | View through different lenses, scales, or timeframes |

Use one or combine several. The goal: a direction that's specific and surprising.

### 4. Research and Explore

Let the concept's needs drive tool usage, not a fixed checklist. Err toward more research.

**Tools:**
- **Web search**: Artists, movements, techniques, cultural context, contemporary trends
- **Image search**: Visual examples of styles, palettes, compositions, lighting
- **Web fetch**: Full articles when search snippets aren't enough
- **User files**: Mood boards, brand guidelines, previous briefs from connected tools

**Research threads (pursue as relevant):**

- **Visual language analysis** — What makes a style work? Composition patterns, color relationships, texture, light behavior. See `references/visual-language.md` for vocabulary.
- **Reference mapping** — Find specific artists, photographers, films, works. Describe what each reference *contributes*, not just "like this."
- **Cross-pollination** — Unexpected connections across disciplines. A cyberpunk scene might learn from ukiyo-e composition.
- **User context integration** — Existing materials represent decisions already made and taste already established.

### 5. Synthesize and Structure

Choose the output format that fits. See `references/output-formats.md` for full templates:

| Format | Use When |
|--------|----------|
| **Concept Brief** | Complex or multi-image projects needing cohesive direction |
| **Prompt Package** | AI image generation — a system for iteration, not just one prompt |
| **Exploration Document** | Open-ended concepts with multiple possible directions |
| **Photography Direction** | Planned shoots with shot lists and lighting direction |

### 6. Save the Research

Always save output as markdown to `$OKONOMI_HOME/research/` with a descriptive filename (e.g., `solarpunk-cityscape-concept-brief.md`). The file should be self-contained — someone reading it cold should understand the concept, references, and direction without needing the conversation.

## Reference Dimensions

Quick lookup for visual decision-making. Full tables in `references/visual-language.md`:

| Dimension | Key Options | Reference |
|-----------|-------------|-----------|
| **Composition** | Rule of thirds, golden ratio, centered symmetry, leading lines, negative space | `references/visual-language.md` § Composition |
| **Color harmony** | Monochromatic, analogous, complementary, split-complementary, triadic | `references/visual-language.md` § Color |
| **Lighting** | Hard/soft, front/side/back, Rembrandt, butterfly, split, rim | `references/visual-language.md` § Lighting |
| **Texture** | Rough, smooth, organic, geometric, translucent, reflective, gritty | `references/visual-language.md` § Texture |
| **Art movements** | Impressionism, Art Deco, Surrealism, Brutalism, Vaporwave, Solarpunk | `references/visual-language.md` § Movements |
| **Creative strategies** | Conceptual blending, forced connections, assumption reversal | `references/creative-strategies.md` |

## Principles

- **Specific over generic.** "Warm golden hour light raking across textured concrete at a low angle" beats "nice warm lighting."
- **Go to the source.** Fetch full articles when snippets aren't enough — depth is in the details.
- **Explain the why.** Not just "use complementary colors" — explain what it does to the viewer's eye and why it serves this concept.
- **Respect the user's taste.** Research and suggest, don't override.
- **Match the medium.** Search for current best practices — prompt syntax and model capabilities change frequently.
- **Scale to complexity.** A quick prompt needs one search. A visual world needs multi-threaded research across all tools.
