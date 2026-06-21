
# Video Generation Pipeline

## Creative Standard

Video is a time-based medium — every frame must serve the motion and the story. Before generating a single clip, decide what the camera reveals, how the subject moves, and what the viewer should feel.

**Descriptive, not instructional.** Video prompts describe a scene unfolding — never tell the model what to do.

Bad: "Make the water move and add some wind"
Good: "A serene lake with gentle ripples spreading across the surface, soft wind rustling through willow branches"

**One motion per clip.** Each generation should have a single, clear camera movement. Combining dolly + pan + zoom produces confused, unnatural results.

**Lighting sells realism.** Lighting direction matters more than subject detail. Always specify it.

## Two-tier doctrine

Video generation in okonomi runs through **exactly two tiers**: ideation (cheap, fast, sweep many motion takes) and refinement (slower, more expensive, the hero clip). Don't escalate to refinement until ideation has produced motion the user wants to keep — video is the most expensive thing this agent does.

### Tier 1 — Ideation (fast, cheap, sweep many motion takes)

Use this for motion exploration, "show me 3 ways the camera could move", motion previews, and rough cuts. Run with multiple seeds or short durations to compare directions.

- `fal-ai/ltx-2.3-22b/image-to-video/lora` — fast, low-cost. Optional LoRA support for stylized motion. Strong default for "does this motion idea work" checks.

Takes an `image_url` (source frame) and a `prompt` (motion description). Keep durations short during ideation — 3–5 seconds is enough to read whether the motion concept works. **Don't pass a `loras` parameter by default** — call as base unless the user has explicitly asked for one (same convention as the image-generation skill).

### Tier 2 — Refinement (the hero clip, slower and pricier)

Once the user has picked the motion direction, generate the final clip on the higher-quality endpoint:

- `bytedance/seedance-2.0/image-to-video` — full Seedance 2.0 (not the `/fast` variant). Best motion realism, longer-form-friendly, strong on cinematic camera moves and character consistency.

Don't reach outside these tiers for first-attempt video work. If the user has a special request the tiers can't cover (synced audio with Veo, lip-sync, niche style), call `genmedia.search("<keyword>")` to discover alternatives — but document the rationale before spending on it.

### Workflow

1. **IDEATE** — `fal-ai/ltx-2.3-22b/image-to-video/lora`, short duration (3–5s). Run a small sweep — 2–3 takes with different motion prompts is usually enough to find the direction.
2. **REFINE** — take the chosen ideation clip's source image (or its last frame for continuation) and re-run on `bytedance/seedance-2.0/image-to-video`. Single take, full duration.
3. **STITCH** — combine multiple Tier 2 clips into the final video with `ffmpeg` (see `references/post-production.md`).

The cost asymmetry is severe — a Tier 2 clip can cost 10–30× a Tier 1 take. Sweep cheaply, then commit once.

## Modes

| Mode | Input | Output | Endpoint shape |
|------|-------|--------|----------------|
| **Image-to-video** | Single source image | Animated clip with motion | `genmedia.run("<i2v endpoint>", image_url=..., prompt=..., duration=...)` |
| **Frame transition** | First + last frame images | Video bridging two states | `genmedia.run("<first-last endpoint>", first_image_url=..., last_image_url=..., prompt=...)` |
| **Scene production** | Concept → hero frame → variants | Multi-scene storyboard | `genmedia.run("fal-ai/krea/v2/large/text-to-image", ...)` (see `image-generation` skill) → image-to-video |
| **Assembly** | Multiple clips | Stitched final video with transitions and audio | `bash(command="ffmpeg ...")` (see `references/post-production.md`) |

## Prompt Anatomy

Build prompts by layering seven elements. Not all are required, but specificity improves results.

| Element | What It Controls | Reference |
|---------|-----------------|-----------|
| **Shot framing & camera motion** | How the camera frames and moves | `references/camera-and-framing.md` |
| **Visual style** | Medium, film stock, aesthetic | `references/visual-style.md` § Film Stocks |
| **Lighting** | Illumination, mood, atmosphere | `references/visual-style.md` § Lighting |
| **Character descriptions** | Physical traits, clothing, expression | — |
| **Location & environment** | Setting, time, weather, textures | — |
| **Action & motion** | What unfolds in the scene | — |
| **Audio & sound** (Veo 3.1) | Dialogue, ambient, SFX, music | — |

### Character Descriptions
Specific beats generic. Layer physical traits, clothing, expression, and motion.

Weak: "A man walks through a city"
Strong: "A tall man in his thirties with a close-cropped beard, wearing a rumpled linen suit, walks briskly through narrow cobblestone streets, glancing over his shoulder"

### Location & Environment
Build the world with sensory detail — textures, atmosphere, scale.

Weak: "A jazz club"
Strong: "A dimly lit basement jazz club, blue cigarette smoke hanging in spotlights, exposed brick walls lined with framed black-and-white photographs, a small stage with a worn upright piano"

### Action & Motion
Describe what happens as a continuous unfolding, not a command.

Weak: "The bird flies away"
Strong: "A heron launches from the riverbank, wings spreading wide, water droplets falling from its legs as it climbs into a pale morning sky"

### Audio & Sound (Veo 3.1)
Use separate sentences to describe audio. Veo 3.1 generates synchronized sound.

- **Dialogue**: "The man in the red hat says: Where is the rabbit?"
- **Ambient**: "The hum of distant traffic, crickets chirping"
- **SFX**: "Footsteps echoing on marble, a glass clinking"
- **Music**: "A slow jazz piano melody playing softly in the background"

## Prompt Templates

### Image-to-Video
```
[Camera movement], [lighting conditions]. [Subject description] [action/motion].
[Environmental details in motion]. [Atmospheric elements].
```

Example: "Slow dolly forward, warm golden hour lighting. A woman in a flowing white dress stands at the edge of a cliff, her hair and dress billowing in a strong coastal wind. Waves crash against rocks below, sending spray into the amber light. Seabirds glide past in the distance."

### First-Last Frame Transition
```
[Opening state description]. [Transition action — what changes and how].
[Ending state description]. [Continuous elements that persist throughout].
```

Example: "A closed flower bud in soft morning light. Petals slowly unfurl in timelapse, dewdrops catching prismatic light as they roll off opening petals. The fully bloomed flower sways gently, a bee approaching from the right. Soft bokeh background throughout."

### Cinematic Narrative
```
[Visual style and medium]. [Establishing shot/framing].
[Character introduction with physical detail]. [Action sequence].
[Environmental reaction]. [Emotional beat/resolution].
[Sound design notes].
```

## Production Workflow

The most effective approach: **build scenes as stills, then animate**. Match each step to its tier.

### Step 1: Generate the Hero Frame (image-generation skill)
Create the primary scene with Krea: sweep with `genmedia.run("fal-ai/krea/v2/medium/turbo/text-to-image", ...)`, then render the chosen take at full fidelity with `genmedia.run("fal-ai/krea/v2/large/text-to-image", ...)`. Refine composition, lighting, and character design until the still is right. See the `image-generation` skill for the two-tier doctrine on stills.

### Step 2: Build Scene Variants (image-editing skill)
Use `genmedia.run("openai/gpt-image-2/edit", image_url=..., prompt=...)` to derive scene variants from the hero — change time of day, swap backgrounds, alter expressions. This gives a storyboard of consistent frames without regenerating from scratch.

### Step 3: Ideate Motion (Tier 1)
For each scene, run a short Tier 1 take to test whether the motion idea works:

```python
ideation = genmedia.run(
    "fal-ai/ltx-2.3-22b/image-to-video/lora",   # Tier 1 — fast & cheap
    image_url=hero_url,
    prompt="Camera slowly dollies forward, rain falling through the cone of lamplight, the detective's coat rippling in the wind",
    duration=4,
)
```

Compare a couple of motion prompts before committing.

### Step 4: Refine the Hero Clip (Tier 2)
Once the motion direction is right, regenerate at Tier 2 with full Seedance 2.0:

```python
hero_clip = genmedia.run(
    "bytedance/seedance-2.0/image-to-video",   # Tier 2 — refinement
    image_url=hero_url,
    prompt=(
        "Camera slowly dollies forward, rain falling through the cone of lamplight, "
        "the detective's coat rippling in the wind. Distant thunder rolls. "
        "Footsteps echo on wet cobblestone."
    ),
    duration=8,
)
```

### Step 5: Bridge Scenes with Transitions
For first-last-frame transitions, run on the same Tier 2 endpoint when the result is part of the final cut, or Tier 1 if you're testing the transition concept first:

```python
transition = genmedia.run(
    "bytedance/seedance-2.0/image-to-video",   # Tier 2 transition for the final cut
    first_image_url=hero_url,
    last_image_url=scene_office_url,
    prompt="The detective turns and walks through a doorway, the rain-soaked street dissolving into a warm dim office interior",
)
```

### Step 6: Stitch & Finish
Combine clips into a final video with ffmpeg via the standalone `bash` tool. Add crossfades, audio, color grading. See `references/post-production.md` for the full ffmpeg toolkit.

## Common Mistakes

- **Instructional language**: "Make it zoom in" → "Camera slowly zooms into..."
- **Vague descriptions**: "A cool scene" → describe exactly what makes it cool
- **Conflicting camera movements**: One primary motion per clip
- **Missing lighting**: Always specify — it drives mood more than subject detail
- **No atmosphere**: Add particles — dust, fog, rain, floating embers
- **Overloaded prompts**: Each clip should be one coherent moment
- **Skipping stills**: Always refine the frame before animating it
- **Expensive re-runs**: If a generation fails on JSON parsing, `print(result)` to inspect — don't regenerate

## Style Quick Reference

| Intent | Prompt Fragment |
|--------|----------------|
| Dreamy/ethereal | "Soft focus, diffused backlight, slow motion, pastel color palette" |
| Gritty/realistic | "Handheld camera, desaturated colors, harsh available light, film grain" |
| Epic/grand | "Wide aerial shot, dramatic orchestral tone, golden hour, vast landscape" |
| Intimate/personal | "Tight close-up, shallow depth of field, warm practical lighting, quiet" |
| Suspenseful | "Low angle, deep shadows, slow dolly forward, muted color palette" |
| Joyful/energetic | "Bright saturated colors, dynamic tracking shot, fast motion, lens flare" |
| Nostalgic | "Super 8 film texture, warm color grading, slightly overexposed, soft vignette" |
| Futuristic | "Clean digital, cool blue-teal palette, volumetric lighting, reflective surfaces" |

For detailed lookup tables: `references/camera-and-framing.md`, `references/visual-style.md`, `references/post-production.md`.
